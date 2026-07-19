import argparse
import gc
import json
import os.path
from pathlib import Path

import tensorflow
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from utils.util import *

plt.switch_backend('agg')

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

CHECKPOINTS_FOLDER = "checkpoints"
CHECKPOINT_NAME = "AC_GAN_model"
LAST_CHECKPOINT_NAME = f"{CHECKPOINT_NAME}_last_model"
TRAINING_METRICS_FILE_NAME = "training_metrics.csv"


# Make Generator Model
def build_generator(latent_dim: int, num_classes: int, number_of_genotypes: int, alph: float):
    generator = Sequential()
    generator.add(
        Dense(int(number_of_genotypes // 1.3), input_shape=(latent_dim + NUMBER_REPEAT_CLASS_VECTOR * num_classes,),
              kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(negative_slope=alph))
    generator.add(Dense(int(number_of_genotypes // 1.2), kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(negative_slope=alph))
    generator.add(Dense(int(number_of_genotypes // 1.1), kernel_regularizer=regularizers.l2(0.0001)))
    generator.add(LeakyReLU(negative_slope=alph))
    generator.add(Dense(number_of_genotypes, activation='tanh'))

    # Generating the output image
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(num_classes,), dtype='float32')
    z = tensorflow.keras.layers.concatenate([label, label, label, label, noise, label, label, label, label])

    sequence = generator(z)

    return Model([noise, label], sequence)


# Make Discriminator Model
def build_discriminator(number_of_genotypes: int, num_classes: int, alph: float, d_activation: str):
    # Define the sequence input

    discriminator = Sequential()
    discriminator.add(
        Dense(number_of_genotypes // 2, input_shape=(number_of_genotypes,), kernel_regularizer=regularizers.l2(0.0001)))

    discriminator.add(LeakyReLU(negative_slope=alph))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(number_of_genotypes // 3, kernel_regularizer=regularizers.l2(0.0001)))
    discriminator.add(LeakyReLU(negative_slope=alph))
    discriminator.add(Dropout(0.2))
    discriminator.add(Dense(number_of_genotypes // 4, kernel_regularizer=regularizers.l2(0.0001)))
    discriminator.add(LeakyReLU(negative_slope=alph))
    discriminator.add(Dropout(0.1))
    sequence = Input(shape=(number_of_genotypes,), dtype='float32')

    # Extract features from images
    features = discriminator(sequence)

    # Building the output layer
    validity = Dense(1, activation=d_activation)(features)
    label = Dense(num_classes, activation="softmax")(features)

    return Model(sequence, [validity, label])


# Make AC-GAN Model
def build_acgan(generator, discriminator):
    for layer in discriminator.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    acgan_output = discriminator(generator.output)
    acgan = Model(generator.input, acgan_output)
    return acgan


def get_checkpoints_path(experiment_results_path: str):
    return os.path.join(experiment_results_path, CHECKPOINTS_FOLDER)


def get_checkpoint_sequences_path(checkpoints_path: str):
    return os.path.join(checkpoints_path, "synthetic_sequences")


def build_checkpoint(generator: Model, discriminator: Model, acgan: Model):
    checkpoint = tf.train.Checkpoint(epoch=tf.Variable(0, dtype=tf.int64),
                                     generator=generator,
                                     discriminator=discriminator,
                                     acgan=acgan,
                                     discriminator_optimizer=discriminator.optimizer,
                                     acgan_optimizer=acgan.optimizer)
    return checkpoint


def checkpoint_file_exists(checkpoint_path: str):
    return os.path.exists(f"{checkpoint_path}.index")


def restore_latest_checkpoint(checkpoint, checkpoints_path: str):
    last_checkpoint_path = os.path.join(checkpoints_path, LAST_CHECKPOINT_NAME)
    if checkpoint_file_exists(last_checkpoint_path):
        checkpoint.restore(last_checkpoint_path).expect_partial()
        checkpoint_epoch = int(checkpoint.epoch.numpy())
        print(f"Loaded checkpoint from epoch {checkpoint_epoch}: {last_checkpoint_path}")
        return checkpoint_epoch

    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoints_path)
    if latest_checkpoint_path is None:
        return None

    checkpoint.restore(latest_checkpoint_path).expect_partial()
    checkpoint_epoch = int(checkpoint.epoch.numpy())
    print(f"Loaded checkpoint from epoch {checkpoint_epoch}: {latest_checkpoint_path}")
    return checkpoint_epoch


def save_checkpoints(
        checkpoint,
        checkpoints_path: str,
        epoch: int,
        generator: Model = None,
        discriminator: Model = None):
    epoch_checkpoint_path = os.path.join(checkpoints_path, f"{CHECKPOINT_NAME}_{epoch}")
    last_checkpoint_path = os.path.join(checkpoints_path, LAST_CHECKPOINT_NAME)

    # TensorFlow object checkpoints are intentionally disabled here. The attack
    # loader now prefers explicit standalone Keras component models.
    # checkpoint.epoch.assign(epoch)
    # checkpoint.write(epoch_checkpoint_path)
    # checkpoint.write(last_checkpoint_path)
    #
    # print(f"Saved checkpoint: {epoch_checkpoint_path}")
    # print(f"Saved latest checkpoint: {last_checkpoint_path}")

    if generator is not None:
        epoch_generator_model_path = os.path.join(
            checkpoints_path,
            f"{CHECKPOINT_NAME}_{epoch}_generator.keras",
        )
        last_generator_model_path = os.path.join(
            checkpoints_path,
            f"{LAST_CHECKPOINT_NAME}_generator.keras",
        )
        epoch_generator_weights_path = os.path.join(
            checkpoints_path,
            f"{CHECKPOINT_NAME}_{epoch}_generator.weights.h5",
        )
        last_generator_weights_path = os.path.join(
            checkpoints_path,
            f"{LAST_CHECKPOINT_NAME}_generator.weights.h5",
        )
        generator.save(epoch_generator_model_path)
        generator.save(last_generator_model_path)
        generator.save_weights(epoch_generator_weights_path)
        generator.save_weights(last_generator_weights_path)
        print(f"Saved generator model: {epoch_generator_model_path}")
        print(f"Saved latest generator model: {last_generator_model_path}")
        print(f"Saved generator weights: {epoch_generator_weights_path}")
        print(f"Saved latest generator weights: {last_generator_weights_path}")

    if discriminator is not None:
        epoch_discriminator_model_path = os.path.join(
            checkpoints_path,
            f"{CHECKPOINT_NAME}_{epoch}_discriminator.keras",
        )
        last_discriminator_model_path = os.path.join(
            checkpoints_path,
            f"{LAST_CHECKPOINT_NAME}_discriminator.keras",
        )
        epoch_discriminator_weights_path = os.path.join(
            checkpoints_path,
            f"{CHECKPOINT_NAME}_{epoch}_discriminator.weights.h5",
        )
        last_discriminator_weights_path = os.path.join(
            checkpoints_path,
            f"{LAST_CHECKPOINT_NAME}_discriminator.weights.h5",
        )
        discriminator.save(epoch_discriminator_model_path)
        discriminator.save(last_discriminator_model_path)
        discriminator.save_weights(epoch_discriminator_weights_path)
        discriminator.save_weights(last_discriminator_weights_path)
        print(f"Saved discriminator model: {epoch_discriminator_model_path}")
        print(f"Saved latest discriminator model: {last_discriminator_model_path}")
        print(f"Saved discriminator weights: {epoch_discriminator_weights_path}")
        print(f"Saved latest discriminator weights: {last_discriminator_weights_path}")


def average_discriminator_score(discriminator: Model, x_values, batch_size: int = 64):
    if x_values is None or len(x_values) == 0:
        return None
    validity_scores, class_scores = discriminator.predict(x_values, batch_size=batch_size, verbose=0)
    average_score = float(np.average(validity_scores))
    del validity_scores, class_scores
    return average_score


def get_checkpoint_discriminator_scores(discriminator: Model, train_dataset: tuple, test_dataset,
                                        generated_genomes_df, batch_size: int):
    train_score = average_discriminator_score(discriminator, train_dataset[0], batch_size=batch_size)
    eval_score = average_discriminator_score(discriminator, test_dataset[0],
                                             batch_size=batch_size) if test_dataset is not None else None
    synthetic_values = generated_genomes_df.drop(columns=['Type']).values
    synth_score = average_discriminator_score(discriminator, synthetic_values, batch_size=batch_size)
    del synthetic_values
    return {
        "train_score": train_score,
        "eval_score": eval_score,
        "synth_score": synth_score,
    }


def print_checkpoint_discriminator_scores(epoch: int, discriminator: Model, train_dataset: tuple, test_dataset,
                                          generated_genomes_df, batch_size: int):
    scores = get_checkpoint_discriminator_scores(discriminator, train_dataset, test_dataset, generated_genomes_df,
                                                 batch_size)

    score_parts = [f"train={scores['train_score']:.4f}", f"synth={scores['synth_score']:.4f}"]
    if scores["eval_score"] is not None:
        score_parts.insert(1, f"eval={scores['eval_score']:.4f}")
    print(f"Checkpoint epoch {epoch} discriminator average scores: " + ", ".join(score_parts))
    return scores


def save_training_metrics(experiment_results_path: str, epoch: int, discriminator_loss: float, generator_loss: float,
                          discriminator_scores: dict, generated_genomes_df, pca_metrics: dict = None):
    metrics_path = os.path.join(experiment_results_path, TRAINING_METRICS_FILE_NAME)
    write_header = not os.path.exists(metrics_path)
    memory_metrics = get_memory_metrics()
    metrics = {
        "epoch": epoch,
        "discriminator_loss": discriminator_loss,
        "generator_loss": generator_loss,
        "train_discriminator_score": discriminator_scores["train_score"],
        "eval_discriminator_score": discriminator_scores["eval_score"],
        "synthetic_discriminator_score": discriminator_scores["synth_score"],
        "synthetic_samples": len(generated_genomes_df),
        **(pca_metrics or {}),
        **memory_metrics,
    }
    pd.DataFrame([metrics]).to_csv(metrics_path, mode='a', header=write_header, index=False)
    print(f"Saved training metrics: {metrics_path}")


def get_memory_metrics():
    metrics = {
        "cpu_rss_mb": None,
        "gpu_current_mb": None,
        "gpu_peak_mb": None,
    }
    if psutil is not None:
        metrics["cpu_rss_mb"] = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2

    gpu_devices = tf.config.list_logical_devices('GPU')
    if gpu_devices:
        try:
            gpu_name = gpu_devices[0].name.replace('/device:', '')
            gpu_memory = tf.config.experimental.get_memory_info(gpu_name)
            metrics["gpu_current_mb"] = gpu_memory["current"] / 1024 ** 2
            metrics["gpu_peak_mb"] = gpu_memory["peak"] / 1024 ** 2
        except (AttributeError, ValueError, RuntimeError):
            pass
    return metrics


def train(batch_size: int, epochs: int, dataset: tuple, num_classes: int, latent_size: int,
          generator: Model, discriminator: Model, acgan: Model, save_number: int, class_id_to_counts: dict,
          experiment_results_path: str, id_to_class: dict, real_class_names: list, sequence_results_path: str,
          checkpoint, checkpoints_path: str, test_dataset, synthetic_samples_number: int, start_epoch: int = 0):
    """
    genome-ac-gan training process
    :param batch_size: int batch size training
    :param epochs: int number of epochs
    :param dataset: tuple of 2 np.arrays (x, y) sequences of genotypes and labels
    :param num_classes: number of classes in Y_true
    :param latent_size: input latent size for noise(z)
    :param generator: generator model
    :param discriminator: discriminator model
    :param acgan: acgan model
    :param save_number: how many epochs between saving the temp results
    :param class_id_to_counts: map of class translate to id
    :param experiment_results_path: output folder results path
    :param id_to_class: translation of id to class name
    :param real_class_names: all Y_true labels names
    :param sequence_results_path: path under checkpoints that will contain the synthetic sequences
    :param test_dataset: dataset to evaluate the classifier
    """
    class_metric_results = None
    y_real, y_fake = np.ones([batch_size, 1]), np.zeros([batch_size, 1])
    losses = []
    train_dataset = tensorflow.data.Dataset.from_tensor_slices(dataset).shuffle(
        dataset[0].shape[0], reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    # Training iteration
    for e in range(start_epoch, epochs + 1):
        avg_d_loss, avg_g_loss = [], []
        for x_batch_real, Y_batch_real in train_dataset:
            x_batch_real_with_noise = add_noise_real_batch(x_batch_real)
            # x_batch_real_with_noise = x_batch_real - np.random.uniform(0, 0.1, size=(
            #     x_batch_real.shape[0], x_batch_real.shape[1]))
            latent_samples = np.random.normal(loc=0, scale=1,
                                              size=(batch_size, latent_size))  # create noise to be input to generator
            fake_labels_batch = tensorflow.one_hot(
                tensorflow.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tensorflow.int32),
                depth=num_classes)

            X_batch_fake = generator.predict_on_batch([latent_samples, fake_labels_batch])

            x_batch_train = tf.concat(
                [tf.convert_to_tensor(X_batch_fake, dtype=tf.float32), tf.cast(x_batch_real_with_noise, tf.float32)],
                axis=0)
            y_valid_batch_train = tf.concat([y_fake, y_real - np.random.uniform(0, 0.1, size=(
                y_real.shape[0], y_real.shape[1]))], axis=0)
            y_valid_batch_train = get_smoothing_label_batch(tf.cast(y_valid_batch_train, tf.float32))
            y_class_batch_train = tf.concat([fake_labels_batch, tf.cast(Y_batch_real, tf.float32)], axis=0)
            y_batch_true = tf.concat([y_valid_batch_train, y_class_batch_train], axis=1)
            discriminator_batch_train_dataset = tensorflow.data.Dataset.from_tensor_slices(
                (x_batch_train, y_batch_true)).shuffle(
                x_batch_train.shape[0]).batch(int(batch_size), drop_remainder=True)
            d_loss = []
            discriminator.trainable = True
            for x_mini_batch, Y_mini_batch in discriminator_batch_train_dataset:
                y_valid_mini_batch = Y_mini_batch[:, 0]
                y_class_mini_batch = Y_mini_batch[:, 1:]
                d_loss_mini_batch = discriminator.train_on_batch(x_mini_batch, [y_valid_mini_batch,
                                                                                y_class_mini_batch],
                                                                 return_dict=True)
                d_loss.append(d_loss_mini_batch["loss"])
            discriminator.trainable = False
            latent_samples = np.random.normal(loc=0, scale=1,
                                              size=(batch_size, latent_size))  # create noise to be input to generator
            fake_labels_batch = tensorflow.one_hot(
                tensorflow.random.uniform((batch_size,), minval=0, maxval=num_classes, dtype=tensorflow.int32),
                depth=num_classes)
            g_loss = acgan.train_on_batch([latent_samples, fake_labels_batch],
                                          [get_smoothing_label_batch(tf.cast(y_real, tf.float32)), fake_labels_batch],
                                          return_dict=True)

            avg_d_loss.extend(d_loss)
            avg_g_loss.append(g_loss["loss"])
            del (x_batch_real, Y_batch_real, x_batch_real_with_noise, latent_samples, fake_labels_batch,
                 X_batch_fake, x_batch_train, y_valid_batch_train, y_class_batch_train, y_batch_true,
                 discriminator_batch_train_dataset, d_loss, x_mini_batch, Y_mini_batch, y_valid_mini_batch,
                 y_class_mini_batch, d_loss_mini_batch, g_loss)
        discriminator_loss = np.average(avg_d_loss)
        generator_loss = np.average(avg_g_loss)
        losses.append((discriminator_loss, generator_loss))

        print("Epoch:\t%d/%d Discriminator loss: %6.4f Generator loss: %6.4f" % (
            e, epochs, discriminator_loss, generator_loss))
        should_save_checkpoint = save_number == 1 or e % save_number == 0 or e == epochs
        should_save_training_metrics = should_save_checkpoint
        if should_save_checkpoint or should_save_training_metrics:
            if test_dataset is not None:
                # evaluate the classifier
                class_metric_results = save_discriminator_class_pred(discriminator, test_dataset,
                                                                     experiment_results_path, id_to_class,
                                                                     class_metric_results, e)

            generated_genomes_df = generate_fake_samples(class_id_to_counts=class_id_to_counts,
                                                         epoch_number=e,
                                                         generator=generator,
                                                         id_to_class=id_to_class,
                                                         latent_size=latent_size,
                                                         num_classes=num_classes,
                                                         sequence_results_path=sequence_results_path,
                                                         framework="tensorflow",
                                                         total_generated_samples=synthetic_samples_number)
            discriminator_scores = print_checkpoint_discriminator_scores(e, discriminator, dataset, test_dataset,
                                                                         generated_genomes_df, batch_size=batch_size)

            pca_metrics = None

            if should_save_checkpoint:
                save_checkpoints(
                    checkpoint,
                    checkpoints_path,
                    e,
                    generator=generator,
                    discriminator=discriminator,
                )
                save_models(generator=generator,
                            discriminator=discriminator,
                            acgan=acgan,
                            experiment_results_path=experiment_results_path,
                            suffix="_last_model")

                # plot PCA for all population and for each label
                pca_metrics = plot_pca_comparisons(generator=generator, epoch_number=e,
                                                   class_id_to_counts=class_id_to_counts,
                                                   experiment_results_path=experiment_results_path,
                                                   latent_size=latent_size, num_classes=num_classes, dataset=dataset,
                                                   id_to_class=id_to_class, real_class_names=real_class_names,
                                                   sequence_results_path=sequence_results_path,
                                                   total_generated_samples=synthetic_samples_number,
                                                   generated_genomes_df=generated_genomes_df)

            if should_save_training_metrics:
                save_training_metrics(experiment_results_path, e, discriminator_loss, generator_loss,
                                      discriminator_scores, generated_genomes_df, pca_metrics=pca_metrics)

            del generated_genomes_df, discriminator_scores, pca_metrics
            gc.collect()
        del avg_d_loss, avg_g_loss
        gc.collect()


def polyloss_ce(y_true, y_pred, epsilon=DEFAULT_EPSILON_LCE, alpha=DEFAULT_ALPH_LCE):
    """
    Polyloss-CE classification loss function that describes in the article:
    "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions"
    https://arxiv.org/pdf/2204.12511.pdf
    :param y_true: sequences input y true labels
    :param y_pred: sequences input y predictions labels
    :param epsilon: epsilon >=-1, penalty weight
    :param alpha: 1>=alpha>=0 smooth labels percentages
    :return: Polyloss-CE score
    """
    num_classes = y_true.get_shape().as_list()[-1]
    smooth_labels = y_true * (1 - alpha) + alpha / num_classes
    one_minus_pt = tensorflow.reduce_sum(smooth_labels * (1 - softmax(y_pred)), axis=-1)
    CE_loss = tensorflow.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=alpha,
                                                              reduction='none')
    CE = CE_loss(y_true, y_pred)
    Poly1 = CE + epsilon * one_minus_pt
    return Poly1


def train_genome_ac_model(hapt_genotypes_path: str, extra_data_path: str, experiment_name: str,
                          latent_size: int = DEFAULT_LATENT_SIZE, alph: float = DEFAULT_ALPH,
                          g_learn: float = DEFAULT_GENERATOR_LEARNING_RATE,
                          d_learn: float = DEFAULT_DISCRIMINATOR_LEARNING_RATE, epochs: int = DEFAULT_EPOCHS,
                          batch_size: int = DEFAULT_BATCH_SIZE, class_loss_weights: float = DEFAULT_CLASS_LOSS_WEIGHTS,
                          save_number: int = DEFAULT_SAVE_NUMBER,
                          synthetic_samples_number: int = DEFAULT_SYNTHETIC_SAMPLES_NUMBER,
                          minimum_samples: int = DEFAULT_MINIMUM_SAMPLES, target_column: str = DEFAULT_TARGET_COLUMN,
                          d_activation: str = DEFAULT_DISCRIMINATOR_ACTIVATION,
                          class_loss_function: str = DEFAULT_CLASS_LOSS_FUNCTION,
                          validation_loss_function: str = DEFAULT_VALIDATION_LOSS_FUNCTION,
                          with_extra_data: bool = False,
                          test_discriminator_classifier: bool = False,
                          test_dataset_path: str = "resource/test_AFR_pop.csv",
                          required_populations: list[str] = None,
                          resume_from_checkpoint: bool = True):
    experiment_results_path = os.path.join(DEFAULT_RESULTS_FOLDER, experiment_name)
    checkpoints_path = get_checkpoints_path(experiment_results_path)
    sequence_results_path = get_checkpoint_sequences_path(checkpoints_path)
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)
    Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
    Path(sequence_results_path).mkdir(parents=True, exist_ok=True)

    K.clear_session()
    init_gpus()
    target_column = " ".join(target_column.split("_"))
    required_populations = required_populations if required_populations is not None and len(
        required_populations) > 0 else None
    dataset, class_id_to_counts, num_classes, class_to_id = init_dataset(hapt_genotypes_path=hapt_genotypes_path,
                                                                         extra_data_path=extra_data_path,
                                                                         target_column=target_column,
                                                                         minimum_samples=minimum_samples,
                                                                         with_extra_data=with_extra_data,
                                                                         required_populations=required_populations
                                                                         )
    dataset = (dataset[0].astype(np.float32), dataset[1].astype(np.float32))
    # save class id map
    with open(os.path.join(experiment_results_path, 'class_id_map.json'), 'w') as f:
        json.dump(class_to_id, f)
    number_of_genotypes = dataset[0].shape[1]
    # save the reverse_dict for returning each id
    id_to_class = reverse_dict(class_to_id)
    print(f"dataset shapes: {dataset[0].shape, dataset[1].shape}")
    print(f"classes: {num_classes}, class_id_to_counts: {class_id_to_counts}")
    real_class_names = pd.DataFrame(np.argmax(dataset[1], 1))
    real_class_names = real_class_names[0].replace(id_to_class)
    real_class_names = 'Real_' + real_class_names
    real_class_names = list(real_class_names)

    #  build the Genome-AC-GAN including generator, discriminator and AC-GAN which is combine of both
    generator = build_generator(latent_dim=latent_size, num_classes=num_classes,
                                number_of_genotypes=number_of_genotypes, alph=alph)

    generator.compile(metrics=['accuracy'])
    discriminator = build_discriminator(number_of_genotypes=number_of_genotypes, num_classes=num_classes, alph=alph,
                                        d_activation=d_activation)

    if class_loss_function == "categorical_accuracy":
        class_loss_function = CategoricalAccuracy()
    if class_loss_function == "polyloss_ce":
        class_loss_function = polyloss_ce

    discriminator.compile(optimizer=RMSprop(learning_rate=d_learn),
                          loss=[validation_loss_function, class_loss_function],
                          loss_weights=[1, class_loss_weights],
                          metrics=['binary_accuracy', 'categorical_accuracy'])

    discriminator.trainable = False
    acgan = build_acgan(generator, discriminator)

    acgan.compile(optimizer=RMSprop(learning_rate=g_learn),
                  loss=[validation_loss_function, class_loss_function],
                  loss_weights=[1, class_loss_weights],
                  metrics=['binary_accuracy', 'categorical_accuracy'])

    checkpoint = build_checkpoint(generator=generator,
                                  discriminator=discriminator,
                                  acgan=acgan)
    checkpoint_epoch = None
    if resume_from_checkpoint:
        checkpoint_epoch = restore_latest_checkpoint(checkpoint, checkpoints_path)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0
    if checkpoint_epoch is not None and checkpoint_epoch >= epochs:
        print(f"Checkpoint already reached epoch {checkpoint_epoch}; requested epochs={epochs}. Nothing to train.")
        return

    # prepare test_dataset for classification compression
    test_dataset = prepare_test_and_fake_dataset(experiment_results_path,
                                                 test_path=test_dataset_path,
                                                 target_column=target_column) if test_discriminator_classifier else None

    train(batch_size=batch_size, epochs=epochs, dataset=dataset, num_classes=num_classes,
          latent_size=latent_size, generator=generator, discriminator=discriminator, acgan=acgan,
          save_number=save_number, class_id_to_counts=class_id_to_counts,
          experiment_results_path=experiment_results_path, id_to_class=id_to_class, real_class_names=real_class_names,
          sequence_results_path=sequence_results_path, checkpoint=checkpoint, checkpoints_path=checkpoints_path,
          test_dataset=test_dataset,
          synthetic_samples_number=synthetic_samples_number, start_epoch=start_epoch)


def parse_args():
    parser = argparse.ArgumentParser(description='GS-AC-GAN training parser')
    parser.add_argument('--hapt_genotypes_path', type=str, default=REAL_10K_SNP_1000G_PATH,
                        help='path to real input hapt file')
    parser.add_argument('--experiment_name', type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help='experiment name')
    parser.add_argument('--extra_data_path', type=str, default=REAL_EXTRA_DATA_PATH,
                        help='path to real extra data with classes file')
    parser.add_argument('--latent_size', type=int, default=DEFAULT_LATENT_SIZE,
                        help='input noise latent size')
    parser.add_argument('--alph', type=float, default=DEFAULT_ALPH, help='alpha value for LeakyReLU')
    parser.add_argument('--g_learn', type=float, default=DEFAULT_GENERATOR_LEARNING_RATE,
                        help='generator learning rate')
    parser.add_argument('--d_learn', type=float, default=DEFAULT_DISCRIMINATOR_LEARNING_RATE,
                        help='discriminator learning rate')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='initial batch size')
    parser.add_argument('--class_loss_weights', type=float, default=DEFAULT_CLASS_LOSS_WEIGHTS,
                        help='what is the weight to calculate the loss score for discriminator classes and generator classes')
    parser.add_argument('--save_number', type=int, default=DEFAULT_SAVE_NUMBER,
                        help='how often to save the results')
    parser.add_argument('--synthetic_samples_number', type=int, default=DEFAULT_SYNTHETIC_SAMPLES_NUMBER,
                        help='number of synthetic samples to generate at each checkpoint')
    parser.add_argument('--minimum_samples', type=int, default=DEFAULT_MINIMUM_SAMPLES,
                        help='what is the minimum samples that we find to include the class in the training process')
    parser.add_argument('--target_column', type=str, default=DEFAULT_TARGET_COLUMN,
                        help='class column name', choices=['Population_code', 'Population_name',
                                                           'Superpopulation_code', 'Superpopulation_name'])
    parser.add_argument('--d_activation', type=str, default=DEFAULT_DISCRIMINATOR_ACTIVATION,
                        help='discriminator validation activation function (real/fake)')
    parser.add_argument('--class_loss_function', type=str, default=DEFAULT_CLASS_LOSS_FUNCTION,
                        help='loss function between different classes')
    parser.add_argument('--validation_loss_function', type=str, default=DEFAULT_VALIDATION_LOSS_FUNCTION,
                        help='loss function between different real/fake')
    parser.add_argument('--with_extra_data', action='store_true', default=False,
                        help="don't need to load extra data")
    parser.add_argument('--test_discriminator_classifier', action='store_true', default=False,
                        help="if you want to test the classifier during the training")
    parser.add_argument('--test_dataset_path', type=str, default="resource/test_AFR_pop.csv",
                        help="path to evaluation/test dataset used by discriminator classifier")
    parser.add_argument('--required_populations', nargs='+', help='List of specific populations to filter')
    parser.add_argument('--resume_from_checkpoint', action=argparse.BooleanOptionalAction, default=True,
                        help='continue from the latest TensorFlow checkpoint when it exists')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    experiment_results_path = os.path.join(DEFAULT_RESULTS_FOLDER, args.experiment_name)
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)
    # save the args as a JSON file
    with open(os.path.join(experiment_results_path, 'experiment_args.json'), 'w') as f:
        json.dump(vars(args), f)
    train_genome_ac_model(hapt_genotypes_path=args.hapt_genotypes_path,
                          extra_data_path=args.extra_data_path,
                          experiment_name=args.experiment_name,
                          latent_size=args.latent_size,
                          alph=args.alph,
                          g_learn=args.g_learn,
                          d_learn=args.d_learn,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          class_loss_weights=args.class_loss_weights,
                          save_number=args.save_number,
                          synthetic_samples_number=args.synthetic_samples_number,
                          minimum_samples=args.minimum_samples,
                          target_column=args.target_column,
                          d_activation=args.d_activation,
                          class_loss_function=args.class_loss_function,
                          validation_loss_function=args.validation_loss_function,
                          with_extra_data=args.with_extra_data,
                          test_discriminator_classifier=args.test_discriminator_classifier,
                          test_dataset_path=args.test_dataset_path,
                          required_populations=args.required_populations,
                          resume_from_checkpoint=args.resume_from_checkpoint)
