"""Genome AC-GAN checkpoint adapter.

This wrapper restores TensorFlow checkpoints produced by
`genome_ac_gan_training.py` and exposes the same minimal API used by the attack
runner: `generate`, `get_model_architecture`, plus discriminator scoring
helpers for membership attacks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop


AC_GAN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = AC_GAN_ROOT.parent.parent
if str(AC_GAN_ROOT) not in sys.path:
    sys.path.insert(0, str(AC_GAN_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from genome_ac_gan_training import (  # noqa: E402
    build_acgan,
    build_checkpoint,
    build_discriminator,
    build_generator,
    polyloss_ce,
)
from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper  # noqa: E402


class ACGAN_generative(GenomeGenerativeModelWrapper):
    """AC-GAN wrapper for TensorFlow checkpoints."""

    LABEL_COLUMN = "Superpopulation code"

    def __init__(
        self,
        model_path: str = None,
        generation_batch_size: int = 256,
        latent_size: int = 800,
        alph: float = 0.01,
        d_activation: str = "sigmoid",
        g_learn: float = 0.0001,
        d_learn: float = 0.0008,
        class_loss_weights: float = 1,
        validation_loss_function: str = "binary_crossentropy",
        class_loss_function: str = "polyloss_ce",
        number_of_genotypes: int = None,
        num_classes: int = None,
    ):
        super().__init__(model_path)
        self.model_architecture = "AC-GAN"
        self.generation_batch_size = generation_batch_size
        self.latent_size = latent_size
        self.alph = alph
        self.d_activation = d_activation
        self.g_learn = g_learn
        self.d_learn = d_learn
        self.class_loss_weights = class_loss_weights
        self.validation_loss_function = validation_loss_function
        self.class_loss_function = class_loss_function
        self.number_of_genotypes = number_of_genotypes
        self.num_classes = num_classes
        self.class_to_id = None
        self.id_to_class = None
        self.generator = None
        self.discriminator = None
        self.acgan = None
        self.checkpoint = None
        self.checkpoint_prefix = None

        if model_path:
            self.init(model_path)

    def init(self, file_path: str) -> None:
        """Restore generator, discriminator, and AC-GAN from a TF checkpoint."""
        checkpoint_prefix = self._checkpoint_prefix(file_path)
        self.checkpoint_prefix = checkpoint_prefix
        model_path = Path(file_path)

        self._load_metadata(model_path)
        self._infer_number_of_genotypes(model_path)
        self._infer_metadata_from_checkpoint(checkpoint_prefix)

        if self.number_of_genotypes is None:
            raise ValueError(
                "Could not infer number_of_genotypes. Provide a matching "
                "`<checkpoint>_train.csv` file or pass number_of_genotypes."
            )
        if self.num_classes is None:
            self.num_classes = 5

        self.generator = build_generator(
            latent_dim=self.latent_size,
            num_classes=self.num_classes,
            number_of_genotypes=self.number_of_genotypes,
            alph=self.alph,
        )
        self.generator.compile(metrics=["accuracy"])

        self.discriminator = build_discriminator(
            number_of_genotypes=self.number_of_genotypes,
            num_classes=self.num_classes,
            alph=self.alph,
            d_activation=self.d_activation,
        )
        self.discriminator.compile(
            optimizer=RMSprop(learning_rate=self.d_learn),
            loss=[self.validation_loss_function, self._class_loss()],
            loss_weights=[1, self.class_loss_weights],
            metrics=["binary_accuracy", "categorical_accuracy"],
        )

        self.discriminator.trainable = False
        self.acgan = build_acgan(self.generator, self.discriminator)
        self.acgan.compile(
            optimizer=RMSprop(learning_rate=self.g_learn),
            loss=[self.validation_loss_function, self._class_loss()],
            loss_weights=[1, self.class_loss_weights],
            metrics=["binary_accuracy", "categorical_accuracy"],
        )

        self.checkpoint = build_checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            acgan=self.acgan,
        )
        self.checkpoint.restore(str(checkpoint_prefix)).expect_partial()
        self.model = self.generator

    @staticmethod
    def _checkpoint_prefix(file_path: str) -> Path:
        path = Path(file_path)
        if path.suffix == ".index":
            return path.with_suffix("")
        return path

    def _class_loss(self):
        if self.class_loss_function == "categorical_accuracy":
            return CategoricalAccuracy()
        if self.class_loss_function == "polyloss_ce":
            return polyloss_ce
        return self.class_loss_function

    def _load_metadata(self, model_path: Path) -> None:
        class_map_path = model_path.parent / "class_id_map.json"
        if not class_map_path.exists() and model_path.parent.name == "checkpoints":
            class_map_path = model_path.parent.parent / "class_id_map.json"

        if class_map_path.exists():
            with open(class_map_path, "r") as f:
                self.class_to_id = json.load(f)
            self.id_to_class = {int(v): k for k, v in self.class_to_id.items()}
            self.num_classes = len(self.class_to_id)

    def _infer_number_of_genotypes(self, model_path: Path) -> None:
        if self.number_of_genotypes is not None:
            return

        dataset_path = model_path.parent / f"{model_path.stem}_train.csv"
        if not dataset_path.exists():
            return

        first_row = pd.read_csv(dataset_path, nrows=1)
        genotype_columns = self._genotype_columns(first_row)
        self.number_of_genotypes = len(genotype_columns)

    def get_attack_dataset_paths(self, models_folder: str, base: str) -> dict:
        """Return AC-GAN attack datasets using the checkpoint basename."""
        models_path = Path(models_folder)
        return {
            "train": models_path / f"{base}_train.csv",
            "eval": models_path / f"{base}_eval.csv",
            "test": models_path / f"{base}_test.csv",
        }

    def load_attack_dataset(self, path, nrows=None) -> np.ndarray:
        """Load an AC-GAN CSV attack dataset as a genotype matrix."""
        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        df = self._filter_known_labels(df, path, warn=True)
        genotype_columns = self._genotype_columns(df)
        if not genotype_columns:
            raise ValueError(f"No genotype columns found in AC-GAN dataset: {path}")
        values = df[genotype_columns].apply(pd.to_numeric, errors="raise").values
        return self._as_attack_matrix(values, path)

    def load_attack_labels(self, path, nrows=None):
        """Load AC-GAN class IDs for label-aware discriminator confidence."""
        if not self.class_to_id:
            return None

        df = pd.read_csv(path, nrows=nrows, low_memory=False)
        df = self._filter_known_labels(df, path, warn=False)

        raw_labels = df[self.LABEL_COLUMN].astype(str)
        labels = raw_labels.map(self.class_to_id)
        unknown_labels = raw_labels[labels.isna()]
        if len(unknown_labels) > 0:
            examples = sorted(set(unknown_labels))[:5]
            raise ValueError(
                f"Unknown AC-GAN class labels in {path} column {self.LABEL_COLUMN}: {examples}"
            )
        return labels.astype(int).to_numpy()

    def _filter_known_labels(self, df, path, warn=True):
        if not self.class_to_id:
            return df
        if self.LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"AC-GAN dataset {path} must contain label column {self.LABEL_COLUMN}"
            )

        labels = df[self.LABEL_COLUMN].astype(str)
        known_mask = labels.isin(self.class_to_id)
        if known_mask.all():
            return df

        dropped = labels[~known_mask]
        examples = sorted(set(dropped))[:5]
        if warn:
            print(
                f"Dropping {len(dropped)} AC-GAN row(s) from {path} with labels "
                f"not in class_id_map.json column {self.LABEL_COLUMN}: {examples}"
            )
        return df.loc[known_mask].reset_index(drop=True)

    @staticmethod
    def _genotype_columns(df) -> list:
        """Return AC-GAN genotype columns, skipping leading/trailing metadata.

        AC-GAN CSV attack files may have numeric-looking metadata columns at the
        front, e.g. column "0" contains "Real" and column "1" contains sample
        IDs. Genotype columns are the numeric-named columns whose values are
        numeric for the loaded rows.
        """
        genotype_columns = []
        numeric_named_columns = sorted(
            [column for column in df.columns if str(column).isdigit()],
            key=lambda column: int(column),
        )
        for column in numeric_named_columns:
            if not str(column).isdigit():
                continue
            numeric_values = pd.to_numeric(df[column], errors="coerce")
            if numeric_values.notna().all():
                genotype_columns.append(column)
        return genotype_columns

    def _infer_metadata_from_checkpoint(self, checkpoint_prefix: Path) -> None:
        if self.number_of_genotypes is not None and self.num_classes is not None:
            return

        variables = tf.train.list_variables(str(checkpoint_prefix))

        if self.number_of_genotypes is None:
            one_dimensional_shapes = [
                shape[0]
                for _, shape in variables
                if len(shape) == 1 and shape[0] > 1
            ]
            if one_dimensional_shapes:
                self.number_of_genotypes = max(one_dimensional_shapes)

        if self.num_classes is None:
            for _, shape in variables:
                if len(shape) == 2:
                    inferred_num_classes = (shape[0] - self.latent_size) / 8
                    if inferred_num_classes.is_integer() and inferred_num_classes > 1:
                        self.num_classes = int(inferred_num_classes)
                        break

    def _label_batch(self, n: int, class_ids=None):
        if class_ids is None:
            class_ids = np.arange(n) % self.num_classes
        class_ids = np.asarray(class_ids, dtype=int)
        if class_ids.shape[0] != n:
            raise ValueError("class_ids length must match n")
        return tf.one_hot(class_ids, depth=self.num_classes)

    def _generate_batch(self, n: int, class_ids=None) -> np.ndarray:
        latent_samples = np.random.normal(loc=0, scale=1, size=(n, self.latent_size))
        labels = self._label_batch(n, class_ids=class_ids)
        generated = self.generator.predict([latent_samples, labels], verbose=0)
        generated[generated < 0] = 0
        generated = np.rint(generated)
        return generated.astype(np.int8, copy=False)

    def generate(self, n: int, class_ids=None) -> np.ndarray:
        """Generate n synthetic genomes."""
        if self.generator is None:
            raise ValueError("Model not loaded. Call init() first.")
        if n < 1:
            raise ValueError("n must be at least 1")

        generated_batches = []
        generated_so_far = 0
        while generated_so_far < n:
            current_batch_size = min(self.generation_batch_size, n - generated_so_far)
            batch_class_ids = None
            if class_ids is not None:
                batch_class_ids = class_ids[
                    generated_so_far:generated_so_far + current_batch_size
                ]
            generated_batches.append(
                self._generate_batch(current_batch_size, batch_class_ids)
            )
            generated_so_far += current_batch_size
        return np.vstack(generated_batches)

    def discriminator_outputs(self, samples, batch_size: int = None):
        """Return discriminator validity and class probabilities."""
        if self.discriminator is None:
            raise ValueError("Discriminator not loaded. Call init() first.")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        validity, class_scores = self.discriminator.predict(
            samples,
            batch_size=batch_size or self.generation_batch_size,
            verbose=0,
        )
        return validity.reshape(-1), class_scores

    def discriminator_score(self, samples, batch_size: int = None) -> np.ndarray:
        validity, _ = self.discriminator_outputs(samples, batch_size=batch_size)
        return validity

    def class_confidence(self, samples, batch_size: int = None) -> np.ndarray:
        _, class_scores = self.discriminator_outputs(samples, batch_size=batch_size)
        return np.max(class_scores, axis=1)
