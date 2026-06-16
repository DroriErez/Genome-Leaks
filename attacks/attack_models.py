import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from attacks.random.random_attack import RandomAttack
from attacks.MonteCarlo.MonteCarlo_attack import MonteCarlo_attack
from attacks.ReconstructionLossAttack.reconstruction_loss_attack import ReconstructionLossAttack
from attacks.CriticScoreAttack.critic_score_attack import CriticScoreAttack
from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper 
from models.models_factory import create_model_wrapper
from AA_Simulation.measurements import calc_AA


# Define paths
models_folder = "attacks/models_to_attack"
results_folder = "attacks/results"

# Development filter. Set to None to attack every checkpoint in models_folder.
# Examples:
#   MODEL_NAME_REGEX = r"^WGAN.*\.pth$"
#   MODEL_NAME_REGEX = r"^(WGAN|VAE)_model_last_model\.pth$"
MODEL_NAME_REGEX = r"^VAE.*\.pth$"
MAX_TEST_SET_SIZE = 1000
TEST_PREDICTION_BATCH_SIZE = 32
MONTE_CARLO_N_SAMPLES = 1000000
MONTE_CARLO_D_MIN_N_SAMPLES = 1000
MONTE_CARLO_GENERATION_BATCH_SIZE = 2048
SYNTHETIC_CACHE_BATCH_SIZE = 2048
CANDIDATE_BATCH_SIZE = 32
MODEL_GENERATION_BATCH_SIZE = 256
# TEST_PREDICTION_BATCH_SIZE = 32000
# MONTE_CARLO_N_SAMPLES = 1000000
# MONTE_CARLO_D_MIN_N_SAMPLES = 10000
# MONTE_CARLO_GENERATION_BATCH_SIZE = 4096
# SYNTHETIC_CACHE_BATCH_SIZE = 8192
# CANDIDATE_BATCH_SIZE = 128
# MODEL_GENERATION_BATCH_SIZE = 512
SYNTHETIC_CACHE_DIR = Path(results_folder) / "tmp_synthetic"
BALANCED_ACCURACY_THRESHOLD = 0.5

os.makedirs(results_folder, exist_ok=True)


def get_model_files(models_dir, model_name_regex=None):
    """Return checkpoint files selected by an optional filename regex."""
    models_path = Path(models_dir)
    model_files = sorted(path for path in models_path.glob("*.pth") if path.is_file())

    if model_name_regex is None:
        return model_files

    pattern = re.compile(model_name_regex)
    return [path for path in model_files if pattern.search(path.name)]


def save_attack_results(model_name, attack_results, output_dir):
    """Save one aggregated metrics CSV per model and one metrics CSV per attack."""
    out_paths = []
    summary_data_list = []
    for attack_name, data in attack_results.items():
        attack_metrics = data["metrics"]
        summary_data = {
            "model": model_name,
            "attack": attack_name,
            "tp": attack_metrics["tp"],
            "tn": attack_metrics["tn"],
            "fp": attack_metrics["fp"],
            "fn": attack_metrics["fn"],
            "accuracy": attack_metrics["accuracy"],
            "optimal_accuracy": attack_metrics["optimal_accuracy"],
            "optimal_accuracy_threshold": attack_metrics["optimal_accuracy_threshold"],
            "precision": attack_metrics["precision"],
            "recall": attack_metrics["recall"],
            "specificity": attack_metrics["specificity"],
            "f1_score": attack_metrics["f1_score"],
            "auc": attack_metrics["auc"],
            "tpr_01": attack_metrics["tpr_01"],
            "tpr_001": attack_metrics["tpr_001"],
            "AA_train": attack_metrics["AA_train"],
            "AA_test": attack_metrics["AA_test"],
            "privacy_loss": attack_metrics["privacy_loss"],
            "attack_calibration_dataset_size": attack_metrics["attack_calibration_dataset_size"],
            "testing_dataset_size": attack_metrics["testing_dataset_size"],
            "true_samples_count": attack_metrics["true_samples_count"],
            "false_samples_count": attack_metrics["false_samples_count"],
            "true_samples_percentage": attack_metrics["true_samples_percentage"],
            "false_samples_percentage": attack_metrics["false_samples_percentage"],
        }
        summary_data_list.append(summary_data)
        out_path = Path(output_dir) / f"{model_name}_{attack_name}_metrics.csv"
        pd.DataFrame([summary_data]).to_csv(out_path, index=False)
        out_paths.append(out_path)
    out_path = Path(output_dir) / f"{model_name}_metrics.csv"
    pd.DataFrame(summary_data_list).to_csv(out_path, index=False)
    out_paths.append(out_path)
    return out_paths


def save_attack_metrics(model_name, attack_results, output_dir):
    return save_attack_results(model_name, attack_results, output_dir)


def save_attack_predictions(
    model_name,
    attack_name,
    true_labels,
    predictions,
    scores,
    output_dir,
    sample_indices=None,
    raw_scores=None,
    inside_d_min_percent=None,
    threshold=None,
    raw_score_threshold=None,
    raw_score_threshold_0_5=None,
    predictions_threshold_0_5=None,
):
    """Save attack predictions and scores to CSV."""
    out_path = Path(output_dir) / f"{model_name}_{attack_name}_predictions.csv"
    prediction_data = {
        "true_label": true_labels,
        "prediction": predictions,
        "score": scores,
        "percentile": scores,
    }
    if sample_indices is not None:
        prediction_data["sample_index"] = sample_indices
    if raw_scores is not None:
        prediction_data["raw_score"] = raw_scores
    if inside_d_min_percent is not None:
        prediction_data["inside_d_min_percent"] = inside_d_min_percent
    if threshold is not None:
        prediction_data["threshold"] = threshold
    if raw_score_threshold is not None:
        prediction_data["raw_score_threshold"] = raw_score_threshold
    if raw_score_threshold_0_5 is not None:
        prediction_data["raw_score_threshold_0_5"] = raw_score_threshold_0_5
    if predictions_threshold_0_5 is not None:
        prediction_data["prediction_threshold_0_5"] = predictions_threshold_0_5
    pd.DataFrame(prediction_data).to_csv(out_path, index=False)
    return out_path

def call_AA_dist_metrics(training_points, test_points, synth_points):

        AAtr, real2real_dists_tr, real2synth_dists_tr, synth2synth_dists_tr = calc_AA(training_points, synth_points)
        AAte, real2real_dists_te, real2synth_dists_te, synth2synth_dists_te = calc_AA(test_points, synth_points)

        privacy_loss = AAte - AAtr

        return AAtr, AAte, privacy_loss


# TPR at fixed FPR
def tpr_at_fpr(fpr, tpr, target):
    return np.interp(target, fpr, tpr)

def evaluate_predictions(y_true, y_pred, scores):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    accuracy = (tp + tn) / len(y_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # TPR

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # TNR

    balanced_accuracy = (recall + specificity) / 2

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    auc = roc_auc_score(y_true, scores)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    tpr_01 = tpr_at_fpr(fpr, tpr, 0.01)
    tpr_001 = tpr_at_fpr(fpr, tpr, 0.001)

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,   # same as TPR
        "specificity": specificity,
        "fpr": fpr,
        "f1_score": f1,
        "auc": auc,
        "tpr_01": tpr_01,
        "tpr_001": tpr_001,
    }


def predictions_at_threshold(scores, threshold=BALANCED_ACCURACY_THRESHOLD):
    return np.asarray(scores) >= threshold


def optimal_accuracy_from_scores(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)

    best_accuracy = -1.0
    best_threshold = None
    candidate_thresholds = np.unique(scores)
    candidate_thresholds = np.concatenate(
        ([np.nextafter(np.max(scores), np.inf)], candidate_thresholds)
    )

    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        accuracy = np.mean(predictions == y_true)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_accuracy, best_threshold


def predict_attack_in_batches(
    attack_instance,
    data,
    indices,
    batch_size,
    labels=None,
    predict_kwargs=None,
):
    """Run attack predictions over selected data indices without one large allocation."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    predict_kwargs = {} if predict_kwargs is None else predict_kwargs
    predictions = []
    scores = []
    raw_scores = []
    has_raw_scores = False
    total = len(indices)
    num_batches = int(np.ceil(total / batch_size))

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_indices = indices[start:end]
        batch_predictions, batch_scores = attack_instance.predict(
            data[batch_indices],
            **predict_kwargs,
        )
        batch_predictions = np.asarray(batch_predictions)
        batch_scores = np.asarray(batch_scores)

        predictions.append(batch_predictions)
        scores.append(batch_scores)
        batch_raw_scores = getattr(attack_instance, "last_raw_scores", None)
        batch_has_raw_scores = (
            batch_raw_scores is not None and len(batch_raw_scores) == len(batch_scores)
        )
        if batch_has_raw_scores:
            batch_raw_scores = np.asarray(batch_raw_scores)
            raw_scores.append(batch_raw_scores)
            has_raw_scores = True

        predicted_members = int(np.sum(batch_predictions.astype(int)))
        message = (
            f"  Predicted batch {start // batch_size + 1}/{num_batches}: "
            f"{end}/{total}, "
            f"predicted_members={predicted_members}/{len(batch_predictions)}, "
            f"score_min={np.min(batch_scores):.4f}, "
            f"score_mean={np.mean(batch_scores):.4f}, "
            f"score_max={np.max(batch_scores):.4f}"
        )
        if batch_has_raw_scores:
            message += (
                f", raw_min={np.min(batch_raw_scores):.6f}, "
                f"raw_median={np.median(batch_raw_scores):.6f}, "
                f"raw_max={np.max(batch_raw_scores):.6f}"
            )
        if labels is not None:
            batch_labels = labels[batch_indices]
            batch_predictions_int = batch_predictions.astype(int)
            batch_labels_int = batch_labels.astype(int)
            batch_accuracy = np.mean(batch_predictions_int == batch_labels_int)
            batch_optimal_accuracy, _ = optimal_accuracy_from_scores(
                batch_labels_int,
                batch_scores,
            )
            batch_tp = np.sum((batch_labels_int == 1) & (batch_predictions_int == 1))
            batch_fn = np.sum((batch_labels_int == 1) & (batch_predictions_int == 0))
            batch_tpr = (
                batch_tp / (batch_tp + batch_fn)
                if (batch_tp + batch_fn) > 0 else 0.0
            )
            message += (
                f", batch_accuracy={batch_accuracy:.4f}, "
                f"batch_tpr={batch_tpr:.4f}, "
                f"batch_optimal_accuracy={batch_optimal_accuracy:.4f}"
            )
        print(message)

    raw_scores = np.concatenate(raw_scores) if has_raw_scores else None
    return np.concatenate(predictions), np.concatenate(scores), raw_scores

def plot_roc_curve(y_true, scores, title="ROC Curve", show=False, save_path=None):
    """Plot ROC curve for binary scores."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

        
def run_attacks(
    model,
    train_path,
    attack_train_path,
    non_train_path,
    load_n=500,
    max_test_set_size=None,
    synthetic_cache_key=None,
) -> dict:
    """Run a set of attacks on a model wrapper.

    This function is intended as the centralized attacker interface for the
    `attack_models.py` script.

    Args:
        wrapper: instance of GenomeGenerativeModelWrapper.
        train_path: path to training data file.
        attack_train_path: path to attack training data file.
        non_train_path: path to non-training data file.
        test_path: path to test data file.

    Returns:
        dict: summary of attack metrics.
    """
    results = {}
    AA_n = 100


    """Run a set of attacks on a model wrapper.
    This function is intended as the centralized attacker interface for the `attack_models.py` script.
    Args:   
        wrapper: instance of GenomeGenerativeModelWrapper.
        train_path: path to training data file.
        attack_train_path: path to attack training data file.
        non_train_path: path to non-training data file.
        test_path: path to test data file.
    Returns:
        dict: summary of attack metrics.
    """

    print(
        f"Running attacks on model: {model.model_name}, "
        f"with number of samples: {load_n}, "
        f"max test set size: {max_test_set_size}"
    )

    # Load a bounded sample of data for the attack/evaluation split.

    train_path_df = pd.read_csv(train_path, sep=' ', header=None, nrows=load_n)
    train_data = train_path_df.drop(train_path_df.columns[0:2], axis=1).values

    attack_train_path_df = pd.read_csv(attack_train_path, sep=' ', header=None, nrows=load_n)
    attack_train_data = attack_train_path_df.drop(attack_train_path_df.columns[0:2], axis=1).values

    non_train_path_df = pd.read_csv(non_train_path, sep=' ', header=None, nrows=load_n)
    non_train_data = non_train_path_df.drop(non_train_path_df.columns[0:2], axis=1).values

    rng = np.random.default_rng(42)

    n = min(len(train_data), len(non_train_data), load_n)
    if max_test_set_size is not None:
        n = min(n, max_test_set_size // 2)

    if n < 1:
        raise ValueError("Test set must contain at least one member and one non-member sample")

    idx_train = rng.choice(len(train_data), size=n, replace=False)
    idx_non = rng.choice(len(non_train_data), size=n, replace=False)

    train_samples = train_data[idx_train]
    non_train_samples = non_train_data[idx_non]
    y_train = np.ones(n, dtype=int)      # members
    y_non = np.zeros(n, dtype=int)       # non-members

    X = np.concatenate([train_samples, non_train_samples], axis=0)
    y = np.concatenate([y_train, y_non], axis=0)

    perm = rng.permutation(len(X))
    if max_test_set_size is not None:
        perm = perm[:max_test_set_size]

    test_labels = y[perm]
    print(f"Using test set size: {len(perm)}")

    aa_n = min(AA_n, len(train_samples), len(non_train_samples))

    attacks = [
        # RandomAttack(),
        # MonteCarlo_attack(
        #     n_samples=MONTE_CARLO_N_SAMPLES,
        #     d_min_n_samples=MONTE_CARLO_D_MIN_N_SAMPLES,
        #     generation_batch_size=MONTE_CARLO_GENERATION_BATCH_SIZE,
        #     distance_metric="euclidean",
        #     candidate_batch_size=CANDIDATE_BATCH_SIZE,
        #     synthetic_cache_dir=SYNTHETIC_CACHE_DIR,
        #     synthetic_cache_batch_size=SYNTHETIC_CACHE_BATCH_SIZE,
        #     synthetic_cache_key=synthetic_cache_key,
        # ),
        # CriticScoreAttack(n_repeats=3, batch_size=16),
        ReconstructionLossAttack()
    ]  # Attack instances

    attack_thresholds = {
        "monte_carlo_attack": 0.99,
        "reconstruction_loss_attack": 0.99,
        "critic_score_attack": 0.99,
    }

    synthetic_samples = model.generate(n=aa_n)

    AAtr,AAte, privacy_loss = call_AA_dist_metrics(train_samples[:aa_n], non_train_samples[:aa_n], synthetic_samples[:aa_n])
    print(f"AA on model {model.model_name}: AAtr: {AAtr:.4f}, AAte: {AAte:.4f}, Privacy Loss: {privacy_loss:.4f}")   

    for attack_instance in attacks:
        if not attack_instance.is_attack_applicable(model):
            print(f"Attack {attack_instance.name} is not applicable to model {model.model_name}. Skipping.")
            continue

        attack_name = attack_instance.name
        print(f"Running attack: {attack_name}")
        if attack_name == "monte_carlo_attack" and attack_instance.synthetic_cache_path is not None:
            print(f"Using synthetic cache for Monte Carlo attack: {attack_instance.synthetic_cache_path}")

        attack_instance.fit(
            non_train_data=attack_train_data,
            thr=attack_thresholds.get(attack_name, 0.5),
            modelWrapper=model,
        )  # Fit on synthetic and non-training samples

        predictions, scores, raw_scores = predict_attack_in_batches(
            attack_instance,
            X,
            perm,
            TEST_PREDICTION_BATCH_SIZE,
            labels=y,
        )
        inside_d_min_percent = None
        if attack_name == "monte_carlo_attack" and raw_scores is not None:
            inside_d_min_percent = raw_scores * 100
        if attack_name == "critic_score_attack" and raw_scores is not None:
            print(
                "Critic raw score results: "
                f"min={np.min(raw_scores):.6f}, "
                f"median={np.median(raw_scores):.6f}, "
                f"mean={np.mean(raw_scores):.6f}, "
                f"max={np.max(raw_scores):.6f}"
            )

        # print("test_label,predicted_label,score")
        # for test_label, predicted_label, score in zip(test_labels, predictions, scores):
        #     print(f"{int(test_label)},{int(predicted_label)},{score:.6f}")

        # print("threshold 99%", np.percentile(scores, 99))
        # print("threshold 99.9%", np.percentile(scores, 99.9))


        predictions_threshold_0_5 = predictions_at_threshold(scores)
        eval_metrics = evaluate_predictions(test_labels, predictions, scores)
        optimal_accuracy, optimal_accuracy_threshold = optimal_accuracy_from_scores(
            test_labels,
            scores,
        )
        eval_metrics["optimal_accuracy"] = optimal_accuracy
        eval_metrics["optimal_accuracy_threshold"] = optimal_accuracy_threshold
        eval_metrics["AA_train"] = AAtr
        eval_metrics["AA_test"] = AAte
        eval_metrics["privacy_loss"] = privacy_loss
        true_samples_count = int(np.sum(test_labels == 1))
        false_samples_count = int(np.sum(test_labels == 0))
        testing_dataset_size = len(test_labels)
        eval_metrics["attack_calibration_dataset_size"] = len(attack_train_data)
        eval_metrics["testing_dataset_size"] = testing_dataset_size
        eval_metrics["true_samples_count"] = true_samples_count
        eval_metrics["false_samples_count"] = false_samples_count
        eval_metrics["true_samples_percentage"] = (
            100 * true_samples_count / testing_dataset_size
            if testing_dataset_size > 0 else 0.0
        )
        eval_metrics["false_samples_percentage"] = (
            100 * false_samples_count / testing_dataset_size
            if testing_dataset_size > 0 else 0.0
        )

        prediction_threshold = getattr(attack_instance, "score_threshold", None)
        if prediction_threshold is None:
            prediction_threshold = getattr(attack_instance, "threshold", None)

        raw_prediction_threshold = getattr(
            attack_instance,
            "raw_score_threshold",
            None,
        )
        raw_threshold_0_5 = getattr(
            attack_instance,
            "raw_score_threshold_0_5",
            None,
        )
        threshold_message = ""
        if raw_prediction_threshold is not None:
            threshold_message += (
                f", Raw threshold: {raw_prediction_threshold:.6f}"
            )
        if raw_threshold_0_5 is not None:
            threshold_message += (
                f", Raw threshold @ 0.5: {raw_threshold_0_5:.6f}"
            )

        print(
            f"Attack: {attack_name}, "
            f"Accuracy: {eval_metrics['accuracy']:.4f}, "
            f"Optimal Accuracy: {eval_metrics['optimal_accuracy']:.4f}, "
            f"Optimal Accuracy Threshold: "
            f"{eval_metrics['optimal_accuracy_threshold']:.4f}, "
            f"AUC: {eval_metrics['auc']:.4f}, "
            f"TPR@FPR=0.01: {eval_metrics['tpr_01']:.4f}, "
            f"TPR@FPR=0.001: {eval_metrics['tpr_001']:.4f}"
            f"{threshold_message}"
        )

        prediction_path = save_attack_predictions(
            model.model_name,
            attack_name,
            test_labels,
            predictions,
            scores,
            results_folder,
            sample_indices=perm,
            raw_scores=raw_scores,
            inside_d_min_percent=inside_d_min_percent,
            threshold=prediction_threshold,
            raw_score_threshold=raw_prediction_threshold,
            raw_score_threshold_0_5=raw_threshold_0_5,
            predictions_threshold_0_5=predictions_threshold_0_5,
        )
        raw_log_message = ""
        if raw_scores is not None:
            raw_log_message = ", including raw_score"
            if raw_prediction_threshold is not None:
                raw_log_message += ", raw_score_threshold"
            if raw_threshold_0_5 is not None:
                raw_log_message += ", raw_score_threshold_0_5"
        print(f"Saved prediction log: {prediction_path}{raw_log_message}")

        results[attack_name] = {
            "metrics": eval_metrics,
            "true_labels": test_labels,
            "predictions": predictions,
            "predictions_threshold_0_5": predictions_threshold_0_5,
            "scores": scores,
            "raw_scores": raw_scores,
            "inside_d_min_percent": inside_d_min_percent,
        }

        plot_roc_curve(
            test_labels,
            scores,
            title=f"ROC Curve - {attack_name}",
            show=False,
            save_path=f"{results_folder}/{model.model_name}_{attack_name}_roc.png"

        )

    return results


# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

model_files = get_model_files(models_folder, MODEL_NAME_REGEX)
if MODEL_NAME_REGEX is not None:
    print(f"Using model filename regex: {MODEL_NAME_REGEX}")
print(f"Found {len(model_files)} model checkpoint(s) to attack.")

# Loop over selected models in the folder
for model_file in model_files:
    print(f"\nProcessing model: {model_file.name}")
    
    # Build wrapper from model file
    model = create_model_wrapper(file_name=str(model_file))
    if hasattr(model, "generation_batch_size"):
        model.generation_batch_size = MODEL_GENERATION_BATCH_SIZE
        print(f"Using model generation batch size: {model.generation_batch_size}")
    print(f"Using wrapper model architecture: {model.get_model_architecture()}")

    # Generate dataset paths based on model file name
    base = model_file.stem
    train_path = f"{models_folder}/{base}_train.hapt"
    eval_path = f"{models_folder}/{base}_eval.hapt"
    test_path = f"{models_folder}/{base}_test.hapt"
    print(f"Dataset paths: Train: {train_path}, Eval: {eval_path}, Test: {test_path}")

    # Run attacks
    print(f"Running attacks on {model_file.name}...")
    attack_results = run_attacks(
        model,
        train_path,
        eval_path,
        test_path,
        max_test_set_size=MAX_TEST_SET_SIZE,
        synthetic_cache_key=f"{base}_{model_file.stat().st_mtime_ns}",
    )

    metric_paths = save_attack_results(model.model_name, attack_results, results_folder)
    for metric_path in metric_paths:
        print(f"Saved metrics log: {metric_path}")

print("\nAll models processed successfully!")
