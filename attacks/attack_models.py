import argparse
import gc
import os
import re
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLOT_LABEL_FONT_SIZE = 22
PLOT_TITLE_FONT_SIZE = 20
PLOT_LEGEND_FONT_SIZE = 20
PLOT_TICK_FONT_SIZE = 18
PLOT_CALLOUT_FONT_SIZE = 20
PLOT_FIG_SIZE = (6.5, 5.7)
RUN_SUMMARY_COLUMNS = [
    "Model name",
    "Epoch",
    "Model type",
    "Attack type",
    "Attack",
    "Timestep",
    "N repeats",
    "Accuracy @FPR=0.01",
    "Accuracy @0.5",
    "AUC",
    "TPR@FPR=0.01",
    "TPR@FPR=0.001",
    "AA train",
    "AA test",
    "AA samples",
    "Privacy loss",
    "AF_MAE",
    "AF_MSE",
    "AF_RMSE",
    "AF_MaxAbsError",
    "AF_Pearson",
    "AF_Spearman",
    "Real polymorphic SNP fraction",
    "Synthetic polymorphic SNP fraction",
    "PCA2 W distance",
    "real_vs_synth_AUC",
    "Quality AA",
    "Quality AA samples",
    "Real-to-real distance mean",
    "Real-to-synthetic distance mean",
    "Synthetic-to-synthetic distance mean",
    "Quality n real",
    "Quality n synthetic",
    "Quality n SNPs",
]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from attacks.random.random_attack import RandomAttack
from attacks.MonteCarlo.MonteCarlo_attack import MonteCarlo_attack
from attacks.ReconstructionLossAttack.reconstruction_loss_attack import ReconstructionLossAttack
from attacks.CriticScoreAttack.critic_score_attack import CriticScoreAttack
from attacks.DiscriminatorScoreAttack.discriminator_score_attack import DiscriminatorScoreAttack
from attacks.DiffusionLossAttack.diffusion_loss_attack import DiffusionLossAttack
from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper 
from models.evaluate_model_quality import evaluate_model_quality
from models.models_factory import create_model_wrapper
from AA_Simulation.measurements import calc_AA
from utils.device import get_device, get_gpu_memory_gb, get_memory_based_batch_size


# Define paths
DEFAULT_MODELS_FOLDER = "attacks/models_to_attack"
PARAM_MODEL_FOLDER = r"F:\Thesis\models_to_attack"
DEFAULT_RESULTS_FOLDER = "attacks/results"
PARAM_RESULTS_FOLDER = r"F:\Thesis\results"

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    "-m",
    action="store_true",
    help=f"Use the parameter models folder ({PARAM_MODEL_FOLDER})",
)
argument_parser.add_argument(
    "-r",
    action="store_true",
    help=f"Use the parameter results folder ({PARAM_RESULTS_FOLDER})",
)
command_line_args = argument_parser.parse_args()

models_folder = (
    PARAM_MODEL_FOLDER if command_line_args.m else DEFAULT_MODELS_FOLDER
)
results_folder = (
    PARAM_RESULTS_FOLDER if command_line_args.r else DEFAULT_RESULTS_FOLDER
)

# Development filter. Set to None to attack every checkpoint in models_folder.
# Examples:
MODEL_NAME_REGEX = r"^PCA_DM_model_(200|1000|2000|8000|19000|20000)\.pth$"
# MODEL_NAME_REGEX = r"^(WGAN|VAE)_model_\d+\.pth$"
# MODEL_NAME_REGEX = r"^WGAN_model_(0|10000)\.pth$"
# MODEL_NAME_REGEX = r"^(WGAN_model_(1000|10000)|VAE_model_10000)\.pth$"
# MODEL_NAME_REGEX = r"^WGAN_model_\d+\.pth$"
# MODEL_NAME_REGEX = r"^VAE.*\.pth$"
# MODEL_NAME_REGEX = r"^WGAN_model_10000\.pth$"
MAX_TEST_SET_SIZE = 1000
DENOISING_LOSS_N_REPEATS = 16
MODEL_SYN_TEST_SIZE = MAX_TEST_SET_SIZE // 2
TEST_PREDICTION_BATCH_SIZE = 32
MONTE_CARLO_N_SAMPLES = 1000000
MONTE_CARLO_D_MIN_N_SAMPLES = 1000
MONTE_CARLO_GENERATION_BATCH_SIZE = 2048
SYNTHETIC_CACHE_BATCH_SIZE = 2048
CANDIDATE_BATCH_SIZE = 32
MODEL_GENERATION_BATCH_SIZE = get_memory_based_batch_size(  
    small_batch_size=8,
    large_batch_size=512,
    large_gpu_memory_gb=16,
)
# TEST_PREDICTION_BATCH_SIZE = 32000
# MONTE_CARLO_N_SAMPLES = 1000000
# MONTE_CARLO_D_MIN_N_SAMPLES = 10000
# MONTE_CARLO_GENERATION_BATCH_SIZE = 4096
# SYNTHETIC_CACHE_BATCH_SIZE = 8192
# CANDIDATE_BATCH_SIZE = 128
# MODEL_GENERATION_BATCH_SIZE = 512
BALANCED_ACCURACY_THRESHOLD = 0.5
CRITIC_SCORE_DIAGNOSTIC_MAX_SAMPLES = MODEL_SYN_TEST_SIZE
RECONSTRUCTION_LOSS_DIAGNOSTIC_MAX_SAMPLES = MODEL_SYN_TEST_SIZE
DIFFUSION_LOSS_DIAGNOSTIC_MAX_SAMPLES = MODEL_SYN_TEST_SIZE
DISCRIMINATOR_SCORE_DIAGNOSTIC_MAX_SAMPLES = MODEL_SYN_TEST_SIZE
RUN_MODEL_QUALITY_EVALUATION = True
RUN_AA_EVALUATION = True
RUN_DENOISING_LOSS_DIAGNOSTICS = False
MODEL_QUALITY_N_SAMPLES = MODEL_SYN_TEST_SIZE
MODEL_QUALITY_PCA_PLOT_SAMPLES = MODEL_SYN_TEST_SIZE
MODEL_QUALITY_AA_SAMPLES = 200
MODEL_QUALITY_CLASSIFIER_PCA_COMPONENTS = 50
MODEL_QUALITY_RANDOM_SEED = 42
DEVICE = get_device()
GPU_MEMORY_GB = get_gpu_memory_gb()

print(f"Models folder: {models_folder}")
print(f"Results folder: {results_folder}")
print(f"Model filename filter: {MODEL_NAME_REGEX}")

os.makedirs(results_folder, exist_ok=True)


def get_model_files(models_dir, model_name_regex=None):
    """Return checkpoint files selected by an optional filename regex."""
    models_path = Path(models_dir)
    model_files = sorted(
        (
            path
            for pattern in ("*.pth", "*.index", "*.keras", "*.weights.h5")
            for path in models_path.glob(pattern)
            if path.is_file()
        ),
        key=lambda path: path.name,
    )

    if model_name_regex is None:
        return model_files

    pattern = re.compile(model_name_regex)
    selected_files = [path for path in model_files if pattern.search(path.name)]
    selected_by_base = {}
    for path in selected_files:
        base = model_checkpoint_base(path)
        existing = selected_by_base.get(base)
        if existing is None or (
            path.name.endswith("_generator.keras")
            and not existing.name.endswith("_generator.keras")
        ):
            selected_by_base[base] = path
    return [selected_by_base[base] for base in sorted(selected_by_base)]


def model_checkpoint_base(model_file):
    """Return the shared checkpoint stem for index and Keras component files."""
    stem = Path(model_file).stem
    if stem.endswith(".weights"):
        stem = stem[:-len(".weights")]
    if stem.endswith("_generator"):
        return stem[:-len("_generator")]
    if stem.endswith("_discriminator"):
        return stem[:-len("_discriminator")]
    return stem


def numeric_model_epoch(model_name):
    """Return the numeric checkpoint epoch, or an empty value if unavailable."""
    epoch = GenomeGenerativeModelWrapper.infer_model_epochs(model_name)
    return int(epoch) if str(epoch).isdigit() else ""


def save_attack_results(model_name, attack_results, output_dir):
    """Save one aggregated metrics CSV per model and one metrics CSV per attack."""
    out_paths = []
    summary_data_list = []
    for attack_name, data in attack_results.items():
        attack_metrics = data["metrics"]
        quality_metrics = data.get("quality_metrics", {})
        attack_display_name = data.get("display_name", attack_name)
        summary_data = {
            "model": model_name,
            "epoch": numeric_model_epoch(model_name),
            "attack": attack_name,
            "attack_display_name": attack_display_name,
            "timestep": data.get("timestep", ""),
            "n_repeats": data.get("n_repeats", ""),
            "tp": attack_metrics["tp"],
            "tn": attack_metrics["tn"],
            "fp": attack_metrics["fp"],
            "fn": attack_metrics["fn"],
            "accuracy": attack_metrics["accuracy"],
            "accuracy_threshold_0_5": attack_metrics["accuracy_threshold_0_5"],
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
            "AA_samples": attack_metrics["AA_samples"],
            "privacy_loss": attack_metrics["privacy_loss"],
            "attack_calibration_dataset_size": attack_metrics["attack_calibration_dataset_size"],
            "testing_dataset_size": attack_metrics["testing_dataset_size"],
            "true_samples_count": attack_metrics["true_samples_count"],
            "false_samples_count": attack_metrics["false_samples_count"],
            "true_samples_percentage": attack_metrics["true_samples_percentage"],
            "false_samples_percentage": attack_metrics["false_samples_percentage"],
            "allele_frequency_mae": quality_metrics.get("allele_frequency_mae", np.nan),
            "allele_frequency_mse": quality_metrics.get("allele_frequency_mse", np.nan),
            "allele_frequency_rmse": quality_metrics.get("allele_frequency_rmse", np.nan),
            "allele_frequency_max_abs_error": quality_metrics.get(
                "allele_frequency_max_abs_error", np.nan
            ),
            "allele_frequency_pearson": quality_metrics.get("allele_frequency_pearson", np.nan),
            "allele_frequency_spearman": quality_metrics.get("allele_frequency_spearman", np.nan),
            "real_polymorphic_snp_fraction": quality_metrics.get(
                "real_polymorphic_snp_fraction", np.nan
            ),
            "synth_polymorphic_snp_fraction": quality_metrics.get(
                "synth_polymorphic_snp_fraction", np.nan
            ),
            "pca_wasserstein_distance": quality_metrics.get("pca_wasserstein_distance", np.nan),
            "real_vs_synthetic_classifier_auc": quality_metrics.get(
                "real_vs_synthetic_classifier_auc", np.nan
            ),
            "quality_aa": quality_metrics.get("aa", np.nan),
            "quality_aa_samples": quality_metrics.get("aa_n_samples", np.nan),
            "real_to_real_distance_mean": quality_metrics.get(
                "real_to_real_distance_mean", np.nan
            ),
            "real_to_synth_distance_mean": quality_metrics.get(
                "real_to_synth_distance_mean", np.nan
            ),
            "synth_to_synth_distance_mean": quality_metrics.get(
                "synth_to_synth_distance_mean", np.nan
            ),
            "quality_n_real": quality_metrics.get("n_real", np.nan),
            "quality_n_synthetic": quality_metrics.get("n_synthetic", np.nan),
            "quality_n_snps": quality_metrics.get("n_snps", np.nan),
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


def accuracy_at_fpr(y_true, scores, target_fpr=0.01):
    """Return accuracy at the ROC threshold with FPR nearest to, but not above, target_fpr."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    fpr, _, thresholds = roc_curve(y_true, scores)
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) > 0:
        threshold_index = valid_indices[-1]
    else:
        threshold_index = int(np.argmin(np.abs(fpr - target_fpr)))

    threshold = thresholds[threshold_index]
    predictions = (scores >= threshold).astype(int)
    return float(np.mean(predictions == y_true))


def build_run_summary_rows(model_name, attack_results, quality_metrics=None, model_type=None):
    """Build compact model/attack rows for the run summary CSV."""
    quality_metrics = quality_metrics or {}
    if model_type is None:
        model_type = GenomeGenerativeModelWrapper.infer_model_type(model_name)
    rows = []
    for attack_name, data in attack_results.items():
        attack_metrics = data["metrics"]
        attack_display_name = data.get("display_name", attack_name)
        rows.append(
            {
                "Model name": model_name,
                "Epoch": numeric_model_epoch(model_name),
                "Model type": model_type,
                "Attack type": attack_name,
                "Attack": attack_display_name,
                "Timestep": data.get("timestep", ""),
                "N repeats": data.get("n_repeats", ""),
                "Accuracy @FPR=0.01": accuracy_at_fpr(
                    data["true_labels"],
                    data["scores"],
                    target_fpr=0.01,
                ),
                "Accuracy @0.5": attack_metrics["accuracy_threshold_0_5"],
                "AUC": attack_metrics["auc"],
                "TPR@FPR=0.01": attack_metrics["tpr_01"],
                "TPR@FPR=0.001": attack_metrics["tpr_001"],
                "AA train": attack_metrics["AA_train"],
                "AA test": attack_metrics["AA_test"],
                "AA samples": attack_metrics["AA_samples"],
                "Privacy loss": attack_metrics["privacy_loss"],
                "AF_MAE": quality_metrics.get("allele_frequency_mae", np.nan),
                "AF_MSE": quality_metrics.get("allele_frequency_mse", np.nan),
                "AF_RMSE": quality_metrics.get("allele_frequency_rmse", np.nan),
                "AF_MaxAbsError": quality_metrics.get(
                    "allele_frequency_max_abs_error", np.nan
                ),
                "AF_Pearson": quality_metrics.get("allele_frequency_pearson", np.nan),
                "AF_Spearman": quality_metrics.get("allele_frequency_spearman", np.nan),
                "Real polymorphic SNP fraction": quality_metrics.get(
                    "real_polymorphic_snp_fraction", np.nan
                ),
                "Synthetic polymorphic SNP fraction": quality_metrics.get(
                    "synth_polymorphic_snp_fraction", np.nan
                ),
                "PCA2 W distance": quality_metrics.get("pca_wasserstein_distance", np.nan),
                "real_vs_synth_AUC": quality_metrics.get(
                    "real_vs_synthetic_classifier_auc",
                    np.nan,
                ),
                "Quality AA": quality_metrics.get("aa", np.nan),
                "Quality AA samples": quality_metrics.get("aa_n_samples", np.nan),
                "Real-to-real distance mean": quality_metrics.get(
                    "real_to_real_distance_mean", np.nan
                ),
                "Real-to-synthetic distance mean": quality_metrics.get(
                    "real_to_synth_distance_mean", np.nan
                ),
                "Synthetic-to-synthetic distance mean": quality_metrics.get(
                    "synth_to_synth_distance_mean", np.nan
                ),
                "Quality n real": quality_metrics.get("n_real", np.nan),
                "Quality n synthetic": quality_metrics.get("n_synthetic", np.nan),
                "Quality n SNPs": quality_metrics.get("n_snps", np.nan),
            }
        )
    return rows


def create_run_output_dir(base_output_dir):
    """Create a unique timestamped directory for this script run."""
    base_output_dir = Path(base_output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"run_{timestamp}"
    output_dir = base_output_dir / run_folder_name
    suffix = 2
    while output_dir.exists():
        output_dir = base_output_dir / f"{run_folder_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def create_run_summary_path(output_dir):
    return Path(output_dir) / "attack_model_results.csv"


def initialize_run_summary(output_path):
    pd.DataFrame(columns=RUN_SUMMARY_COLUMNS).to_csv(output_path, index=False)
    return output_path


def append_run_summary_rows(rows, output_path):
    pd.DataFrame(rows, columns=RUN_SUMMARY_COLUMNS).to_csv(
        output_path,
        mode="a",
        header=False,
        index=False,
    )
    return output_path


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
    timestep="",
    n_repeats="",
):
    """Save attack predictions and scores to CSV."""
    out_path = Path(output_dir) / f"{model_name}_{attack_name}_predictions.csv"
    prediction_data = {
        "model": np.repeat(model_name, len(true_labels)),
        "epoch": np.repeat(numeric_model_epoch(model_name), len(true_labels)),
        "attack": np.repeat(attack_name, len(true_labels)),
        "timestep": np.repeat(timestep, len(true_labels)),
        "n_repeats": np.repeat(n_repeats, len(true_labels)),
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
        training_points = as_aa_matrix(training_points, "training_points")
        test_points = as_aa_matrix(test_points, "test_points")
        synth_points = as_aa_matrix(synth_points, "synth_points")

        AAtr, real2real_dists_tr, real2synth_dists_tr, synth2synth_dists_tr = calc_AA(training_points, synth_points)
        AAte, real2real_dists_te, real2synth_dists_te, synth2synth_dists_te = calc_AA(test_points, synth_points)

        privacy_loss = AAte - AAtr

        return AAtr, AAte, privacy_loss


def as_aa_matrix(points, name):
    matrix = np.asarray(points, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix, got shape {matrix.shape}")
    if matrix.shape[0] < 2:
        raise ValueError(f"{name} must contain at least 2 samples, got shape {matrix.shape}")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least 1 feature, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


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
        batch_predict_kwargs = dict(predict_kwargs)
        if "class_ids" in batch_predict_kwargs and batch_predict_kwargs["class_ids"] is not None:
            batch_predict_kwargs["class_ids"] = batch_predict_kwargs["class_ids"][batch_indices]
        batch_predictions, batch_scores = attack_instance.predict(
            data[batch_indices],
            **batch_predict_kwargs,
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
    target_fpr = 0.01
    tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)

    plt.figure(figsize=PLOT_FIG_SIZE)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random (AUC=0.5)")
    plt.scatter(target_fpr, tpr_at_target_fpr, color="tab:red", zorder=3)
    plt.annotate(
        f"TPR @ FPR = 0.01: {tpr_at_target_fpr:.3f}",
        xy=(target_fpr, tpr_at_target_fpr),
        xytext=(0.30, 0.12),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "tab:red", "alpha": 0.9},
        fontsize=PLOT_CALLOUT_FONT_SIZE,
    )
    plt.xlabel("False Positive Rate (FPR)", fontsize=PLOT_LABEL_FONT_SIZE)
    plt.ylabel("True Positive Rate (TPR)", fontsize=PLOT_LABEL_FONT_SIZE)
    plt.title(title, fontsize=PLOT_TITLE_FONT_SIZE)
    plt.legend(loc="center right", fontsize=PLOT_LEGEND_FONT_SIZE)
    plt.tick_params(axis="both", labelsize=PLOT_TICK_FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()


def get_attacks():
    """Build the attacks enabled for the checkpoint filter in this run."""
    return [
        # RandomAttack(),
        # MonteCarlo_attack(
        #     n_samples=MONTE_CARLO_N_SAMPLES,
        #     d_min_n_samples=MONTE_CARLO_D_MIN_N_SAMPLES,
        #     generation_batch_size=MONTE_CARLO_GENERATION_BATCH_SIZE,
        #     distance_metric="euclidean",
        #     candidate_batch_size=CANDIDATE_BATCH_SIZE,
        # ),
        # CriticScoreAttack(
        #     n_repeats=1,
        #     batch_size=16,
        #     pack_mode="repeat_candidate",
        # ),
        # ReconstructionLossAttack(),
        # DiscriminatorScoreAttack(
        #     discriminator_weight=1.0,
        #     class_confidence_weight=0.0,
        #     batch_size=8,
        # ),
        # DiscriminatorScoreAttack(
        #     discriminator_weight=0.0,
        #     class_confidence_weight=1.0,
        #     batch_size=8,
        #     class_confidence_mode="predicted_class",
        # ),
        # DiscriminatorScoreAttack(
        #     discriminator_weight=0.5,
        #     class_confidence_weight=0.5,
        #     batch_size=8,
        #     class_confidence_mode="predicted_class",
        # ),
        # DiscriminatorScoreAttack(
        #     discriminator_weight=0.0,
        #     class_confidence_weight=1.0,
        #     batch_size=8,
        #     class_confidence_mode="true_class",
        # ),
        # DiscriminatorScoreAttack(
        #     discriminator_weight=0.5,
        #     class_confidence_weight=0.5,
        #     batch_size=8,
        #     class_confidence_mode="true_class",
        # ),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS, timestep=0),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS, timestep=100),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS, timestep=200),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS, timestep=500),
        DiffusionLossAttack(n_repeats=DENOISING_LOSS_N_REPEATS, timestep=1000),
    ]

        
def run_attacks(
    model,
    train_path,
    attack_train_path,
    non_train_path,
    load_n=500,
    max_test_set_size=None,
    run_summary_rows=None,
    run_summary_path=None,
    quality_metrics=None,
    model_type=None,
    output_dir=results_folder,
    synthetic_data=None,
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
    quality_metrics = quality_metrics or {}
    output_dir = Path(output_dir)
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

    train_data = model.load_attack_dataset(train_path, nrows=load_n)
    attack_train_data = model.load_attack_dataset(attack_train_path, nrows=load_n)
    non_train_data = model.load_attack_dataset(non_train_path, nrows=load_n)
    train_class_ids = model.load_attack_labels(train_path, nrows=load_n)
    attack_train_class_ids = model.load_attack_labels(attack_train_path, nrows=load_n)
    non_train_class_ids = model.load_attack_labels(non_train_path, nrows=load_n)
    print(
        "Loaded attack datasets: "
        f"train={train_data.shape}/{train_data.dtype}, "
        f"eval={attack_train_data.shape}/{attack_train_data.dtype}, "
        f"test={non_train_data.shape}/{non_train_data.dtype}"
    )
    if train_class_ids is not None:
        print(
            "Loaded attack class labels: "
            f"train={train_class_ids.shape}, "
            f"eval={attack_train_class_ids.shape}, "
            f"test={non_train_class_ids.shape}"
        )

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
    train_sample_class_ids = (
        train_class_ids[idx_train] if train_class_ids is not None else None
    )
    non_train_sample_class_ids = (
        non_train_class_ids[idx_non] if non_train_class_ids is not None else None
    )
    y_train = np.ones(n, dtype=int)      # members
    y_non = np.zeros(n, dtype=int)       # non-members

    X = np.concatenate([train_samples, non_train_samples], axis=0)
    y = np.concatenate([y_train, y_non], axis=0)
    X_class_ids = None
    if train_sample_class_ids is not None and non_train_sample_class_ids is not None:
        X_class_ids = np.concatenate([train_sample_class_ids, non_train_sample_class_ids], axis=0)

    perm = rng.permutation(len(X))
    if max_test_set_size is not None:
        perm = perm[:max_test_set_size]

    test_labels = y[perm]
    print(f"Using test set size: {len(perm)}")

    aa_n = min(AA_n, len(train_samples), len(non_train_samples))

    attacks = get_attacks()

    attack_thresholds = {
        "diffusion_loss_attack": 0.99,
        "monte_carlo_attack": 0.99,
        "reconstruction_loss_attack": 0.99,
        "critic_score_attack": 0.99,
        "discriminator_score_attack_d1p00_c0p00": 0.99,
        "discriminator_score_attack_d0p00_c1p00_predicted_class": 0.99,
        "discriminator_score_attack_d0p50_c0p50_predicted_class": 0.99,
        "discriminator_score_attack_d0p00_c1p00_true_class": 0.99,
        "discriminator_score_attack_d0p50_c0p50_true_class": 0.99,
    }

    AAtr = AAte = privacy_loss = np.nan
    if RUN_AA_EVALUATION:
        if synthetic_data is not None and len(synthetic_data) < aa_n:
            raise ValueError(
                f"Synthetic cache has {len(synthetic_data)} samples; AA requires {aa_n}"
            )
        synthetic_samples = (
            synthetic_data[:aa_n]
            if synthetic_data is not None
            else model.generate(n=aa_n)
        )
        AAtr, AAte, privacy_loss = call_AA_dist_metrics(
            train_samples[:aa_n], non_train_samples[:aa_n], synthetic_samples[:aa_n]
        )
        print(
            f"AA on model {model.model_name}: AAtr: {AAtr:.4f}, "
            f"AAte: {AAte:.4f}, Privacy Loss: {privacy_loss:.4f}"
        )

    for attack_instance in attacks:
        if not attack_instance.is_attack_applicable(model):
            print(
                f"Attack {attack_instance.name} is not applicable to "
                f"model {model.model_name}. Skipping."
            )
            continue

        attack_name = attack_instance.name
        attack_display_name = attack_instance.get_display_name()
        attack_timestep = (
            "random"
            if isinstance(attack_instance, DiffusionLossAttack)
            and attack_instance.timestep is None
            else getattr(attack_instance, "timestep", "")
        )
        attack_n_repeats = getattr(attack_instance, "n_repeats", "")
        print(f"Running attack: {attack_display_name} ({attack_name})")
        fit_kwargs = {}
        if isinstance(attack_instance, MonteCarlo_attack):
            fit_kwargs["synthetic_data"] = synthetic_data
        elif isinstance(attack_instance, DiscriminatorScoreAttack):
            if attack_instance.requires_true_labels and (
                attack_train_class_ids is None or X_class_ids is None
            ):
                raise ValueError(
                    "True class labels are required for label-aware "
                    f"discriminator attack {attack_name}"
                )
            fit_kwargs["non_train_labels"] = attack_train_class_ids
        attack_instance.fit(
            non_train_data=attack_train_data,
            thr=attack_thresholds.get(
                attack_name,
                0.99 if attack_name.startswith("diffusion_loss_attack") else 0.5,
            ),
            modelWrapper=model,
            **fit_kwargs,
        )  # Fit on synthetic and non-training samples

        if attack_name.startswith("discriminator_score_attack"):
            diagnostic_n = min(
                DISCRIMINATOR_SCORE_DIAGNOSTIC_MAX_SAMPLES,
                len(train_samples),
                len(non_train_samples),
            )
            train_diagnostic_labels = (
                train_sample_class_ids[:diagnostic_n]
                if train_sample_class_ids is not None else None
            )
            non_train_diagnostic_labels = (
                non_train_sample_class_ids[:diagnostic_n]
                if non_train_sample_class_ids is not None else None
            )
            diagnostic = attack_instance.score_diagnostic(
                train_samples[:diagnostic_n],
                non_train_samples[:diagnostic_n],
                train_labels=train_diagnostic_labels,
                non_train_labels=non_train_diagnostic_labels,
                model_name=model.model_name,
                output_dir=output_dir,
            )
            print(
                "Discriminator score diagnostic "
                f"(n={diagnostic_n}, "
                f"disc_weight={diagnostic['discriminator_weight']:.2f}, "
                f"class_weight={diagnostic['class_confidence_weight']:.2f}, "
                f"class_mode={diagnostic['class_confidence_mode']}): "
                f"train_mean={diagnostic['train_mean']:.6f}, "
                f"train_median={diagnostic['train_median']:.6f}, "
                f"train_range=[{diagnostic['train_min']:.6f}, "
                f"{diagnostic['train_max']:.6f}], "
                f"non_train_mean={diagnostic['non_train_mean']:.6f}, "
                f"non_train_median={diagnostic['non_train_median']:.6f}, "
                f"non_train_range=[{diagnostic['non_train_min']:.6f}, "
                f"{diagnostic['non_train_max']:.6f}]"
            )
            print(
                "Saved per-sample discriminator score diagnostic: "
                f"{diagnostic['path']}"
            )
            print(
                "Saved discriminator score distribution plot: "
                f"{diagnostic['plot_path']}"
            )

        if attack_name == "critic_score_attack":
            diagnostic_n = min(
                CRITIC_SCORE_DIAGNOSTIC_MAX_SAMPLES,
                len(train_samples),
                len(non_train_samples),
            )
            diagnostic_synthetic_samples = (
                synthetic_data[:diagnostic_n]
                if synthetic_data is not None
                else model.generate(n=diagnostic_n)
            )
            diagnostic = attack_instance.raw_score_diagnostic(
                train_samples[:diagnostic_n],
                non_train_samples[:diagnostic_n],
                diagnostic_synthetic_samples,
                model_name=model.model_name,
                output_dir=output_dir,
            )
            print(
                "Critic raw score diagnostic "
                f"(n={diagnostic_n}, pack_mode={diagnostic['pack_mode']}): "
                f"train_mean={diagnostic['train_mean']:.6f}, "
                f"train_median={diagnostic['train_median']:.6f}, "
                f"train_range=[{diagnostic['train_min']:.6f}, "
                f"{diagnostic['train_max']:.6f}], "
                f"non_train_mean={diagnostic['non_train_mean']:.6f}, "
                f"non_train_median={diagnostic['non_train_median']:.6f}, "
                f"non_train_range=[{diagnostic['non_train_min']:.6f}, "
                f"{diagnostic['non_train_max']:.6f}], "
                f"synthetic_mean={diagnostic['synthetic_mean']:.6f}, "
                f"synthetic_median={diagnostic['synthetic_median']:.6f}, "
                f"synthetic_range=[{diagnostic['synthetic_min']:.6f}, "
                f"{diagnostic['synthetic_max']:.6f}]"
            )
            print(
                "Saved per-sample critic raw score diagnostic: "
                f"{diagnostic['path']}"
            )
            print(
                "Saved critic raw score distribution plot: "
                f"{diagnostic['plot_path']}"
            )

        if attack_name == "reconstruction_loss_attack":
            diagnostic_n = min(
                RECONSTRUCTION_LOSS_DIAGNOSTIC_MAX_SAMPLES,
                len(train_samples),
                len(non_train_samples),
            )
            diagnostic_synthetic_samples = (
                synthetic_data[:diagnostic_n]
                if synthetic_data is not None
                else model.generate(n=diagnostic_n)
            )
            diagnostic = attack_instance.reconstruction_loss_diagnostic(
                train_samples[:diagnostic_n],
                non_train_samples[:diagnostic_n],
                diagnostic_synthetic_samples,
                model_name=model.model_name,
                output_dir=output_dir,
            )
            print(
                "Reconstruction loss diagnostic "
                f"(n={diagnostic_n}): "
                f"train_mean={diagnostic['train_mean']:.6f}, "
                f"train_median={diagnostic['train_median']:.6f}, "
                f"train_range=[{diagnostic['train_min']:.6f}, "
                f"{diagnostic['train_max']:.6f}], "
                f"non_train_mean={diagnostic['non_train_mean']:.6f}, "
                f"non_train_median={diagnostic['non_train_median']:.6f}, "
                f"non_train_range=[{diagnostic['non_train_min']:.6f}, "
                f"{diagnostic['non_train_max']:.6f}], "
                f"synthetic_mean={diagnostic['synthetic_mean']:.6f}, "
                f"synthetic_median={diagnostic['synthetic_median']:.6f}, "
                f"synthetic_range=[{diagnostic['synthetic_min']:.6f}, "
                f"{diagnostic['synthetic_max']:.6f}]"
            )
            print(
                "Saved per-sample reconstruction loss diagnostic: "
                f"{diagnostic['path']}"
            )
            print(
                "Saved reconstruction loss distribution plot: "
                f"{diagnostic['plot_path']}"
            )

        if (
            RUN_DENOISING_LOSS_DIAGNOSTICS
            and attack_name.startswith("diffusion_loss_attack")
        ):
            diagnostic_n = min(
                DIFFUSION_LOSS_DIAGNOSTIC_MAX_SAMPLES,
                len(train_samples),
                len(non_train_samples),
            )
            diagnostic_synthetic_samples = (
                synthetic_data[:diagnostic_n]
                if synthetic_data is not None
                else model.generate(n=diagnostic_n)
            )
            diagnostic = attack_instance.diffusion_loss_diagnostic(
                train_samples[:diagnostic_n],
                non_train_samples[:diagnostic_n],
                diagnostic_synthetic_samples,
                model_name=model.model_name,
                output_dir=output_dir,
            )
            print(
                "Denoising-loss diagnostic "
                f"(n={diagnostic_n}, repeats={diagnostic['n_repeats']}): "
                f"train_mean={diagnostic['train_mean']:.6f}, "
                f"non_train_mean={diagnostic['non_train_mean']:.6f}, "
                f"synthetic_mean={diagnostic['synthetic_mean']:.6f}"
            )
            print(f"Saved per-sample denoising-loss diagnostic: {diagnostic['path']}")
            print(f"Saved denoising-loss distribution plot: {diagnostic['plot_path']}")

        predictions, scores, raw_scores = predict_attack_in_batches(
            attack_instance,
            X,
            perm,
            TEST_PREDICTION_BATCH_SIZE,
            labels=y,
            predict_kwargs={
                "class_ids": X_class_ids,
            } if isinstance(attack_instance, DiscriminatorScoreAttack) else None,
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
        eval_metrics_threshold_0_5 = evaluate_predictions(
            test_labels,
            predictions_threshold_0_5,
            scores,
        )
        eval_metrics["accuracy_threshold_0_5"] = (
            eval_metrics_threshold_0_5["accuracy"]
        )
        optimal_accuracy, optimal_accuracy_threshold = optimal_accuracy_from_scores(
            test_labels,
            scores,
        )
        eval_metrics["optimal_accuracy"] = optimal_accuracy
        eval_metrics["optimal_accuracy_threshold"] = optimal_accuracy_threshold
        eval_metrics["AA_train"] = AAtr
        eval_metrics["AA_test"] = AAte
        eval_metrics["AA_samples"] = aa_n if RUN_AA_EVALUATION else 0
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
        threshold_label = (
            f"{prediction_threshold:.4f}"
            if prediction_threshold is not None else "attack default"
        )

        print(
            f"Attack: {attack_display_name}, "
            f"Accuracy @ threshold {threshold_label}: "
            f"{eval_metrics['accuracy']:.4f}, "
            f"Accuracy @ threshold 0.5: "
            f"{eval_metrics_threshold_0_5['accuracy']:.4f}, "
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
            output_dir,
            sample_indices=perm,
            raw_scores=raw_scores,
            inside_d_min_percent=inside_d_min_percent,
            threshold=prediction_threshold,
            raw_score_threshold=raw_prediction_threshold,
            raw_score_threshold_0_5=raw_threshold_0_5,
            predictions_threshold_0_5=predictions_threshold_0_5,
            timestep=attack_timestep,
            n_repeats=attack_n_repeats,
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
            "display_name": attack_display_name,
            "timestep": attack_timestep,
            "n_repeats": attack_n_repeats,
            "quality_metrics": dict(quality_metrics),
            "metrics": eval_metrics,
            "true_labels": test_labels,
            "predictions": predictions,
            "predictions_threshold_0_5": predictions_threshold_0_5,
            "scores": scores,
            "raw_scores": raw_scores,
            "inside_d_min_percent": inside_d_min_percent,
        }

        if run_summary_rows is not None and run_summary_path is not None:
            new_summary_rows = build_run_summary_rows(
                model.model_name,
                {attack_name: results[attack_name]},
                quality_metrics=quality_metrics,
                model_type=model_type,
            )
            run_summary_rows.extend(new_summary_rows)
            append_run_summary_rows(new_summary_rows, run_summary_path)
            print(f"Updated run summary CSV: {run_summary_path}")

        plot_roc_curve(
            test_labels,
            scores,
            title=f"ROC Curve - {attack_display_name}\n{model.get_model_title_suffix()}",
            show=False,
            save_path=output_dir / f"{model.model_name}_{attack_name}_roc.png"

        )

    return results


# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)
run_output_dir = create_run_output_dir(results_folder)
synthetic_cache_dir = Path(results_folder) / "tmp_synthetic"
print(f"Writing run artifacts under: {run_output_dir}")
print(f"Using shared synthetic cache under: {synthetic_cache_dir}")

model_files = get_model_files(models_folder, MODEL_NAME_REGEX)
print(f"Found {len(model_files)} model checkpoint(s) to attack.")
if model_files:
    print("Model execution order:")
    for model_index, model_file in enumerate(model_files, start=1):
        print(f"  {model_index}. {model_file.name}")
print(f"Using device: {DEVICE}, GPU memory: {GPU_MEMORY_GB:.2f} GB")

run_summary_rows = []
run_summary_path = create_run_summary_path(run_output_dir)
initialize_run_summary(run_summary_path)
print(f"Writing incremental run summary CSV: {run_summary_path}")

# Loop over selected models in the folder
for model_file in model_files:
    print(f"\nProcessing model: {model_file.name}")
    
    # Build wrapper from model file
    model = create_model_wrapper(
        file_name=str(model_file),
        device=DEVICE,
        generation_batch_size=MODEL_GENERATION_BATCH_SIZE,
    )
    if hasattr(model, "generation_batch_size"):
        print(f"Using model generation batch size: {model.generation_batch_size}")
    print(f"Using wrapper model architecture: {model.get_model_architecture()}")

    base = model_checkpoint_base(model_file)
    attack_dataset_paths = model.get_attack_dataset_paths(models_folder, base)
    train_path = attack_dataset_paths["train"]
    eval_path = attack_dataset_paths["eval"]
    test_path = attack_dataset_paths["test"]
    print(f"Dataset paths: Train: {train_path}, Eval: {eval_path}, Test: {test_path}")

    shared_synthetic_data = None
    if RUN_MODEL_QUALITY_EVALUATION or RUN_AA_EVALUATION or RUN_DENOISING_LOSS_DIAGNOSTICS:
        shared_synthetic_n_samples = max(
            100,
            MODEL_QUALITY_N_SAMPLES if RUN_MODEL_QUALITY_EVALUATION else 0,
            DIFFUSION_LOSS_DIAGNOSTIC_MAX_SAMPLES
            if RUN_DENOISING_LOSS_DIAGNOSTICS else 0,
        )
        cache_key = f"{base}_{model_file.stat().st_mtime_ns}"
        expected_feature_shape = tuple(
            model.load_attack_dataset(test_path, nrows=1).shape[1:]
        )
        shared_synthetic_data = model.get_synthetic_data(
            n=shared_synthetic_n_samples,
            cache_dir=synthetic_cache_dir,
            cache_key=cache_key,
            batch_size=SYNTHETIC_CACHE_BATCH_SIZE,
            expected_feature_shape=expected_feature_shape,
        )

    quality_metrics = {}
    if RUN_MODEL_QUALITY_EVALUATION:
        print(f"Running quality evaluation on {model_file.name}...")
        quality_results = evaluate_model_quality(
            model=model,
            real_path=test_path,
            output_dir=run_output_dir,
            n_samples=MODEL_QUALITY_N_SAMPLES,
            pca_plot_samples=MODEL_QUALITY_PCA_PLOT_SAMPLES,
            aa_samples=MODEL_QUALITY_AA_SAMPLES,
            pca_components=MODEL_QUALITY_CLASSIFIER_PCA_COMPONENTS,
            seed=MODEL_QUALITY_RANDOM_SEED,
            synthetic_data=shared_synthetic_data,
        )
        quality_metrics = quality_results["metrics"]
        print(
            f"Quality metrics for {model.model_name}: "
            f"AF_MAE={quality_metrics['allele_frequency_mae']:.4f}, "
            f"AF_Pearson={quality_metrics['allele_frequency_pearson']:.4f}, "
            f"W_distance={quality_metrics['pca_wasserstein_distance']:.4f}, "
            f"AA={quality_metrics.get('aa', np.nan):.4f}, "
            f"real_vs_synth_AUC={quality_metrics['real_vs_synthetic_classifier_auc']:.4f}"
        )

    # Run attacks
    print(f"Running attacks on {model_file.name}...")
    attack_results = run_attacks(
        model,
        train_path,
        eval_path,
        test_path,
        max_test_set_size=MAX_TEST_SET_SIZE,
        run_summary_rows=run_summary_rows,
        run_summary_path=run_summary_path,
        quality_metrics=quality_metrics,
        model_type=model.get_model_type(),
        output_dir=run_output_dir,
        synthetic_data=shared_synthetic_data,
    )

    metric_paths = save_attack_results(model.model_name, attack_results, run_output_dir)
    for metric_path in metric_paths:
        print(f"Saved metrics log: {metric_path}")

    if hasattr(model, "cleanup"):
        model.cleanup()
    del model, attack_results, quality_metrics, shared_synthetic_data
    if "quality_results" in locals():
        del quality_results
    gc.collect()

print("\nAll models processed successfully!")
