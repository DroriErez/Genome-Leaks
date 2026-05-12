import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt



PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from attacks.random.random_attack import RandomAttack
from attacks.MonteCarlo.MonteCarlo_attack import MonteCarlo_attack
from attacks.RecostructionAttack.reconstruction_attack import ReconstructionAttack
from models.Gen_Model_Wrapper import GenomeGenerativeModelWrapper 
from models.models_factory import create_model_wrapper
from AA_Simulation.measurements import calc_AA


# Define paths
models_folder = "attacks/models_to_attack"
results_folder = "attacks/results"

os.makedirs(results_folder, exist_ok=True)


def save_attack_metrics(model_name, attack_results, output_dir):
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
            "balanced_accuracy": attack_metrics["balanced_accuracy"],
            "precision": attack_metrics["precision"],
            "recall": attack_metrics["recall"],
            "specificity": attack_metrics["specificity"],
            "f1_score": attack_metrics["f1_score"],
            "auc": attack_metrics["auc"],
            "tpr_01": attack_metrics["tpr_01"],
            "tpr_001": attack_metrics["tpr_001"],
        }
        summary_data_list.append(summary_data)
    out_path = Path(output_dir) / f"{model_name}_metrics.csv"
    pd.DataFrame(summary_data_list).to_csv(out_path, index=False)
    return out_path


def save_attack_predictions(model_name, attack_name, true_labels, predictions, scores, output_dir):
    """Save attack predictions and scores to CSV."""
    out_path = Path(output_dir) / f"{model_name}_{attack_name}_predictions.csv"
    pd.DataFrame({
        "true_label": true_labels,
        "prediction": predictions,
        "score": scores
    }).to_csv(out_path, index=False)
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

def plot_roc_curve(y_true, scores, title="ROC Curve", show=True, save_path=None):
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

    plt.show(block=show)
    plt.close()

        
def run_attacks(model, train_path, attack_train_path, non_train_path, load_n=100) -> dict:
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
    load_n = 500
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

    print(f"Running attacks on model: {model.model_name}, with number of samples: {load_n}")

    # Load a sample of test data in batches for efficiency

    train_path_df = pd.read_csv(train_path, sep=' ', header=None, nrows=load_n)  # Load first 1000 rows as sample
    train_data = train_path_df.drop(train_path_df.columns[0:2], axis=1).values

    attack_train_path_df = pd.read_csv(attack_train_path, sep=' ', header=None, nrows=load_n)  # Load first 1000 rows as sample
    attack_train_data = attack_train_path_df.drop(attack_train_path_df.columns[0:2], axis=1).values

    non_train_path_df = pd.read_csv(non_train_path, sep=' ', header=None, nrows=load_n)  # Load first 1000 rows as sample
    non_train_data = non_train_path_df.drop(non_train_path_df.columns[0:2], axis=1).values

    rng = np.random.default_rng(42)

    n = min(len(train_data), len(non_train_data), load_n)

    idx_train = rng.choice(len(train_data), size=n, replace=False)
    idx_non = rng.choice(len(non_train_data), size=n, replace=False)

    train_samples = train_data[idx_train]
    non_train_samples = non_train_data[idx_non]
    y_train = np.ones(n, dtype=int)      # members
    y_non = np.zeros(n, dtype=int)       # non-members

    X = np.concatenate([train_samples, non_train_samples], axis=0)
    y = np.concatenate([y_train, y_non], axis=0)

    perm = rng.permutation(len(X))

    test_data = X[perm]
    test_labels = y[perm]

    # Generate synthetic samples using the wrapper model
    synthetic_samples = model.generate(n=AA_n)

    attacks = [
        # RandomAttack(),
        MonteCarlo_attack(n_samples=100000, distance_metric="euclidean"),
        # ReconstructionAttack()
    ]  # Attack instances

    attack_thresholds = {
        "monte_carlo_attack": 0.99,
        "reconstruction_attack": 0.5,
    }


    AAtr,AAte, privacy_loss = call_AA_dist_metrics(train_samples[:AA_n], non_train_samples[:AA_n], synthetic_samples[:AA_n])
    print(f"AA on model {model.model_name}: AAtr: {AAtr:.4f}, AAte: {AAte:.4f}, Privacy Loss: {privacy_loss:.4f}")   

    for attack_instance in attacks:
        if not attack_instance.is_attack_applicable(model):
            print(f"Attack {attack_instance.name} is not applicable to model {model.model_name}. Skipping.")
            continue

        attack_name = attack_instance.name
        print(f"Running attack: {attack_name}")
        attack_instance.fit(
            non_train_data=non_train_samples,
            thr=attack_thresholds.get(attack_name, 0.5),
            modelWrapper=model,
        )  # Fit on synthetic and non-training samples

        predictions, scores = attack_instance.predict(test_data)

        # print("test_label,predicted_label,score")
        # for test_label, predicted_label, score in zip(test_labels, predictions, scores):
        #     print(f"{int(test_label)},{int(predicted_label)},{score:.6f}")

        # print("threshold 99%", np.percentile(scores, 99))
        # print("threshold 99.9%", np.percentile(scores, 99.9))


        eval_metrics = evaluate_predictions(test_labels, predictions, scores)

        print(
            f"Attack: {attack_name}, "
            f"Accuracy: {eval_metrics['accuracy']:.4f}, "
            f"AUC: {eval_metrics['auc']:.4f}, "
            f"TPR@FPR=0.01: {eval_metrics['tpr_01']:.4f}, "
            f"TPR@FPR=0.001: {eval_metrics['tpr_001']:.4f}"
        )

        prediction_path = save_attack_predictions(model.model_name, attack_name, test_labels, predictions, scores, results_folder)

        results[attack_name] = {
            "metrics": eval_metrics,
            "true_labels": test_labels,
            "predictions": predictions,
            "scores": scores,
        }

        plot_roc_curve(
            test_labels,
            scores,
            title=f"ROC Curve - {attack_name}",
            show=True,
            save_path=f"{results_folder}/{model.model_name}_{attack_name}_roc.png"

        )

    return results


# Create results folder if it doesn't exist
os.makedirs(results_folder, exist_ok=True)

# Loop over all models in the folder
for model_file in Path(models_folder).glob("*.pth"):
    if model_file.is_file():
        print(f"\nProcessing model: {model_file.name}")
        
        # Build wrapper from model file
        model = create_model_wrapper(file_name=str(model_file))
        print(f"Using wrapper model architecture: {model.get_model_architecture()}")

        # Generate dataset paths based on model file name
        base = model_file.stem
        train_path = f"{models_folder}/{base}_train.hapt"
        eval_path = f"{models_folder}/{base}_eval.hapt"
        test_path = f"{models_folder}/{base}_test.hapt"
        print(f"Dataset paths: Train: {train_path}, Eval: {eval_path}, Test: {test_path}")

        # Run attacks
        print(f"Running attacks on {model_file.name}...")
        attack_results = run_attacks(model, train_path, eval_path, test_path)

        metric_path = save_attack_metrics(model.model_name, attack_results, results_folder)

print("\nAll models processed successfully!")
