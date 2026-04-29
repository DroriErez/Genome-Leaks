"""Membership inference/disclosure evaluation for synthetic DNA models.

Contains routines to run membership disclosure experiments for varying fractions of copied sequence.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from generators import DNAGenerator, SyntheticDNAGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from models import MemberDisclosureDiscriminator


def evaluate_member_disclosure(N, S, n_trials=1000, fractions=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2], show_plot=True):
    # Call DNA model evaluation function
    generator = DNAGenerator(N)

    # Generate training and test sets
    training_set = [generator.generate() for _ in range(S)]

    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    tprs = []
    fprs = []

    p_coverage_Training_Set = 0.99
    f_coverage = int(np.log(S) + np.log(1 / (1 - p_coverage_Training_Set)))

    # Evaluate membership inference accuracy for different fractions of copied sequence
    for frac in fractions:
        n = int(N * frac)
        overlap_limit = n
        synth_model = SyntheticDNAGenerator(training_set, n, generator)
        synthetic_set = [synth_model.generate() for _ in range(f_coverage*S)]
        discriminator = MemberDisclosureDiscriminator(synthetic_set, overlap_limit)

        correct = 0
        y_true = []
        y_pred = []
        for i in range(n_trials):
            # Membership inference experiment: 0.5 probability pick from training set, 0.5 generate synthetic DNA
            if np.random.rand() < 0.5:
                dna = np.random.choice(training_set)
                label = 1
            else:
                dna = generator.generate()
                label = 0
            # Predict membership
            pred = discriminator.is_in_training_set(dna)
            # Record results
            y_true.append(label)
            y_pred.append(int(pred))
            if pred == bool(label):
                correct += 1
        # Calculate metrics
        accuracy = correct / n_trials
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        tprs.append(tpr)
        fprs.append(fpr)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fraction: {frac:.2f}, Membership inference accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")

    if show_plot:
        # Plotting the membership inference results
        plt.figure(figsize=(10, 6))
        plt.plot(fractions, accuracies, marker='o', label='Accuracy')
        plt.plot(fractions, tprs, marker='s', label='True Positive Rate')
        plt.plot(fractions, fprs, marker='^', label='False Positive Rate')

        # plt.plot(fractions, precisions, marker='s', label='Precision')
        # plt.plot(fractions, recalls, marker='^', label='Recall')
        # plt.plot(fractions, f1s, marker='d', label='F1 Score')
        plt.xlabel('Fraction of Sequence Copied (n/N)')
        plt.ylabel('Metric Value')
        plt.title('Membership Inference Metrics vs. Copied Fraction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=True)

    return accuracies, precisions, recalls, f1s
