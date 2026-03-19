# AA_Simulation

This folder contains synthetic DNA privacy experiments and metrics used in this project.

## Main Scripts

- `Synthetic_Pseudo_DNA_GEN.py`: top-level evaluation script that runs:
  - `evaluate_AA_dist_metrics` (AA + privacy loss vs copied fraction)
  - `evaluate_AA_jaccard_similarities_metrics` (combined AA + weighted Jaccard metrics)
  - `evaluate_member_disclosure` (membership inference metrics)

- `Half_Circle_AA_Simulation.py`: toy 2D synthetic half-circle data and AA visualization.

## Core modules

- `generators.py`: synthetic DNA generators (mutation & copy-based synthetic generation).
- `measurements.py`: privacy metrics utilities (AA, jaccard similarity, rolling windows, p-values).
- `jaccard_z.py`: Jaccard z-score evaluation logic.
- `AA_dist.py`: AA distance-based simulation with plotting.
- `AA_jaccard_similarity.py`: AA using weighted Jaccard similarity and threshold metrics.
- `member_disclosure.py`: membership inference evaluation and classification metrics.
- `models.py`: simple membership inference discriminators.

## Running

From the `AA_Simulation` directory:

```bash
python Synthetic_Pseudo_DNA_GEN.py
```

If imports fail, run with `PYTHONPATH` including this folder:

```bash
cd AA_Simulation
set PYTHONPATH=%CD%
python Synthetic_Pseudo_DNA_GEN.py
```

## Notes

- Use smaller `N` and `S` when iterating quickly.
- Metrics are deterministic only if seed is fixed (set `np.random.seed(42)`).
