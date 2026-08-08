# PCA-DM for 1000 Genomes haplotypes

This version trains an unconditional diffusion model on a PCA representation of
binary 1000 Genomes haplotypes. It has no phenotype, sex, height, or population
label dependency.

From the repository root, run:

```powershell
python models/PCA_DM/human/MULTI/train.py
```

The default input is
`Data/1000G_real_genomes/10K_SNP_1000G_real_train.hapt`. A different binary
haplotype file can be supplied with `--data-path`:

```powershell
python models/PCA_DM/human/MULTI/train.py --data-path Data/1000G_real_genomes/805_SNP_1000G_real_train.hapt
```

Each input row must have two metadata fields followed by space-separated binary
SNP calls. PCA is fitted from the training haplotypes and saved with each run.
Generated files use the same two-metadata-column `.hapt` format and contain only
0/1 SNP calls.

Useful options include `--pca-components`, `--epochs`, `--batch-size`,
`--checkpoint-every`, `--generate-samples`, and `--output-dir`. Run artifacts
are placed directly under `checkpoints/` by default.

When `--batch-size` is omitted, the existing GPU-memory helper selects it and
the trainer selects conservative NoisePredictor hidden dimensions from the same
detected memory. The selected device, GPU memory, hidden dimensions, and batch
size are printed before data loading. Passing `--batch-size` overrides only the
automatic batch size.

Every checkpoint saves files whose names contain the epoch, including the
fitted PCA components, mean, explained variance, and training allele
frequencies:

- a compact FP16 inference checkpoint containing model weights and architecture
  metadata (optimizer and scheduler states are intentionally omitted);
- generated binary haplotypes in `.hapt` format;
- real and synthetic allele frequencies plus a comparison plot;
- a two-component PCA embedding CSV and plot;
- quality metrics in JSON and CSV formats.

`PCA_DM_checkpoint_results.csv` records the training loss, evaluation loss,
allele-frequency MAE, and PCA Wasserstein distance at every checkpoint.
`PCA_DM_training_losses.csv` records the training loss after every epoch.
Evaluation loss is the DDPM noise-prediction loss on test
haplotypes transformed with the training PCA.

Evaluation uses `10K_SNP_1000G_real_test.hapt` by default. Override it with
`--evaluation-data-path`. Scalar evaluation results are also printed after each
checkpoint. The AA metric is skipped with an explanatory status when its Numba
dependency is unavailable; all other evaluations continue.

Training can load model weights automatically from the highest
`PCA_DM_model_<epoch>.pth` in the output directory. Because checkpoints are
optimized for inference, continuation starts with a new optimizer and learning
rate scheduler. Use `--no-resume` to begin again at epoch 1 without loading the
weights; existing checkpoint files are not deleted.
