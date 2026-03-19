import numpy as np
import pandas as pd

# ---- parameters ----
input_file = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_full_test.hapt"     # your original file
train_file = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_test.hapt"
test_file  = "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_eval.hapt"
train_frac = 0.5             # 50% train, 50% test
random_seed = 42

# ---- load ----
df = pd.read_csv(input_file, sep=r"\s+", header=None)

# ---- shuffle indices ----
n = len(df)
rng = np.random.default_rng(random_seed)
perm = rng.permutation(n)

n_train = int(train_frac * n)
train_idx = perm[:n_train]
test_idx  = perm[n_train:]

train_df = df.iloc[train_idx]
test_df  = df.iloc[test_idx]

# ---- save back in same format ----
train_df.to_csv(train_file, sep=" ", header=False, index=False)
test_df.to_csv(test_file, sep=" ", header=False, index=False)

print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")

