import pandas as pd

files = ["Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_train.hapt", 
         "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_test.hapt",
         "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED_eval.hapt",
         "Data/1000G_real_genomes/10K_SNP_1000G_real_PADDED.hapt"]
for path in files:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # df = pd.read_hdf(path, key="df1")
    n_dna = len(df)          # or df.shape[0]

    print("Number of DNAs:", n_dna, "for file:", path)