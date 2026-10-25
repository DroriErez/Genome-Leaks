"""Input dimensions for the bundled 1000 Genomes 10K implementation."""

# 1000 Genomes dataset/architecture assumptions. Review and change these together
# when changing the input SNP count, padding layout, or dataset origin. The
# biological SNPs must come first, followed by trailing zero padding. A different
# padded length must also match the convolution architecture and trained weights;
# changing these constants alone does not resize an existing checkpoint.
UNPADDED_SNP_COUNT = 10000
PADDED_GENOME_LENGTH = 16383
CONV_LATENT_DEPTH_FACTOR = 12  # padded length = latent_size * 2**depth - 1
