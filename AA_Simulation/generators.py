"""DNA generation utilities for reference-based mutation and synthetic copying.

Includes:
- DNAGenerator: random reference sequence with per-base mutation generation.
- SyntheticDNAGenerator: generates synthetic sequences by copying local segments from training sequences.
"""

import numpy as np

class DNAGenerator:
    """Generate random DNA strings by mutating a reference genome."""

    def __init__(self, N, mutation_prob=0.03):
        """Initialize generator.

        Args:
            N (int): Genome length.
            mutation_prob (float): Per-base mutation probability.
        """
        self.N = N
        self.mutation_prob = mutation_prob
        self.reference_genome = self._generate_reference_genome()

        self.reference_p = np.full(self.N, self.mutation_prob)
        self.reference_p_base_is_1 = np.array([
            1 - self.mutation_prob if b == '1' else self.mutation_prob
            for b in self.reference_genome
        ])

    def _generate_reference_genome(self):
        """Generate a random reference genome string of length N."""
        return ''.join(np.random.choice(['0', '1'], size=self.N))

    def generate(self):
        """Generate one mutated genome using reference-based flips."""
        ref = np.array(list(self.reference_genome), dtype='U1')
        mutations = np.random.rand(self.N) < self.mutation_prob
        mutated = np.where(mutations, np.where(ref == '0', '1', '0'), ref)
        return ''.join(mutated)


class SyntheticDNAGenerator:
    """Generate a synthetic genome by copying a segment from training samples."""

    def __init__(self, training_set, n, generator: 'DNAGenerator'):
        """Initialize synthetic generator.

        Args:
            training_set (List[str]): DNA strings with length generator.N.
            n (int): Number of bases to copy from one training sample per sequence.
            generator (DNAGenerator): Underlying random base generator.
        """
        self.training_set = training_set
        self.n = n
        self.generator = generator
        self.N = generator.N

    def generate(self):
        """Generate one synthetic DNA string by mixing random base with copied subsequence."""
        base_dna = list(self.generator.generate())
        sample = np.random.choice(self.training_set)
        if self.N - self.n >= 0:
            start = np.random.randint(0, self.N - self.n + 1)
            base_dna[start:start+self.n] = list(sample[start:start+self.n])
        return ''.join(base_dna)
        self.N = N
        self.mutation_prob = mutation_prob
        self.reference_genome = self._generate_reference_genome()

        self.reference_p = np.full(self.N, self.mutation_prob)
        # Calculate the probability of '1' at each position in the reference genome
        self.reference_p_base_is_1 = np.array([
            1 - self.mutation_prob if b == '1' else self.mutation_prob
            for b in self.reference_genome
        ])
    def _generate_reference_genome(self):
        # Generate a random genome of length N with '0' and '1'
        return ''.join(np.random.choice(['0', '1'], size=self.N))

    def generate(self):
        # Generate a mutated genome based on the reference genome
        ref = np.array(list(self.reference_genome), dtype='U1')
        mutations = np.random.rand(self.N) < self.mutation_prob
        # Flip '0' to '1' and '1' to '0' where mutation occurs
        mutated = np.where(mutations, np.where(ref == '0', '1', '0'), ref)
        return ''.join(mutated)

class SyntheticDNAGenerator:
    def __init__(self, training_set, n, generator: 'DNAGenerator'):
        """
        training_set: list of DNA strings (same length as generator.N)
        n: length of sequence to copy from training set
        generator: instance of DNAGenerator
        """
        self.training_set = training_set
        self.n = n
        self.generator = generator
        self.N = generator.N

    def generate(self):
        # Generate base DNA using the generator
        base_dna = list(self.generator.generate())
        # Choose a random training sample
        sample = np.random.choice(self.training_set)
        # Choose a random start position for the sequence to copy
        if self.N - self.n >= 0:
            start = np.random.randint(0, self.N - self.n + 1)
            # Copy n-length sequence from sample to base_dna
            base_dna[start:start+self.n] = list(sample[start:start+self.n])
        return ''.join(base_dna)