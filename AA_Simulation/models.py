"""Simple membership disclosure models used in synthetic DNA privacy experiments."""

import measurements

class MemberDisclosureDiscriminator:
    def __init__(self, synthetic_set, overlap_limit):
        """
        synthetic_set: list of DNA strings (same length)
        overlap_limit: int, minimum length of overlap to decide 'in training set'
        """
        self.synthetic_set = synthetic_set
        self.overlap_limit = overlap_limit

    def max_overlap_length(self, dna1, dna2):
        """
        Returns the length of the largest contiguous matching substring between dna1 and dna2,
        comparing only at the same positions.
        """
        max_len = 0
        current_len = 0
        for a, b in zip(dna1, dna2):
            if a == b:
                current_len += 1
                if current_len > max_len:
                    max_len = current_len
            else:
                current_len = 0
        return max_len

    def is_in_training_set(self, dna):
        """
        Returns True if the DNA has an overlapping section >= overlap_limit with any synthetic set member.
        """
        for synth_dna in self.synthetic_set:
            if self.max_overlap_length(dna, synth_dna) >= self.overlap_limit:
                return True
        return False

class MemberDisclosureDiscriminatorLRT:
    def __init__(self, synthetic_set, p_0, tao):
        """
        synthetic_set: list of DNA strings (same length)
        tao: float, threshold for likelihood ratio test to decide 'in training set'
        """
        self.synthetic_set = synthetic_set
        self.p0 = p_0
        self.p1 = measurements.estimate_probs(synthetic_set)
        self.tao = tao

    def is_in_training_set(self, dna):
        """
        Returns True if LRT(dna) >= tao, meaning strong evidence
        that dna belongs to the training set.        
        """
        # if measurements.compute_LRT(dna, self.p1, self.p0) >= self.tao:
        #     return True
        if measurements.compute_LRT_best_synth(dna, self.synthetic_set, self.p0) >= self.tao:
            return True
        
        return False
