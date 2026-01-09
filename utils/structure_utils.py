import torch
import numpy as np

"""
structure_utils.py

Utility functions for parsing RNA secondary structure in dot-bracket notation
and validating base pairs according to canonical and wobble pairing rules.

Allowed pairs:
- G-C (canonical)
- A-U (canonical)
- G-U (wobble)
"""

# Valid base pairs (bidirectional)
VALID_PAIRS = {
    ('G', 'C'), ('C', 'G'),
    ('A', 'U'), ('U', 'A'),
    ('G', 'U'), ('U', 'G')  # Wobble pairs
}

# Nucleotide to index mapping
NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_NUC = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def parse_dot_bracket(structure):
    """
    Parse dot-bracket notation to find paired positions.
    
    Args:
        structure: String in dot-bracket notation (e.g., "(((...)))")
    
    Returns:
        List of tuples (i, j) where i < j representing paired positions
    """
    pairs = []
    stack = []
    
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))  # (opening_pos, closing_pos)
    
    return pairs


def get_pair_matrix(structure, seq_len):
    """
    Create a binary matrix indicating paired positions.
    
    Args:
        structure: Dot-bracket string
        seq_len: Length of sequence
    
    Returns:
        numpy array of shape (seq_len, seq_len) with 1s at paired positions
    """
    pair_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
    pairs = parse_dot_bracket(structure)
    
    for i, j in pairs:
        if i < seq_len and j < seq_len:
            pair_matrix[i, j] = 1.0
            pair_matrix[j, i] = 1.0
    
    return pair_matrix


def encode_structure(structure, max_len=None):
    """
    Encode dot-bracket structure to numerical representation.
    
    '.' -> 0 (unpaired)
    '(' -> 1 (opening)
    ')' -> 2 (closing)
    
    Args:
        structure: Dot-bracket string
        max_len: Maximum length (for padding)
    
    Returns:
        numpy array of encoded structure
    """
    encoding = {'.': 0, '(': 1, ')': 2, 'N': 0}  # N treated as unpaired
    
    if max_len is None:
        max_len = len(structure)
    
    encoded = np.zeros(max_len, dtype=np.float32)
    for i, char in enumerate(structure[:max_len]):
        encoded[i] = encoding.get(char, 0)
    
    return encoded


def one_hot_structure(structure, max_len=None):
    """
    One-hot encode the structure.
    
    Args:
        structure: Dot-bracket string
        max_len: Maximum length (for padding)
    
    Returns:
        numpy array of shape (max_len, 3) - one-hot encoded structure
    """
    if max_len is None:
        max_len = len(structure)
    
    one_hot = np.zeros((max_len, 3), dtype=np.float32)
    encoding = {'.': 0, '(': 1, ')': 2, 'N': 0}
    
    for i, char in enumerate(structure[:max_len]):
        idx = encoding.get(char, 0)
        one_hot[i, idx] = 1.0
    
    # Pad remaining positions as unpaired
    for i in range(len(structure), max_len):
        one_hot[i, 0] = 1.0
    
    return one_hot


def is_valid_pair(nuc1, nuc2):
    """Check if two nucleotides form a valid pair."""
    return (nuc1, nuc2) in VALID_PAIRS


def calculate_pair_accuracy(sequence, structure):
    """
    Calculate the percentage of correctly paired positions.
    
    Args:
        sequence: RNA sequence string (A, C, G, U)
        structure: Dot-bracket structure string
    
    Returns:
        float: Percentage of valid pairs (0-100)
    """
    pairs = parse_dot_bracket(structure)
    
    if len(pairs) == 0:
        return 100.0  # No pairs to validate
    
    valid_count = 0
    for i, j in pairs:
        if i < len(sequence) and j < len(sequence):
            nuc1 = sequence[i]
            nuc2 = sequence[j]
            if is_valid_pair(nuc1, nuc2):
                valid_count += 1
    
    return (valid_count / len(pairs)) * 100.0


def calculate_pair_accuracy_batch(sequences_one_hot, structures, device='cpu'):
    """
    Calculate pair accuracy for a batch of sequences.
    
    Args:
        sequences_one_hot: Tensor of shape (batch_size, seq_len, 4)
        structures: List of dot-bracket strings
        device: torch device
    
    Returns:
        float: Average pair accuracy across batch
    """
    batch_size = sequences_one_hot.shape[0]
    total_accuracy = 0.0
    
    # Convert one-hot to sequences
    sequences = one_hot_to_sequences(sequences_one_hot)
    
    for i in range(batch_size):
        seq = sequences[i]
        struct = structures[i] if i < len(structures) else ''
        total_accuracy += calculate_pair_accuracy(seq, struct)
    
    return total_accuracy / batch_size


def one_hot_to_sequences(one_hot_tensor):
    """
    Convert one-hot encoded tensor to sequence strings.
    
    Args:
        one_hot_tensor: Tensor of shape (batch_size, seq_len, 4)
    
    Returns:
        List of sequence strings
    """
    sequences = []
    batch_size = one_hot_tensor.shape[0]
    
    for i in range(batch_size):
        seq = []
        for j in range(one_hot_tensor.shape[1]):
            idx = torch.argmax(one_hot_tensor[i, j]).item()
            seq.append(IDX_TO_NUC[idx])
        sequences.append(''.join(seq))
    
    return sequences


def calculate_pairing_penalty(sequences_one_hot, structures, device='cpu'):
    """
    Calculate a differentiable penalty for incorrect pairings.
    
    Args:
        sequences_one_hot: Tensor of shape (batch_size, seq_len, 4) - soft probabilities
        structures: List of dot-bracket strings
        device: torch device
    
    Returns:
        Tensor: Penalty value (lower is better)
    """
    batch_size = sequences_one_hot.shape[0]
    seq_len = sequences_one_hot.shape[1]
    
    # Create valid pair probability matrix
    # For each position pair (i, j), calculate P(valid pair)
    # P(valid) = P(G,C) + P(C,G) + P(A,U) + P(U,A) + P(G,U) + P(U,G)
    
    total_penalty = torch.tensor(0.0, device=device)
    
    for b in range(batch_size):
        struct = structures[b] if b < len(structures) else ''
        pairs = parse_dot_bracket(struct)
        
        if len(pairs) == 0:
            continue
        
        for i, j in pairs:
            if i >= seq_len or j >= seq_len:
                continue
            
            prob_i = sequences_one_hot[b, i]  # (4,) - probabilities for position i
            prob_j = sequences_one_hot[b, j]  # (4,) - probabilities for position j
            
            # Calculate probability of valid pairing
            # G-C: prob_i[G] * prob_j[C] + prob_i[C] * prob_j[G]
            # A-U: prob_i[A] * prob_j[U] + prob_i[U] * prob_j[A]
            # G-U: prob_i[G] * prob_j[U] + prob_i[U] * prob_j[G]
            
            p_gc = prob_i[2] * prob_j[1] + prob_i[1] * prob_j[2]  # G=2, C=1
            p_au = prob_i[0] * prob_j[3] + prob_i[3] * prob_j[0]  # A=0, U=3
            p_gu = prob_i[2] * prob_j[3] + prob_i[3] * prob_j[2]  # G=2, U=3
            
            p_valid = p_gc + p_au + p_gu
            
            # Penalty is (1 - p_valid), we want to maximize valid pairs
            total_penalty += (1.0 - p_valid)
    
    # Normalize by batch size
    return total_penalty / batch_size


def calculate_nucleotide_distribution(sequences_one_hot):
    """
    Calculate the distribution of nucleotides in generated sequences.
    Useful for detecting mode collapse.
    
    Args:
        sequences_one_hot: Tensor of shape (batch_size, seq_len, 4)
    
    Returns:
        dict: Distribution of each nucleotide (A, C, G, U) as percentages
    """
    # Get argmax predictions
    predictions = torch.argmax(sequences_one_hot, dim=-1)  # (batch_size, seq_len)
    
    total_positions = predictions.numel()
    
    distribution = {}
    for idx, nuc in IDX_TO_NUC.items():
        count = (predictions == idx).sum().item()
        distribution[nuc] = (count / total_positions) * 100.0
    
    return distribution


def get_structure_mask(structure, seq_len):
    """
    Create masks for paired and unpaired positions.
    
    Args:
        structure: Dot-bracket string
        seq_len: Sequence length
    
    Returns:
        tuple: (paired_mask, unpaired_mask) as numpy arrays
    """
    paired_mask = np.zeros(seq_len, dtype=np.float32)
    unpaired_mask = np.zeros(seq_len, dtype=np.float32)
    
    pairs = parse_dot_bracket(structure)
    paired_positions = set()
    
    for i, j in pairs:
        if i < seq_len:
            paired_positions.add(i)
        if j < seq_len:
            paired_positions.add(j)
    
    for i in range(min(len(structure), seq_len)):
        if i in paired_positions:
            paired_mask[i] = 1.0
        else:
            unpaired_mask[i] = 1.0
    
    return paired_mask, unpaired_mask
