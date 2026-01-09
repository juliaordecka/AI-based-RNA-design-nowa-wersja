import torch
from torch.utils.data import Dataset, DataLoader
import random
import math
import numpy as np
from Bio import SeqIO
import re


# Convert sequence to RNA format (ensure all T are converted to U)
def convert_to_rna(sequence):
    return sequence.replace('T', 'U')


"""
IUPAC nucleotide code converter for RNA sequences.
Based on: https://www.bioinformatics.org/sms/iupac.html
"""


def iupac_converter(nucleotide):
    nucleotide_map = {
        'R': ['A', 'G'],
        'Y': ['C', 'U'],
        'S': ['G', 'C'],
        'W': ['A', 'U'],
        'K': ['G', 'U'],
        'M': ['A', 'C'],
        'B': ['C', 'G', 'U'],
        'D': ['A', 'G', 'U'],
        'H': ['A', 'C', 'U'],
        'V': ['A', 'C', 'G'],
        'N': ['A', 'C', 'G', 'U']
    }

    if nucleotide in nucleotide_map:
        elements = nucleotide_map[nucleotide]
        return random.choice(elements)
    return nucleotide


# One-hot encoding for sequences
def one_hot_encoding(sequence):
    nuc = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1],
    }
    return [nuc.get(nucleotide, [0.25, 0.25, 0.25, 0.25]) for nucleotide in sequence]


# One-hot encoding for structures
def one_hot_structure(structure):

    struct_map = {
        '.': [1, 0, 0],
        '(': [0, 1, 0],
        ')': [0, 0, 1],
        'N': [1, 0, 0],  # Treat padding as unpaired
    }
    return [struct_map.get(char, [1, 0, 0]) for char in structure]


class FastDatasetRNA(Dataset):


    def __init__(self, file_path, sequence_length=None, batch_size=32, shuffle=True, drop_last=True):
        self.data = []
        self.sequence_length = sequence_length
        self.file_path = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._load_sequences()
        self.dataloader = DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            collate_fn=self.collate_fn
        )

    def _load_sequences(self):
        """Load sequences and their structures from file."""
        raw_data = []

        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        # Parse the file - detect format
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line.startswith('>'):
                header = line[1:]
                i += 1

                if i >= len(lines):
                    break

                sequence = lines[i].strip().upper()
                i += 1

                # Check if next line is structure or another header
                if i < len(lines):
                    next_line = lines[i].strip()

                    if next_line.startswith('>'):
                        # Check if it's a struct header
                        if 'struct' in next_line.lower():
                            i += 1
                            if i < len(lines):
                                structure = lines[i].strip()
                                i += 1
                            else:
                                structure = '.' * len(sequence)
                        else:
                            # Structure is on the line after sequence (training format)
                            # Re-read: the sequence line might have structure below it
                            structure = '.' * len(sequence)
                    elif self._is_structure(next_line):
                        # This is a structure line
                        structure = next_line
                        i += 1
                    else:
                        structure = '.' * len(sequence)
                else:
                    structure = '.' * len(sequence)

                raw_data.append((header, sequence, structure))
            else:
                i += 1

        # Determine sequence length from data if not specified
        if self.sequence_length is None:
            lengths = [len(seq) for _, seq, _ in raw_data]
            self.sequence_length = math.ceil(np.percentile(lengths, 98))

        # Process and pad sequences
        for header, sequence, structure in raw_data:
            # Truncate or pad sequence
            sequence = sequence[:self.sequence_length]
            structure = structure[:self.sequence_length]

            if len(sequence) < self.sequence_length:
                sequence += 'N' * (self.sequence_length - len(sequence))
            if len(structure) < self.sequence_length:
                structure += '.' * (self.sequence_length - len(structure))

            # Convert to RNA format
            sequence = convert_to_rna(sequence)
            sequence = ''.join(iupac_converter(nuc) for nuc in sequence)

            self.data.append((header, sequence, structure))

    def _is_structure(self, line):
        """Check if a line is a structure (contains only ., (, ), or N)"""
        valid_chars = set('.()N')
        return all(c in valid_chars for c in line) and len(line) > 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        header, sequence, structure = self.data[idx]
        seq_one_hot = one_hot_encoding(sequence)
        struct_one_hot = one_hot_structure(structure)

        return {
            'sequence': torch.tensor(seq_one_hot, dtype=torch.float32),
            'structure': torch.tensor(struct_one_hot, dtype=torch.float32),
            'structure_str': structure,
            'sequence_str': sequence,
            'header': header
        }

    def collate_fn(self, batch):
        """Custom collate function to handle string data."""
        sequences = torch.stack([item['sequence'] for item in batch])
        structures = torch.stack([item['structure'] for item in batch])
        structure_strs = [item['structure_str'] for item in batch]
        sequence_strs = [item['sequence_str'] for item in batch]
        headers = [item['header'] for item in batch]

        return {
            'sequence': sequences,
            'structure': structures,
            'structure_str': structure_strs,
            'sequence_str': sequence_strs,
            'header': headers
        }

    def __iter__(self):
        return iter(self.dataloader)

    def percentile_length(self, percentile):
        lengths = [len(seq) for _, seq, _ in self.data]
        return np.percentile(lengths, percentile)

    def get_structures(self):
        """Return all structure strings."""
        return [struct for _, _, struct in self.data]


class TestDatasetRNA(Dataset):


    def __init__(self, file_path, sequence_length=None):
        self.data = []
        self.file_path = file_path
        self.sequence_length = sequence_length
        self._load_structures()

    def _load_structures(self):
        """Load test structures from file."""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()

        entries = {}
        current_name = None
        current_content = None

        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    entries[current_name] = current_content
                current_name = line[1:]
                current_content = ''
            else:
                if current_content is not None:
                    current_content += line

        if current_name is not None:
            entries[current_name] = current_content

        # Match sequences with their structures
        processed = set()
        for name, content in entries.items():
            if name.startswith('struct_'):
                base_name = name[7:]  # Remove 'struct_' prefix
                if base_name in entries:
                    sequence = entries[base_name]
                    structure = content

                    # Determine length
                    seq_len = len(structure)
                    if self.sequence_length is not None:
                        seq_len = self.sequence_length
                        structure = structure[:seq_len]
                        if len(structure) < seq_len:
                            structure += '.' * (seq_len - len(structure))

                    self.data.append({
                        'name': base_name,
                        'target_sequence': sequence,
                        'structure': structure
                    })
                    processed.add(base_name)
                    processed.add(name)

        # Update sequence_length based on data
        if self.sequence_length is None and len(self.data) > 0:
            self.sequence_length = max(len(d['structure']) for d in self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        structure = item['structure']

        # Pad if necessary
        if self.sequence_length is not None:
            if len(structure) < self.sequence_length:
                structure = structure + '.' * (self.sequence_length - len(structure))
            structure = structure[:self.sequence_length]

        struct_one_hot = one_hot_structure(structure)

        return {
            'name': item['name'],
            'target_sequence': item['target_sequence'],
            'structure': torch.tensor(struct_one_hot, dtype=torch.float32),
            'structure_str': structure
        }
