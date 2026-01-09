import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

"""
critic.py

Structure-aware Critic model for WGAN-GP training.

The critic evaluates:
1. Whether the sequence looks like a real RNA sequence
2. Whether the sequence is compatible with the given structure

The architecture combines CNN for local feature extraction with
BiLSTM for capturing sequential dependencies, conditioned on structure.
"""


class StructureCritic(nn.Module):
    """
    Structure-conditioned critic for WGAN-GP.

    Takes both sequence and structure as input and outputs
    a score indicating "realness" and structure compatibility.
    """

    def __init__(self, sequence_length, hidden_size=128, num_layers=1, dropout=0.5):
        super(StructureCritic, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        # Input: 4 (nucleotides) + 3 (structure) = 7 channels
        input_size = 7

        # Convolutional layers for local feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        # BiLSTM for sequential dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if num_layers == 1 else dropout
        )

        # Output layers with spectral normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )

    def forward(self, sequence, structure):
        """
        Evaluate sequence-structure pair.

        Args:
            sequence: Tensor of shape (batch_size, seq_len, 4) - one-hot nucleotides
            structure: Tensor of shape (batch_size, seq_len, 3) - one-hot structure

        Returns:
            Tensor of shape (batch_size, 1) - critic score
        """
        # Concatenate sequence and structure
        x = torch.cat([sequence, structure], dim=-1)  # (batch, seq_len, 7)

        # Permute for conv1d: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Convolutional processing
        x = self.conv_layers(x)

        # Permute back for LSTM: (batch, seq_len, hidden)
        x = x.permute(0, 2, 1)

        # LSTM processing
        x, _ = self.lstm(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Output
        x = self.fc_layers(x)

        return x


class PairAwareCritic(nn.Module):
    """
    Enhanced critic that explicitly considers base pairing.

    Uses attention mechanism to focus on paired positions.
    """

    def __init__(self, sequence_length, hidden_size=128, num_layers=1, dropout=0.5, num_heads=4):
        super(PairAwareCritic, self).__init__()

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size

        # Input processing
        input_size = 7  # 4 (seq) + 3 (struct)

        # Initial projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        # Self-attention for pair interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if num_layers == 1 else dropout
        )

        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size, 1))
        )

    def forward(self, sequence, structure):
        """
        Evaluate sequence-structure pair with attention to pairings.

        Args:
            sequence: Tensor of shape (batch_size, seq_len, 4)
            structure: Tensor of shape (batch_size, seq_len, 3)

        Returns:
            Tensor of shape (batch_size, 1)
        """
        # Concatenate and project
        x = torch.cat([sequence, structure], dim=-1)  # (batch, seq_len, 7)
        x = self.input_proj(x)  # (batch, seq_len, hidden)

        # Convolutional processing
        x = x.permute(0, 2, 1)  # (batch, hidden, seq_len)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden)

        # Self-attention (captures pair relationships)
        attn_out, _ = self.attention(x, x, x)
        x = self.attention_norm(x + attn_out)

        # LSTM processing
        x, _ = self.lstm(x)

        # Global average pooling
        x = torch.mean(x, dim=1)

        # Output
        x = self.fc_layers(x)

        return x


# Keep the original Critic for compatibility
class Critic(nn.Module):
    """Original critic without structure conditioning (for reference)."""

    def __init__(self, sequence_length, hidden_size=128, num_layers=1, dropout=0.5):
        super(Critic, self).__init__()

        self.sequence_length = sequence_length
        input_size = 4  # One-hot nucleotides only

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size // 2, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.3),
            nn.Dropout(dropout),
            spectral_norm(nn.Linear(hidden_size // 2, 1))
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = torch.mean(x, dim=1)
        x = self.fc_layers(x)
        return x


# Test
if __name__ == "__main__":
    batch_size = 32
    seq_len = 100

    # Create random inputs
    sequence = torch.zeros(batch_size, seq_len, 4)
    seq_indices = torch.randint(0, 4, (batch_size, seq_len))
    sequence.scatter_(2, seq_indices.unsqueeze(-1), 1)

    structure = torch.zeros(batch_size, seq_len, 3)
    struct_indices = torch.randint(0, 3, (batch_size, seq_len))
    structure.scatter_(2, struct_indices.unsqueeze(-1), 1)

    # Test StructureCritic
    critic = StructureCritic(seq_len)
    output = critic(sequence, structure)
    print(f"StructureCritic output shape: {output.shape}")  # Should be (32, 1)

    # Test PairAwareCritic
    pair_critic = PairAwareCritic(seq_len)
    output2 = pair_critic(sequence, structure)
    print(f"PairAwareCritic output shape: {output2.shape}")  # Should be (32, 1)
