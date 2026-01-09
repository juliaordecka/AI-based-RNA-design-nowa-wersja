import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
resnet_generator_rna.py

Structure-conditioned ResNet-based generator for WGAN-GP training.
The generator takes random noise AND a secondary structure (in one-hot format)
and produces an RNA sequence that should satisfy the structural constraints.

The architecture:
1. Projects noise to sequence representation
2. Combines with structure encoding
3. Processes through residual blocks
4. Outputs nucleotide probabilities conditioned on structure
"""


class NoiseToRNAEmbedding(nn.Module):
    """Projects noise vector to sequence-like embedding."""

    def __init__(self, noise_dim, sequence_length, embedding_size):
        super(NoiseToRNAEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.linear = nn.Linear(noise_dim, sequence_length * embedding_size)

    def forward(self, noise):
        batch_size = noise.shape[0]
        h = self.linear(noise)
        return h.view(batch_size, self.sequence_length, self.embedding_size) * np.sqrt(self.embedding_size)


class StructureEncoder(nn.Module):
    """Encodes structure information for conditioning."""

    def __init__(self, sequence_length, structure_dim=3, embed_dim=64):
        super(StructureEncoder, self).__init__()
        self.sequence_length = sequence_length

        # Process structure with 1D convolutions
        self.conv_layers = nn.Sequential(
            nn.Conv1d(structure_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(embed_dim // 2, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Linear projection to match embedding dimension
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, structure):
        """
        Args:
            structure: Tensor of shape (batch_size, seq_len, 3) - one-hot structure
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Permute for conv1d: (batch, channels, length)
        x = structure.permute(0, 2, 1)
        x = self.conv_layers(x)
        # Permute back: (batch, length, channels)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with structure-aware processing."""

    def __init__(self, channels, kernel_size=5, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = F.gelu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return out + residual


class StructureAwareResidualBlock(nn.Module):
    """Residual block that incorporates structure information."""

    def __init__(self, channels, structure_channels, kernel_size=5, dropout=0.2):
        super(StructureAwareResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(channels)

        # Structure conditioning (FiLM-like: Feature-wise Linear Modulation)
        self.structure_gamma = nn.Linear(structure_channels, channels)
        self.structure_beta = nn.Linear(structure_channels, channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, structure_embed):
        """
        Args:
            x: Tensor of shape (batch, channels, seq_len)
            structure_embed: Tensor of shape (batch, seq_len, structure_channels)
        """
        residual = x

        # First conv
        out = self.bn1(x)
        out = F.gelu(out)
        out = self.conv1(out)

        # Apply structure conditioning (FiLM)
        # structure_embed: (batch, seq_len, struct_channels)
        gamma = self.structure_gamma(structure_embed)  # (batch, seq_len, channels)
        beta = self.structure_beta(structure_embed)  # (batch, seq_len, channels)

        # Permute for multiplication: out is (batch, channels, seq_len)
        gamma = gamma.permute(0, 2, 1)  # (batch, channels, seq_len)
        beta = beta.permute(0, 2, 1)  # (batch, channels, seq_len)

        out = gamma * out + beta

        # Second conv
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.conv2(out)
        out = self.dropout(out)

        return out + residual


class ResNetGenerator(nn.Module):
    """
    Structure-conditioned ResNet generator for RNA sequence generation.

    Takes noise + structure and generates RNA sequence that satisfies
    the structural constraints.
    """

    def __init__(self, latent_dim, sequence_length, embed_dim=256, structure_dim=3,
                 n_blocks=4, use_structure_aware_blocks=True):
        super().__init__()
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim

        # Noise embedding
        self.noise_embedding = NoiseToRNAEmbedding(latent_dim, sequence_length, embed_dim)

        # Structure encoder
        self.structure_encoder = StructureEncoder(sequence_length, structure_dim, embed_dim // 2)

        # Combine noise and structure
        self.combine_layer = nn.Linear(embed_dim + embed_dim // 2, embed_dim)

        # Residual blocks
        kernel_sizes = [3, 5, 7, 9]

        if use_structure_aware_blocks:
            self.res_blocks = nn.ModuleList([
                StructureAwareResidualBlock(embed_dim, embed_dim // 2, kernel_size=k)
                for k in kernel_sizes
            ])
            self.use_structure_aware = True
        else:
            self.res_blocks = nn.ModuleList([
                ResidualBlock(embed_dim, kernel_size=k)
                for k in kernel_sizes
            ])
            self.use_structure_aware = False

        # Output layer
        self.out = nn.Conv1d(embed_dim, 4, kernel_size=1)

        # Temperature for Gumbel-Softmax
        self.tau = 0.5

    def forward(self, noise, structure):
        """
        Generate RNA sequence conditioned on structure.

        Args:
            noise: Tensor of shape (batch_size, latent_dim)
            structure: Tensor of shape (batch_size, seq_len, 3) - one-hot encoded structure

        Returns:
            Tensor of shape (batch_size, seq_len, 4) - one-hot encoded sequence
        """
        # Encode noise
        noise_embed = self.noise_embedding(noise)  # (batch, seq_len, embed_dim)

        # Encode structure
        struct_embed = self.structure_encoder(structure)  # (batch, seq_len, embed_dim//2)

        # Combine noise and structure
        combined = torch.cat([noise_embed, struct_embed], dim=-1)  # (batch, seq_len, embed_dim + embed_dim//2)
        combined = self.combine_layer(combined)  # (batch, seq_len, embed_dim)

        # Reshape for conv1d: (batch, channels, seq_len)
        x = combined.permute(0, 2, 1)

        # Process through residual blocks
        if self.use_structure_aware:
            for res_block in self.res_blocks:
                x = res_block(x, struct_embed)
        else:
            for res_block in self.res_blocks:
                x = res_block(x)

        # Output layer
        out = self.out(x)  # (batch, 4, seq_len)

        # Reshape to (batch, seq_len, 4)
        out = out.permute(0, 2, 1)

        # Apply Gumbel-Softmax for differentiable discrete sampling
        probs = F.gumbel_softmax(out, tau=self.tau, hard=True)

        return probs

    def generate_soft(self, noise, structure):
        """
        Generate soft probabilities (useful for computing penalties).

        Returns:
            Tensor of shape (batch_size, seq_len, 4) - soft probabilities
        """
        # Encode noise
        noise_embed = self.noise_embedding(noise)

        # Encode structure
        struct_embed = self.structure_encoder(structure)

        # Combine
        combined = torch.cat([noise_embed, struct_embed], dim=-1)
        combined = self.combine_layer(combined)

        # Reshape for conv1d
        x = combined.permute(0, 2, 1)

        # Process through residual blocks
        if self.use_structure_aware:
            for res_block in self.res_blocks:
                x = res_block(x, struct_embed)
        else:
            for res_block in self.res_blocks:
                x = res_block(x)

        # Output layer
        out = self.out(x)
        out = out.permute(0, 2, 1)

        # Return soft probabilities
        return F.softmax(out, dim=-1)


# Test
if __name__ == "__main__":
    batch_size = 32
    seq_len = 100
    latent_dim = 256

    # Create random inputs
    noise = torch.randn(batch_size, latent_dim)
    structure = torch.zeros(batch_size, seq_len, 3)
    # Random structure (one-hot)
    struct_indices = torch.randint(0, 3, (batch_size, seq_len))
    structure.scatter_(2, struct_indices.unsqueeze(-1), 1)

    # Create generator
    generator = ResNetGenerator(latent_dim, seq_len)

    # Generate
    output = generator(noise, structure)
    print(f"Output shape: {output.shape}")  # Should be (32, 100, 4)

    # Test soft generation
    soft_output = generator.generate_soft(noise, structure)
    print(f"Soft output shape: {soft_output.shape}")
    print(f"Soft output sum per position (should be ~1): {soft_output[0, 0].sum()}")
