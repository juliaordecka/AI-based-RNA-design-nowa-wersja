import argparse
from datetime import datetime
from utils.init_device import init_cuda
from loaders.fasta_data_loader import FastDatasetRNA
from models.resnet_generator_rna import ResNetGenerator
from models.critic import StructureCritic, PairAwareCritic
from utils.init_weights import initialize_weights
from train_wgan_gp import train_wgan_gp

"""
run_wgan.py

Main script for training structure-conditioned WGAN-GP for RNA sequence generation.

The model learns to generate RNA sequences that satisfy given secondary structure
constraints. Allowed base pairs are: G-C, A-U, and wobble G-U pairs.

Example usage:
    python run_wgan.py --data data/bp_seq_fixed_train_more_than_50_filtered_padded.fa

    python run_wgan.py --data data/training_data.fa --epochs 50 --batch_size 64 --lambda_pair 1.0
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train WGAN-GP for structure-conditioned RNA sequence generation"
    )

    # Data arguments
    parser.add_argument(
        "--data", type=str,
        default="data/bp_seq_fixed_train_more_than_50_filtered_padded.fa",
        help="Path to the RNA sequence data file (with structures)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="Sequence length. If not set, uses 98th percentile of data"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument("--n_critic", type=int, default=5, help="Critic updates per generator update")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty coefficient")
    parser.add_argument("--lambda_pair", type=float, default=1.0, help="Pairing penalty coefficient")

    # Learning rates
    parser.add_argument("--lr_g", type=float, default=0.0002, help="Generator learning rate")
    parser.add_argument("--lr_c", type=float, default=0.0001, help="Critic learning rate")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for critic")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension for generator")
    parser.add_argument(
        "--critic_type", type=str, default="structure",
        choices=["structure", "pair_aware"],
        help="Type of critic to use"
    )

    # Saving arguments
    parser.add_argument(
        "--save_dir", type=str, default="saved_models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="Directory for training logs"
    )
    parser.add_argument(
        "--save_interval", type=int, default=100,
        help="Save model every N batches"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = init_cuda()

    print("=" * 60)
    print("Structure-Conditioned WGAN-GP for RNA Sequence Generation")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = FastDatasetRNA(
        args.data,
        batch_size=args.batch_size,
        sequence_length=args.seq_len
    )
    args.seq_len = dataset.sequence_length

    print(f"  Dataset path: {args.data}")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Create models
    print("Creating models...")

    # Generator
    generator = ResNetGenerator(
        latent_dim=args.latent_dim,
        sequence_length=args.seq_len,
        embed_dim=args.embed_dim,
        structure_dim=3,
        use_structure_aware_blocks=True
    ).to(device)
    initialize_weights(generator)

    # Critic
    if args.critic_type == "pair_aware":
        critic = PairAwareCritic(
            sequence_length=args.seq_len,
            hidden_size=args.hidden_size
        ).to(device)
        print("  Using PairAwareCritic")
    else:
        critic = StructureCritic(
            sequence_length=args.seq_len,
            hidden_size=args.hidden_size
        ).to(device)
        print("  Using StructureCritic")

    # Print model info
    gen_params = sum(p.numel() for p in generator.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Generator parameters: {gen_params:,}")
    print(f"  Critic parameters: {critic_params:,}")
    print()

    # Print training config
    print("Training configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Latent dimension: {args.latent_dim}")
    print(f"  Critic updates per generator update: {args.n_critic}")
    print(f"  Gradient penalty (lambda_gp): {args.lambda_gp}")
    print(f"  Pairing penalty (lambda_pair): {args.lambda_pair}")
    print(f"  Generator LR: {args.lr_g}")
    print(f"  Critic LR: {args.lr_c}")
    print(f"  Save interval: every {args.save_interval} batches")
    print()

    # Train
    print("Starting training...")
    print("-" * 60)

    train_wgan_gp(generator, critic, dataset, args, device)

    print("-" * 60)
    print(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
