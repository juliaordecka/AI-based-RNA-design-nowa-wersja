import torch
import os
import csv
from datetime import datetime
from utils.noise_generator import generate_noise
from utils.structure_utils import (
    calculate_pair_accuracy_batch,
    calculate_pairing_penalty,
    calculate_nucleotide_distribution,
    one_hot_to_sequences
)

"""
train_wgan_gp.py

WGAN-GP training for structure-conditioned RNA sequence generation.

Features:
- Gradient penalty for WGAN-GP
- Pair accuracy metric
- Pairing penalty for incorrect base pairs
- Nucleotide distribution tracking (for mode collapse detection)
- Date-stamped model saving
"""


def log_metrics(epoch, batch_idx, num_batches, metrics, log_path):
    """Log training metrics to console and file."""

    print(
        f"[Epoch {epoch + 1}] [Batch {batch_idx + 1}/{num_batches}] "
        f"[D loss: {metrics['d_loss']:.4f}] [G loss: {metrics['g_loss']:.4f}] "
        f"[Real: {metrics['critic_real_mean']:.4f}] [Fake: {metrics['critic_fake_mean']:.4f}] "
        f"[W-dist: {metrics['wasserstein_distance']:.4f}] "
        f"[Pair Acc: {metrics['pair_accuracy']:.2f}%] "
        f"[Pair Penalty: {metrics['pair_penalty']:.4f}]"
    )

    # Print nucleotide distribution
    nuc_dist = metrics['nucleotide_distribution']
    print(
        f"    Nucleotide Distribution: "
        f"A={nuc_dist['A']:.1f}% C={nuc_dist['C']:.1f}% "
        f"G={nuc_dist['G']:.1f}% U={nuc_dist['U']:.1f}%"
    )

    if log_path:
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                batch_idx,
                metrics['d_loss'],
                metrics['g_loss'],
                metrics['critic_real_mean'],
                metrics['critic_fake_mean'],
                metrics['wasserstein_distance'],
                metrics['pair_accuracy'],
                metrics['pair_penalty'],
                nuc_dist['A'],
                nuc_dist['C'],
                nuc_dist['G'],
                nuc_dist['U']
            ])


def gradient_penalty(critic, real_samples, fake_samples, structures, device="cpu"):
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        critic: Critic model
        real_samples: Real RNA sequences (batch, seq_len, 4)
        fake_samples: Generated RNA sequences (batch, seq_len, 4)
        structures: Structure encodings (batch, seq_len, 3)
        device: torch device

    Returns:
        Gradient penalty tensor
    """
    batch_size, sequence_length, nucleotides = real_samples.shape

    # Random interpolation coefficient
    epsilon = torch.rand((batch_size, 1, 1), device=device)

    # Interpolate between real and fake
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)

    # Get critic scores for interpolated samples
    with torch.backends.cudnn.flags(enabled=False):
        mixed_scores = critic(interpolated, structures)

    # Compute gradients
    gradient = torch.autograd.grad(
        inputs=interpolated,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)

    return penalty


def get_model_filename(epoch, batch, base_dir):
    """Generate filename with date stamp."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generator_epoch_{epoch + 1}_batch_{batch}_{date_str}.pth"
    return os.path.join(base_dir, filename)


def train_wgan_gp(generator, critic, dataset, args, device):
    """
    Train WGAN-GP with structure conditioning.

    Args:
        generator: Structure-conditioned generator model
        critic: Structure-aware critic model
        dataset: FastDatasetRNA with sequence-structure pairs
        args: Training arguments
        device: torch device
    """
    generator.train()
    critic.train()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup logging
    log_path = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "batch", "d_loss", "g_loss",
                "critic_real_mean", "critic_fake_mean", "wasserstein_distance",
                "pair_accuracy", "pair_penalty",
                "nuc_A", "nuc_C", "nuc_G", "nuc_U"
            ])

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.999))

    total_batches = 0
    generator_loss = torch.tensor(0.0)  # Initialize for first log

    for epoch in range(args.epochs):
        for i, batch_data in enumerate(dataset.dataloader):
            total_batches += 1

            # Unpack batch
            real_rna = batch_data['sequence'].float().to(device)
            structures = batch_data['structure'].float().to(device)
            structure_strs = batch_data['structure_str']

            batch_size = real_rna.size(0)

            # =====================================
            # Train Critic
            # =====================================

            # Generate fake sequences
            z = generate_noise(args.latent_dim, batch_size).to(device)
            fake_rna = generator(z, structures)

            # Critic scores
            critic_real = critic(real_rna, structures)
            critic_fake = critic(fake_rna.detach(), structures)

            # Gradient penalty
            gp = gradient_penalty(critic, real_rna, fake_rna, structures, device)

            # Critic loss: want to maximize (real - fake), so minimize (fake - real)
            critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + (gp * args.lambda_gp)

            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_C.step()

            # =====================================
            # Train Generator (every n_critic steps)
            # =====================================

            if i % args.n_critic == 0:
                # Generate new fake sequences
                z = generate_noise(args.latent_dim, batch_size).to(device)
                fake_rna = generator(z, structures)

                # Get soft probabilities for penalty calculation
                fake_rna_soft = generator.generate_soft(z, structures)

                # Base generator loss: want to maximize critic score on fakes
                gen_base_loss = -torch.mean(critic(fake_rna, structures))

                # Pairing penalty
                pair_penalty = calculate_pairing_penalty(fake_rna_soft, structure_strs, device)

                # Total generator loss
                generator_loss = gen_base_loss + args.lambda_pair * pair_penalty

                generator.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

            # =====================================
            # Calculate Metrics
            # =====================================

            # Calculate pair accuracy
            with torch.no_grad():
                pair_accuracy = calculate_pair_accuracy_batch(fake_rna, structure_strs, device)
                nuc_distribution = calculate_nucleotide_distribution(fake_rna)
                pair_penalty_value = calculate_pairing_penalty(fake_rna, structure_strs, device).item()

            # Compile metrics
            metrics = {
                'd_loss': critic_loss.item(),
                'g_loss': generator_loss.item(),
                'critic_real_mean': torch.mean(critic_real).item(),
                'critic_fake_mean': torch.mean(critic_fake).item(),
                'wasserstein_distance': torch.mean(critic_real).item() - torch.mean(critic_fake).item(),
                'pair_accuracy': pair_accuracy,
                'pair_penalty': pair_penalty_value,
                'nucleotide_distribution': nuc_distribution
            }

            # Log metrics
            log_metrics(epoch, i, len(dataset.dataloader), metrics, log_path)

            # Save model periodically
            if total_batches % args.save_interval == 0:
                model_path = get_model_filename(epoch, total_batches, args.save_dir)
                torch.save(generator.state_dict(), model_path)
                print(f"    Saved model to {model_path}")

    # Save final model
    final_path = get_model_filename(args.epochs - 1, total_batches, args.save_dir)
    torch.save(generator.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


def train_wgan_gp_simple(generator, critic, dataset, args, device):
    """
    Simplified training without structure conditioning.
    For testing/comparison purposes.
    """
    generator.train()
    critic.train()

    os.makedirs(args.save_dir, exist_ok=True)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_C = torch.optim.Adam(critic.parameters(), lr=args.lr_c, betas=(0.5, 0.999))

    total_batches = 0

    for epoch in range(args.epochs):
        for i, batch_data in enumerate(dataset.dataloader):
            total_batches += 1

            real_rna = batch_data['sequence'].float().to(device)
            structures = batch_data['structure'].float().to(device)
            structure_strs = batch_data['structure_str']
            batch_size = real_rna.size(0)

            # Train Critic
            z = generate_noise(args.latent_dim, batch_size).to(device)
            fake_rna = generator(z, structures)

            critic_real = critic(real_rna, structures)
            critic_fake = critic(fake_rna.detach(), structures)

            gp = gradient_penalty(critic, real_rna, fake_rna, structures, device)
            critic_loss = torch.mean(critic_fake) - torch.mean(critic_real) + (gp * args.lambda_gp)

            critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            optimizer_C.step()

            # Train Generator
            if i % args.n_critic == 0:
                z = generate_noise(args.latent_dim, batch_size).to(device)
                fake_rna = generator(z, structures)
                generator_loss = -torch.mean(critic(fake_rna, structures))

                generator.zero_grad()
                generator_loss.backward()
                optimizer_G.step()

            if total_batches % 10 == 0:
                pair_acc = calculate_pair_accuracy_batch(fake_rna, structure_strs, device)
                nuc_dist = calculate_nucleotide_distribution(fake_rna)
                print(f"[Epoch {epoch + 1}][Batch {i + 1}] D_loss: {critic_loss.item():.4f} | "
                      f"G_loss: {generator_loss.item():.4f} | Pair Acc: {pair_acc:.2f}% | "
                      f"A:{nuc_dist['A']:.1f}% C:{nuc_dist['C']:.1f}% G:{nuc_dist['G']:.1f}% U:{nuc_dist['U']:.1f}%")

            if total_batches % args.save_interval == 0:
                model_path = get_model_filename(epoch, total_batches, args.save_dir)
                torch.save(generator.state_dict(), model_path)
