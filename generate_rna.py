import os
import numpy as np
import torch
import argparse
from datetime import datetime
from utils.noise_generator import generate_noise
from utils.structure_utils import (
    calculate_pair_accuracy,
    parse_dot_bracket,
    is_valid_pair
)
from models.resnet_generator_rna import ResNetGenerator
from loaders.fasta_data_loader import TestDatasetRNA, one_hot_structure

"""
generate_rna.py

Generate RNA sequences from secondary structure inputs using a trained generator.

The script reads a test file containing structures in dot-bracket notation
and generates RNA sequences that should satisfy the structural constraints.

Allowed pairs: G-C, A-U, G-U (wobble)

Example usage:
    python generate_rna.py --model_path saved_models/generator_epoch_15.pth --test_file data/test.fa
"""


def decode_sequence(one_hot_sequence):
    """Convert one-hot encoded sequence to string."""
    nucleotides = ['A', 'C', 'G', 'U']
    decoded_seq = []

    for one_hot in one_hot_sequence:
        if isinstance(one_hot, torch.Tensor):
            one_hot = one_hot.numpy()
        max_index = np.argmax(one_hot)
        if np.sum(one_hot) == 1:
            decoded_seq.append(nucleotides[max_index])
        else:
            decoded_seq.append('N')

    return "".join(decoded_seq)


def analyze_sequence(sequence, structure):
    """
    Analyze generated sequence against structure.

    Returns dict with:
    - pair_accuracy: percentage of correct pairs
    - valid_pairs: list of valid pairs
    - invalid_pairs: list of invalid pairs
    - nucleotide_counts: count of each nucleotide
    """
    pairs = parse_dot_bracket(structure)
    valid_pairs = []
    invalid_pairs = []

    for i, j in pairs:
        if i < len(sequence) and j < len(sequence):
            nuc1 = sequence[i]
            nuc2 = sequence[j]
            if is_valid_pair(nuc1, nuc2):
                valid_pairs.append((i, j, nuc1, nuc2))
            else:
                invalid_pairs.append((i, j, nuc1, nuc2))

    # Count nucleotides
    nuc_counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0, 'N': 0}
    for nuc in sequence:
        if nuc in nuc_counts:
            nuc_counts[nuc] += 1
        else:
            nuc_counts['N'] += 1

    pair_accuracy = (len(valid_pairs) / len(pairs) * 100) if pairs else 100.0

    return {
        'pair_accuracy': pair_accuracy,
        'valid_pairs': valid_pairs,
        'invalid_pairs': invalid_pairs,
        'nucleotide_counts': nuc_counts,
        'total_pairs': len(pairs)
    }


def generate_from_structure(generator, structure_str, latent_dim, num_samples=1, device='cpu'):
    """
    Generate sequences from a single structure.

    Args:
        generator: Trained generator model
        structure_str: Dot-bracket structure string
        latent_dim: Latent dimension for noise
        num_samples: Number of sequences to generate
        device: torch device

    Returns:
        List of generated sequences
    """
    generator.eval()
    seq_len = len(structure_str)

    # Encode structure
    struct_one_hot = torch.tensor(
        one_hot_structure(structure_str),
        dtype=torch.float32
    ).unsqueeze(0).repeat(num_samples, 1, 1).to(device)

    sequences = []

    with torch.no_grad():
        z = generate_noise(latent_dim, num_samples).to(device)
        generated = generator(z, struct_one_hot).cpu().numpy()

        for i in range(num_samples):
            seq = decode_sequence(generated[i])
            sequences.append(seq)

    return sequences


def generate_from_file(generator, test_file, latent_dim, num_samples_per_structure=1,
                       device='cpu', sequence_length=None):
    """
    Generate sequences from all structures in a test file.

    Args:
        generator: Trained generator model
        test_file: Path to test file with structures
        latent_dim: Latent dimension for noise
        num_samples_per_structure: How many sequences to generate per structure
        device: torch device
        sequence_length: Expected sequence length

    Returns:
        List of (name, structure, generated_sequences, analyses) tuples
    """
    # Load test data
    test_dataset = TestDatasetRNA(test_file, sequence_length=sequence_length)

    results = []

    generator.eval()

    for item in test_dataset.data:
        name = item['name']
        target_seq = item['target_sequence']
        structure = item['structure']

        # Adjust structure length if needed
        if sequence_length is not None:
            if len(structure) < sequence_length:
                structure = structure + '.' * (sequence_length - len(structure))
            structure = structure[:sequence_length]

        # Generate sequences
        generated_seqs = generate_from_structure(
            generator, structure, latent_dim, num_samples_per_structure, device
        )

        # Analyze each generated sequence
        analyses = []
        for seq in generated_seqs:
            analysis = analyze_sequence(seq, structure)
            analyses.append(analysis)

        results.append({
            'name': name,
            'target_sequence': target_seq,
            'structure': structure,
            'generated_sequences': generated_seqs,
            'analyses': analyses
        })

    return results


def save_results(results, output_dir, include_analysis=True):
    """
    Save generated sequences to FASTA file and optionally analysis report.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save FASTA
    fasta_path = os.path.join(output_dir, "generated_sequences.fasta")
    with open(fasta_path, 'w') as f:
        for result in results:
            name = result['name']
            structure = result['structure']

            for i, seq in enumerate(result['generated_sequences']):
                header = f">{name}_generated_{i + 1}"
                f.write(f"{header}\n{seq}\n")

    print(f"Saved generated sequences to {fasta_path}")

    # Save analysis report
    if include_analysis:
        report_path = os.path.join(output_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("RNA Generation Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            total_accuracy = 0
            total_samples = 0

            for result in results:
                f.write(f"Structure: {result['name']}\n")
                f.write(f"Target sequence:\n  {result['target_sequence'][:80]}...\n")
                f.write(f"Structure:\n  {result['structure'][:80]}...\n")
                f.write(f"Number of pairs: {result['analyses'][0]['total_pairs']}\n\n")

                for i, (seq, analysis) in enumerate(zip(result['generated_sequences'], result['analyses'])):
                    f.write(f"  Generated {i + 1}: {seq[:60]}...\n")
                    f.write(f"    Pair accuracy: {analysis['pair_accuracy']:.2f}%\n")
                    f.write(f"    Valid pairs: {len(analysis['valid_pairs'])}\n")
                    f.write(f"    Invalid pairs: {len(analysis['invalid_pairs'])}\n")

                    # Show nucleotide distribution
                    nuc = analysis['nucleotide_counts']
                    total_nuc = sum(nuc.values())
                    f.write(f"    Nucleotide distribution: ")
                    f.write(f"A={nuc['A'] / total_nuc * 100:.1f}% ")
                    f.write(f"C={nuc['C'] / total_nuc * 100:.1f}% ")
                    f.write(f"G={nuc['G'] / total_nuc * 100:.1f}% ")
                    f.write(f"U={nuc['U'] / total_nuc * 100:.1f}%\n")

                    # Show some invalid pairs
                    if analysis['invalid_pairs']:
                        f.write(f"    Sample invalid pairs: ")
                        for ip in analysis['invalid_pairs'][:3]:
                            f.write(f"({ip[0]},{ip[1]}):{ip[2]}-{ip[3]} ")
                        f.write("\n")

                    total_accuracy += analysis['pair_accuracy']
                    total_samples += 1
                    f.write("\n")

                f.write("-" * 60 + "\n\n")

            avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0
            f.write(f"\nOverall Statistics:\n")
            f.write(f"  Total structures: {len(results)}\n")
            f.write(f"  Total sequences generated: {total_samples}\n")
            f.write(f"  Average pair accuracy: {avg_accuracy:.2f}%\n")

        print(f"Saved analysis report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RNA sequences from secondary structures"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained generator model"
    )
    parser.add_argument(
        "--test_file", type=str, required=True,
        help="Path to test file with structures"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="Number of sequences to generate per structure"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=97,
        help="Sequence length (must match training)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_output",
        help="Directory to save generated sequences"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=256,
        help="Latent dimension (must match training)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256,
        help="Embedding dimension (must match training)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RNA Sequence Generation from Secondary Structure")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Load model
    print("Loading generator model...")
    generator = ResNetGenerator(
        latent_dim=args.latent_dim,
        sequence_length=args.sequence_length,
        embed_dim=args.embed_dim
    ).to(args.device)

    generator.load_state_dict(
        torch.load(args.model_path, map_location=args.device, weights_only=True)
    )
    generator.eval()
    print("Model loaded successfully.")
    print()

    # Generate sequences
    print("Generating sequences...")
    results = generate_from_file(
        generator=generator,
        test_file=args.test_file,
        latent_dim=args.latent_dim,
        num_samples_per_structure=args.num_samples,
        device=args.device,
        sequence_length=args.sequence_length
    )

    # Save results
    print()
    print("Saving results...")
    save_results(results, args.output_dir, include_analysis=True)

    # Print summary
    print()
    print("=" * 60)
    print("Summary:")
    total_acc = 0
    count = 0
    for result in results:
        for analysis in result['analyses']:
            total_acc += analysis['pair_accuracy']
            count += 1

    print(f"  Structures processed: {len(results)}")
    print(f"  Total sequences generated: {count}")
    print(f"  Average pair accuracy: {total_acc / count:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
