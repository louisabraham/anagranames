# Prepare data from FR.csv for training
from collections import Counter, defaultdict
from unidecode import unidecode
import numpy as np
from tqdm import tqdm

def normalize_name(name):
    """Normalize name using unidecode and removing special chars."""
    return unidecode(name).replace(" ", "").replace("-", "").replace("'", "").upper()

def load_french_names(csv_path):
    """Load and process French names from CSV.

    Returns:
        edge_f: np.array of first name indices
        edge_l: np.array of last name indices
        edge_x: np.array of counts for each pair
        first_vocab: dict mapping first name -> index
        last_vocab: dict mapping last name -> index
        first_names: list of first names (index -> name)
        last_names: list of last names (index -> name)
    """
    print("Loading names from CSV...")

    # Count total lines first for progress bar
    with open(csv_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    # Count pairs
    pair_counts = Counter()

    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Reading CSV", ncols=100):
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue

            firstname = normalize_name(parts[0])
            lastname = normalize_name(parts[1])

            # Skip empty names
            if not firstname or not lastname:
                continue

            pair_counts[(firstname, lastname)] += 1

    print(f"Unique pairs: {len(pair_counts):,}")

    # Build vocabularies
    print("Building vocabularies...")
    first_set = set()
    last_set = set()
    for (f, l) in pair_counts:
        first_set.add(f)
        last_set.add(l)

    first_names = sorted(first_set)
    last_names = sorted(last_set)

    first_vocab = {name: idx for idx, name in enumerate(first_names)}
    last_vocab = {name: idx for idx, name in enumerate(last_names)}

    print(f"Unique first names: {len(first_names):,}")
    print(f"Unique last names: {len(last_names):,}")

    # Create edge lists
    print("Creating edge lists...")
    nnz = len(pair_counts)
    edge_f = np.zeros(nnz, dtype=np.int32)
    edge_l = np.zeros(nnz, dtype=np.int32)
    edge_x = np.zeros(nnz, dtype=np.float32)

    for i, ((firstname, lastname), count) in enumerate(
        tqdm(pair_counts.items(), desc="Building edges", ncols=100)
    ):
        edge_f[i] = first_vocab[firstname]
        edge_l[i] = last_vocab[lastname]
        edge_x[i] = count

    print(f"Edge list size: {nnz:,}")
    print(f"Total count: {edge_x.sum():,.0f}")

    return edge_f, edge_l, edge_x, first_vocab, last_vocab, first_names, last_names


if __name__ == "__main__":
    csv_path = "name_dataset/data/FR.csv"

    edge_f, edge_l, edge_x, first_vocab, last_vocab, first_names, last_names = load_french_names(csv_path)

    # Save to files
    print("\nSaving data...")
    np.save("edge_f.npy", edge_f)
    np.save("edge_l.npy", edge_l)
    np.save("edge_x.npy", edge_x)

    # Save vocabularies as text files
    with open("first_names.txt", "w", encoding="utf-8") as f:
        for name in first_names:
            f.write(name + "\n")

    with open("last_names.txt", "w", encoding="utf-8") as f:
        for name in last_names:
            f.write(name + "\n")

    print("Data saved to edge_f.npy, edge_l.npy, edge_x.npy")
    print("Vocabularies saved to first_names.txt, last_names.txt")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"  n (first names): {len(first_names):,}")
    print(f"  m (last names): {len(last_names):,}")
    print(f"  nnz (edges): {len(edge_f):,}")
    print(f"  Total observations: {edge_x.sum():,.0f}")
    print(f"  Avg count per pair: {edge_x.mean():.2f}")
