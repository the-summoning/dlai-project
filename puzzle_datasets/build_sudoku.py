import numpy as np
import csv
import json

from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

CHARSET = ".123456789"
GRID_SIZE = (9, 9) # standard sudoku grid size


def convert_subset(
    source_repo: str,
    set_name: str,
    output_dir: Path,
    allowed_ratings: tuple[int, ...],
    max_size: int | None = None,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # Read CSV
    all_chars = set()
    buckets = defaultdict(list)

    csv_path = hf_hub_download(
        source_repo,
        f"{set_name}.csv",
        repo_type="dataset"
    )

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for _, q, a, rating in reader:
            rating = int(rating)

            if rating not in allowed_ratings:
                continue

            all_chars.update(q)
            all_chars.update(a)

            buckets[rating].append((q, a, rating))

    assert all_chars == set(CHARSET)

    # Print original distribution
    total_original = sum(len(v) for v in buckets.values())

    print(f"\n[{set_name}] Original distribution")
    print(f"Total samples: {total_original}")

    for rating in sorted(buckets):
        count = len(buckets[rating])
        pct = 100 * count / total_original

        print(f"Rating {rating}: {count} samples ({pct:.2f}%)")

    # Stratified sampling
    samples = []

    if max_size is None:
        for items in buckets.values():
            samples.extend(items)

    else:
        for rating, items in buckets.items():
            proportion = len(items) / total_original
            n_keep = max(1, int(proportion * max_size))
            n_keep = min(n_keep, len(items))

            indices = rng.choice(
                len(items),
                size=n_keep,
                replace=False
            )

            for idx in indices:
                samples.append(items[idx])

    rng.shuffle(samples)

    # Print final distribution
    rating_counts = Counter(r for _, _, r in samples)

    print(f"\n[{set_name}] Final sampled distribution")
    print(f"Total samples: {len(samples)}")

    for rating in sorted(rating_counts):
        count = rating_counts[rating]
        pct = 100 * count / len(samples)

        print(f"Rating {rating}: {count} samples ({pct:.2f}%)")

    # Generate dataset
    results = {k: [] for k in [
        "inputs",
        "labels",
        "ratings",
        "puzzle_indices",
        "group_indices"
    ]}

    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for q, a, rating in tqdm(samples):
        inp = np.frombuffer(q.encode(), dtype=np.uint8).reshape(GRID_SIZE)
        out = np.frombuffer(a.encode(), dtype=np.uint8).reshape(GRID_SIZE)

        results["inputs"].append(inp)
        results["labels"].append(out)
        results["ratings"].append(rating)

        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)

    assert len(all_chars - set(CHARSET)) == 0

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    results = {
        "inputs": np.vstack([char2id[s.reshape(-1)] for s in results["inputs"]]),
        "labels": np.vstack([char2id[s.reshape(-1)] for s in results["labels"]]),
        "ratings": np.array(results["ratings"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
    }

    metadata = {
        "seq_len": int(np.prod(GRID_SIZE)),
        "vocab_size": len(CHARSET) + 1,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": 1,
        "total_puzzles": len(results["group_indices"]) - 1,
        "sets": ["all"],
    }

    save_dir = output_dir / set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    for k, v in results.items():
        np.save(save_dir / f"{k}.npy", v)


if __name__ == "__main__":
    RATINGS = (0, 1, 2)
    TRAIN_MAX_SIZE = 1000
    TEST_MAX_SIZE = 100

    OUTPUT_DIR = Path(f"data/sudoku-extreme_{TRAIN_MAX_SIZE}_below_rating_{max(RATINGS)}")

    convert_subset(
        source_repo="sapientinc/sudoku-extreme",
        set_name="train",
        output_dir=OUTPUT_DIR,
        allowed_ratings=RATINGS,
        max_size=TRAIN_MAX_SIZE
    )

    convert_subset(
        source_repo="sapientinc/sudoku-extreme",
        set_name="test",
        output_dir=OUTPUT_DIR,
        allowed_ratings=RATINGS,
        max_size=TEST_MAX_SIZE
    )