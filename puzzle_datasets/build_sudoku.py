import numpy as np
import csv
import json

from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

CHARSET = ".123456789"
GRID_SIZE = (9, 9) # standard suoku grid size


def convert_subset(source_repo: str, set_name: str, output_dir: Path, skip_rating: int):
    # Read CSV
    all_chars = set()
    inputs = []
    labels = []
    ratings = []
    
    csv_path = hf_hub_download(source_repo, f"{set_name}.csv", repo_type="dataset")

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for _, q, a, rating in reader:
            rating = int(rating)

            if rating > skip_rating:
                continue
            
            all_chars.update(q)
            all_chars.update(a)

            inputs.append(
                np.frombuffer(q.encode(), dtype=np.uint8).reshape(GRID_SIZE)
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(GRID_SIZE)
            )
            ratings.append(rating)

    assert all_chars == set(CHARSET)

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "ratings", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out, rating in zip(tqdm(inputs), labels, ratings):
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
    SKIP_RATING = 5

    convert_subset(
        source_repo="sapientinc/sudoku-extreme",
        set_name="train",
        output_dir=Path(f"data/sudoku-extreme_skip{SKIP_RATING}"),
        skip_rating=SKIP_RATING
    )
    convert_subset(
        source_repo="sapientinc/sudoku-extreme",
        set_name="test",
        output_dir=Path(f"data/sudoku-extreme_skip{SKIP_RATING}"),
        skip_rating=SKIP_RATING
    )