import numpy as np
import csv
import json

from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

CHARSET = "# SGo"
GRID_SIZE = (30, 30)
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""
    
    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)       # horizontal flip
    elif tid == 5:
        return np.flipud(arr)       # vertical flip
    elif tid == 6:
        return arr.T                # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr
    
    
def inverse_dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    return dihedral_transform(arr, DIHEDRAL_INVERSE[tid])


def convert_subset(source_repo: str, set_name: str, output_dir: Path):
    # Read CSV
    all_chars = set()
    inputs = []
    labels = []
    
    csv_path = hf_hub_download(source_repo, f"{set_name}.csv", repo_type="dataset")

    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for _, q, a, _ in reader:
            all_chars.update(q)
            all_chars.update(a)

            inputs.append(
                np.frombuffer(q.encode(), dtype=np.uint8).reshape(GRID_SIZE)
            )
            labels.append(
                np.frombuffer(a.encode(), dtype=np.uint8).reshape(GRID_SIZE)
            )

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        for aug_idx in range(len(DIHEDRAL_INVERSE) if set_name == "train" else 1): 
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
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
    convert_subset(
        source_repo="sapientinc/maze-30x30-hard-1k",
        set_name="train",
        output_dir=Path("data/maze-30x30-hard-1k"),
    )
    convert_subset(
        source_repo="sapientinc/maze-30x30-hard-1k",
        set_name="test",
        output_dir=Path("data/maze-30x30-hard-1k"),
    )