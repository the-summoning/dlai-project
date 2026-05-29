import numpy as np
import json
import pandas as pd

from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

CHARSET = ".123456789"

SIZE_9 = (9, 9)
SIZE_6 = (6, 6)
SIZE_4 = (4, 4)


def convert_subset(
    parquet_path: Path,
    output_dir: Path
):
    df = pd.read_parquet(parquet_path)

    all_chars = set()
    samples = []

    for _, row in df.iterrows():      
        if (row.rows, row.columns) == SIZE_6:
            size = 6
        elif (row.rows, row.columns) == SIZE_4:
            size = 4
        elif (row.rows, row.columns) == SIZE_9:
            size = 9
        else: # ??
            continue
        
        q = row["initial_board"]
        a = row["solution"]
    
        all_chars.update(q)
        all_chars.update(a)

        samples.append((q, a, size))


    assert all_chars == set(CHARSET)

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
        # pad to 9x9
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

    save_dir = output_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    for k, v in results.items():
        np.save(save_dir / f"{k}.npy", v)


if __name__ == "__main__":
    OUTPUT_DIR = Path("data/sudoku-bench")

    convert_subset(
        parquet_path=Path("/home/davide/Scrivania/dl project/data/sudoku-bench/challenge_100.parquet"),
        output_dir=OUTPUT_DIR / "challenge_100"
    )

    convert_subset(
        parquet_path=Path("/home/davide/Scrivania/dl project/data/sudoku-bench/ctc.parquet"),
        output_dir=OUTPUT_DIR / "ctc"
    )

    convert_subset(
        parquet_path=Path("/home/davide/Scrivania/dl project/data/sudoku-bench/nikoli_100.parquet"),
        output_dir=OUTPUT_DIR / "nikoli_100"
    )