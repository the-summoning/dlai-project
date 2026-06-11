import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

CHARSET = "0123456789"
GRID_SIZE = (9, 9)


def convert_subset(source_repo: str, set_name: str, output_dir: Path, n_samples: int, seed: int, max_diff: int):
    inputs = []
    labels = []

    ds = load_dataset(source_repo, split=set_name)
    ds = ds.filter(lambda x: int(x["difficulty"]) <= max_diff)
    ds = ds.shuffle(seed=seed)
    ds = ds.select(range(n_samples))

    for sample in ds:
        q = sample['puzzle'].strip() # type: ignore
        a = sample['solution'].strip() # type: ignore

        assert len(q) == np.prod(GRID_SIZE) == len(a)

        inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
        labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))


    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        results["inputs"].append(orig_inp)
        results["labels"].append(orig_out)
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
            
        results["group_indices"].append(puzzle_id)
        

    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = {
        "seq_len": int(np.prod(GRID_SIZE)),
        "vocab_size": 10 + 1,  # PAD + "0" ... "9"
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": 1,
        "total_puzzles": len(results["group_indices"]) - 1,
        "sets": ["all"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    for k, v in results.items():
        np.save(output_dir / f"{k}.npy", v)


if __name__ == "__main__":
    OUTPUT_DIR = Path(f"data/sudoku")

    convert_subset(
        source_repo="Ritvik19/Sudoku-Dataset",
        set_name="train",
        output_dir=OUTPUT_DIR / "train",
        n_samples=10000,
        seed=42,
        max_diff=2
    )

    convert_subset(
        source_repo="Ritvik19/Sudoku-Dataset",
        set_name="validation",
        output_dir=OUTPUT_DIR / "val",
        n_samples=500,
        seed=42,
        max_diff=2
    )