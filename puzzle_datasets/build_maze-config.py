import torch
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path

from maze_dataset import MazeDataset, MazeDatasetConfig, LatticeMazeGenerators
from maze_dataset.maze import SolvedMaze
from maze_dataset.dataset.filters import MazeDatasetFilters

from torch.utils.data import random_split
from torch.utils.data import ConcatDataset, Dataset
from dihetral import dihedral_transform, DIHEDRAL_INVERSE


CHARSET = "# SGo"

def build_generator_dataset(
    name: str,
    n: int,
    size: int,
    generator,
    seed: int,
):
    cfg = MazeDatasetConfig(
        name=name,
        grid_n=size,
        n_mazes=n,
        maze_ctor=generator,
        seed=seed,
    )

    ds = MazeDataset.from_config(cfg)
    ds = MazeDatasetFilters.remove_duplicates_fast(ds)

    return ds


def create_datasets(
    n_train: int = 5000,
    val_pct: float = 0.10,
    n_test: int = 3000,
    grid_size: int = 6,
    seed: int = 42,
):
    assert 0 < val_pct < 1

    print("Creating datasets...")

    dfs_dataset = build_generator_dataset(
        name="dfs",
        n=n_train // 2,
        size=grid_size,
        generator=LatticeMazeGenerators.gen_dfs,
        seed=seed,
    )

    wilson_dataset = build_generator_dataset(
        name="wilson",
        n=n_train - (n_train // 2),
        size=grid_size,
        generator=LatticeMazeGenerators.gen_wilson,
        seed=seed,
    )


    n_dfs = len(dfs_dataset)
    n_dfs_train = int(n_dfs * (1 - val_pct))
    n_dfs_val = n_dfs - n_dfs_train

    dfs_train, dfs_val = random_split(
        dfs_dataset,
        [n_dfs_train, n_dfs_val],
        generator=torch.Generator().manual_seed(seed),
    )

    n_w = len(wilson_dataset)
    n_w_train = int(n_w * (1 - val_pct))
    n_w_val = n_w - n_w_train

    wilson_train, wilson_val = random_split(
        wilson_dataset,
        [n_w_train, n_w_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train = (dfs_train, wilson_train)
    val = (dfs_val, wilson_val)

    n_test_dfs = n_test // 2
    n_test_wilson = n_test - n_test_dfs

    test_dfs = build_generator_dataset(
        name="test_dfs",
        n=n_test_dfs,
        size=grid_size,
        generator=LatticeMazeGenerators.gen_dfs,
        seed=seed + 1,
    )

    test_wilson = build_generator_dataset(
        name="test_wilson",
        n=n_test_wilson,
        size=grid_size,
        generator=LatticeMazeGenerators.gen_wilson,
        seed=seed + 1,
    )

    test = (test_dfs, test_wilson)


    print(f"Grid:   {grid_size}x{grid_size}")
    print(f"DFS:    {len(dfs_dataset)}")
    print(f"WILSON: {len(wilson_dataset)}")
    print(f"Test DFS: {len(test_dfs)}")
    print(f"Test WILSON: {len(test_wilson)}")

    return train, val, test


def convert_subset(dataset: Dataset, output_dir: Path, size: int, augment: bool = False):
    grid_size_walls = (2*size + 1, 2*size + 1)
    grid_size = (2*size - 1, 2*size - 1)
    all_chars = set()
    inputs = []
    labels = []

    maze: SolvedMaze

    for maze in dataset: # type: ignore
        a = maze.as_ascii().replace('\n', '').strip().replace('X', 'o').replace('E', 'G')
        q = a.replace('o', ' ')

        all_chars.update(q)
        all_chars.update(a)

        inputs.append(
            np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size_walls)[1:-1, 1:-1] # I'm removing external walls
        )
        labels.append(
            np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size_walls)[1:-1, 1:-1] # I'm removing external walls
        )

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(tqdm(inputs), labels):
        for aug_idx in range(len(DIHEDRAL_INVERSE) if augment else 1): 
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
        "seq_len": int(np.prod(grid_size)),
        "vocab_size": len(CHARSET) + 1,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": 1,
        "total_puzzles": len(results["group_indices"]) - 1,
        #"sets": ["all"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    for k, v in results.items():
        np.save(output_dir / f"{k}.npy", v)


if __name__ == "__main__":
    GRID_SIZES = [5, 10, 15]

    for grid_size in GRID_SIZES:
        output_dir = Path(f'data/maze{grid_size}x{grid_size}')

        train, val, test = create_datasets(
            n_train=625, # x8 = 5000
            val_pct=0.10,
            n_test=150,
            grid_size=grid_size,
            seed=42,
        )

        convert_subset(
            ConcatDataset(train),
            output_dir / "train",
            grid_size,
            True
        )

        convert_subset(
            ConcatDataset(train),
            output_dir / "val",
            grid_size,
            False
        )

        convert_subset(
            ConcatDataset(train),
            output_dir / "test",
            grid_size,
            False
        )