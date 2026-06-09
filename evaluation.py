import numpy as np
import argparse
import torch
import torch.nn as nn
import json
import math

from model.trm import TinyRecursiveModel
from torch.nn import functional as F
from model.layers import TransformerBlock
from pathlib import Path
from puzzle_datasets.puzzle_dataset import IGNORE_LABEL_ID


@torch.inference_mode()
def evaluate(
    model: TinyRecursiveModel,
    inputs_path: Path,
    labels_path: Path,
    batch_size: int,
    device: torch.device,
    ignore_label_id: int,
    max_recurrent_steps: int,
    W_slice: int | None = None,
    enable_tracking: bool = False
) -> dict:
    model.eval()

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    exact_puzzles = 0
    total_val_loss = 0.0

    z_trajectories = []
    y_trajectories = []
    halt_trajectories = []

    data_inputs = np.load(inputs_path, mmap_mode="r")
    data_labels = np.load(labels_path, mmap_mode="r")

    for start in range(0, len(data_inputs), batch_size):
        end = start + batch_size

        dataset_input = torch.tensor(data_inputs[start:end], dtype=torch.long).to(device)
        labels = torch.tensor(data_labels[start:end], dtype=torch.long)
        labels = labels.masked_fill(labels == ignore_label_id, IGNORE_LABEL_ID).to(device)
        
        # if probabilist_trm:
        #     pass
        #     _, _, logits, _ = model.probabilistic_predict(dataset_input, max_recurrent_steps, probabilistic_rollouts, 0.2)
        # else:
        #     _, _, logits, _ = model.predict(dataset_input, max_recurrent_steps)

        if enable_tracking:
            model.enable_tracking()

        logits = model.predict(dataset_input, max_recurrent_steps, W_slice)

        if enable_tracking:
            z_trajectories.append(np.stack(model.trajectory["z"], axis=1)) # type: ignore
            y_trajectories.append(np.stack(model.trajectory["y"], axis=1)) # type: ignore
            halt_trajectories.append(np.stack(model.trajectory["halt_logits"], axis=1)) # type: ignore

        final_preds = logits.argmax(-1)  # (B, seq)

        mask = labels != IGNORE_LABEL_ID
        lm_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_LABEL_ID,
            reduction="mean"
        )
        total_val_loss += lm_loss.item()

        correct_tokens += (mask & (final_preds == labels)).sum().item()
        total_tokens += mask.sum().item()

        exact_puzzles += ((final_preds == labels) | ~mask).all(dim=-1).sum().item()
        total_puzzles += labels.shape[0]

    model.disable_tracking()

    return {
        "val_loss": total_val_loss / max(total_puzzles, 1),
        "token_acc": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_puzzles  / max(total_puzzles, 1),
        "trajectories": {
            'z': np.concatenate(z_trajectories, axis=0),
            'y': np.concatenate(y_trajectories, axis=0),
            'halt_logits': np.concatenate(halt_trajectories, axis=0)
        } if enable_tracking else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyRecursiveModel on a PuzzleDataset")
    parser.add_argument("--inputs-path", type=str,
                        default="data/sudoku4x4/test/inputs.npy",
                        help="Path to validation inputs .npy file")
    parser.add_argument("--labels-path", type=str,
                        default="data/sudoku4x4/test/labels.npy",
                        help="Path to validation labels .npy file")
    parser.add_argument("--metadata-path", type=str, default=None,
                        help="Path to metadata.json file (defaults to inputs parent directory)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt",
                        help="Path to the checkpoint file to load")
    parser.add_argument("--results-path", type=str, default="./results/sudoku4x4",
                        help="Path where to save results")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for the PuzzleDataset")
    parser.add_argument("--max-recurrent-steps", type=int, default=12,
                        help="Maximum number of recurrent steps during evaluation")
    parser.add_argument("--T", type=int, default=3,
                        help="Number of high level iterations (T) (# times the model can re-process its own output)")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of low level iterations (n) (# times the model processes the input before producing an output)")
    parser.add_argument("--expansion", type=float, default=4.0,
                        help="Expansion factor for the hidden size in the Transformer layers")
    parser.add_argument("--registers", type=int, default=16,
                        help="Number of registers to be used in the model (including CLS token)")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads in the Transformer layers")
    # parser.add_argument("--probabilistic-trm", action="store_true",
    #                     help="Use probabilistic TRM prediction instead of deterministic prediction")
    # parser.add_argument("--probabilistic-rollouts", type=int, default=1,
    #                     help="Number of rollouts when using probabilistic TRM")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden size for TransformerBlock layers and model")
    parser.add_argument("--num-transformer-blocks", type=int, default=4,
                        help="Number of TransformerBlock layers in the network")
    parser.add_argument("--enable-matryoshka-from", type=int, default=None,
                        help="Enable Matryoshka evaluation from this size (inclusive) and then double to reach hidden size")
    parser.add_argument("--mlp-t", action="store_true",
                        help="Use MLP for temporal mixing instead of self-attention")
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metadata_path = Path(args.metadata_path) if args.metadata_path is not None else Path(args.inputs_path).parent / "metadata.json"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    vocab_size = metadata["vocab_size"]
    seq_len = metadata["seq_len"]
    num_registers = args.registers

    network = nn.Sequential(
        *[
            TransformerBlock(
                num_heads=args.num_heads,
                hidden_size=args.hidden_size,
                expansion=args.expansion,
                seq_len=seq_len,
                num_registers=num_registers,
                use_mlp_t=args.mlp_t
            )
            for _ in range(args.num_transformer_blocks)
        ]
    )

    model = TinyRecursiveModel(
        hidden_size=args.hidden_size,
        vocab_size=vocab_size,
        network=network,
        n=args.n,
        T=args.T,
        num_registers=num_registers,
    )

    checkpoint = torch.load(Path(args.checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["ema_model_state_dict"])
    model.to(device)

    if args.enable_matryoshka_from is not None:
        matryoshka_sizes = [
            args.enable_matryoshka_from * 2**i for i in range(int(math.log2(args.hidden_size // args.enable_matryoshka_from)) + 1)
        ]
        print(f"Matryoshka evaluation enabled with hidden sizes: {matryoshka_sizes}")
    else:
        print("Matryoshka evaluation disabled. Using single hidden size.")
        matryoshka_sizes = [args.hidden_size]

    results_path = Path(args.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    
    output = []
    
    for i, W_slice in enumerate(matryoshka_sizes):
        is_last = (i == len(matryoshka_sizes) - 1)

        print('Evaluating with W_slice =', W_slice)

        results = evaluate(
            model,
            inputs_path=Path(args.inputs_path),
            labels_path=Path(args.labels_path),
            batch_size=args.batch_size,
            device=device,
            ignore_label_id=metadata["ignore_label_id"],
            max_recurrent_steps=args.max_recurrent_steps,
            W_slice=W_slice,
            enable_tracking=is_last
        )

        output.append({
            "W_slice": W_slice,
            "val_loss": f"{results['val_loss']:.4f}",
            "token_acc": f"{results['token_acc']:.4f}",
            "exact_match": f"{results['exact_match']:.4f}"
        })

        # save directly last trajectories
        if is_last:
            np.savez(results_path / 'trajectories.npz', **results['trajectories']) # type: ignore


    with open(results_path / 'results.json', 'w') as f:
        json.dump(output, f, indent=4)


    print('Results saved at ', results_path)