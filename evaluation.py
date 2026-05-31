import argparse
import torch
import torch.nn as nn
import json

from model.trm import TinyRecursiveModel
from model.stablemax_ce_loss import stablemax_cross_entropy
from model.layers import TransformerBlock
from pathlib import Path

from puzzle_datasets.puzzle_dataset import PuzzleDataset, IGNORE_LABEL_ID
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_model(
    model: TinyRecursiveModel,
    val_dataset: PuzzleDataset,
    device: torch.device,
    max_recurrent_steps: int,
    W_slice: int | None = None,
    register_trajectories: bool = True,
    # probabilist_trm: bool = False,
    # probabilistic_rollouts: int = 1,
) -> dict:
    model.eval()
    num_batches = val_dataset.num_groups // val_dataset.batch_size

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    exact_puzzles = 0
    total_val_loss = 0.0

    pbar = tqdm(val_dataset, total=num_batches, desc="Validation")
    
    for batch in pbar:
        dataset_input = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        
        # if probabilist_trm:
        #     pass
        #     _, _, logits, _ = model.probabilistic_predict(dataset_input, max_recurrent_steps, probabilistic_rollouts, 0.2)
        # else:
        #     _, _, logits, _ = model.predict(dataset_input, max_recurrent_steps)
        _, _, logits, _, trajectories = model.predict(dataset_input, max_recurrent_steps, W_slice, register_trajectories)

        final_preds = logits.argmax(-1)  # (B, seq)

        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1).clamp_min(1)
        lm_loss = (
            stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
            / loss_counts.unsqueeze(-1)
        ).sum()
        total_val_loss += lm_loss.item()

        correct_tokens += (mask & (final_preds == labels)).sum().item()
        total_tokens   += mask.sum().item()

        exact_puzzles += ((final_preds == labels) | ~mask).all(dim=-1).sum().item()
        total_puzzles += labels.shape[0]


    return {
        "val_loss": total_val_loss / max(total_puzzles, 1),
        "token_acc": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_puzzles  / max(total_puzzles, 1),
        "trajectories": trajectories if register_trajectories else None
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TinyRecursiveModel on a PuzzleDataset")
    parser.add_argument("--inputs-path", type=str,
                        default="data/sudoku4x4/test/inputs.npy",
                        help="Path to validation inputs .npy file")
    parser.add_argument("--labels-path", type=str,
                        default="data/sudoku4x4/test/labels.npy",
                        help="Path to validation labels .npy file")
    parser.add_argument("--group-indices-path", type=str,
                        default="data/sudoku4x4/test/group_indices.npy",
                        help="Path to validation group indices .npy file")
    parser.add_argument("--metadata-path", type=str, default=None,
                        help="Path to metadata.json file (defaults to inputs parent directory)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt",
                        help="Path to the checkpoint file to load")
    parser.add_argument("--trajectories-path", type=str, default="trajectories.pt",
                        help="Path where to save trajectories")
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
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads in the Transformer layers")
    parser.add_argument("--w-slice", type=int, default=None,
                        help="Number of hidden dimensions to use from prediction/halt heads (W_slice).")
    parser.add_argument("--probabilistic-trm", action="store_true",
                        help="Use probabilistic TRM prediction instead of deterministic prediction")
    parser.add_argument("--probabilistic-rollouts", type=int, default=1,
                        help="Number of rollouts when using probabilistic TRM")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden size for TransformerBlock layers and model")
    parser.add_argument("--num-transformer-blocks", type=int, default=4,
                        help="Number of TransformerBlock layers in the network")
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
    num_registers = 1

    val_dataset = PuzzleDataset(
        inputs_path=args.inputs_path,
        labels_path=args.labels_path,
        group_indices_path=args.group_indices_path,
        batch_size=args.batch_size,
        seed=42,
        ignore_label_id=metadata["ignore_label_id"]
    )

    network = nn.Sequential(
        *[
            TransformerBlock(
                num_heads=args.num_heads,
                hidden_size=args.hidden_size,
                expansion=args.expansion,
                seq_len=seq_len,
                num_registers=num_registers
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

    metrics = evaluate_model(
        model,
        val_dataset,
        device,
        max_recurrent_steps=args.max_recurrent_steps,
        W_slice=args.w_slice,
        register_trajectories=True
    )

    print("Evaluation results:")
    print(f"  val_loss={metrics['val_loss']:.4f}")
    print(f"  token_acc={metrics['token_acc']:.4f}")
    print(f"  exact_match={metrics['exact_match']:.4f}")

    trajectories_path = Path(args.trajectories_path)
    trajectories_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(metrics['trajectories'], trajectories_path)

    print('Trajectories saved at ', trajectories_path)

