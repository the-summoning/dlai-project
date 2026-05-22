import argparse
import copy
import torch
import torch.nn as nn
import json

from datasets.puzzle_dataset import PuzzleDataset
from model.trm import TinyRecursiveModel
from model.layers import TransformerBlock
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from pathlib import Path
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from datasets.puzzle_dataset import IGNORE_LABEL_ID


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / ( 1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log( s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)



@torch.no_grad()
def update_ema_model(
    ema_model: TinyRecursiveModel,
    model: TinyRecursiveModel,
    decay: float,
):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name in ema_params:
        ema_params[name].mul_(decay).add_(
            model_params[name],
            alpha=1.0 - decay
        )

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())

    for name in ema_buffers:
        ema_buffers[name].copy_(model_buffers[name])



@torch.inference_mode()
def validate(
    ema_model: TinyRecursiveModel,
    val_dataset: PuzzleDataset,
    device: torch.device,
    max_recurrent_steps: int,
) -> dict:
    ema_model.eval()
    num_batches = val_dataset.num_groups // val_dataset.batch_size

    total_tokens = 0
    correct_tokens = 0
    total_puzzles = 0
    exact_puzzles = 0
    total_val_loss = 0.0
    correct_path_tokens = 0
    total_path_tokens = 0

    wall_id = int(val_dataset.char2id[ord('#')])

    pbar = tqdm(val_dataset, total=num_batches, desc="Validation")
    
    for batch in pbar:
        dataset_input = batch["inputs"].to(device)
        labels = batch["labels"].to(device)

        _, _, logits, _ = ema_model.predict(dataset_input, N_supervision=max_recurrent_steps)
        final_preds = logits.argmax(-1)  # (B, seq)

        # ── Loss ──────────────────────────────────────────────────────────
        mask = labels != IGNORE_LABEL_ID
        loss_counts = mask.sum(-1).clamp_min(1)
        lm_loss = (
            stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
            / loss_counts.unsqueeze(-1)
        ).sum()
        total_val_loss += lm_loss.item()

        # ── Token Accuracy ─────────────────────────────────────────────────
        correct_tokens += (mask & (final_preds == labels)).sum().item()
        total_tokens   += mask.sum().item()

        # ── Exact Match ────────────────────────────────────────────────────
        exact_puzzles += ((final_preds == labels) | ~mask).all(dim=-1).sum().item()
        total_puzzles += labels.shape[0]

        # ── Path Token Accuracy (esclude muri e padding) ──────────────────────
        path_mask = mask & (labels != wall_id)
        correct_path_tokens += (path_mask & (final_preds == labels)).sum().item()
        total_path_tokens += path_mask.sum().item()

    return {
        "val_loss": total_val_loss / max(total_puzzles, 1),
        "token_acc": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_puzzles  / max(total_puzzles, 1),
        "path_token_acc": correct_path_tokens / max(total_path_tokens, 1),
    }



def train(
    model: TinyRecursiveModel,
    train_dataset: PuzzleDataset,
    val_dataset: PuzzleDataset,
    device: torch.device,
    learning_rate: float = 1e-4,
    weight_decay: float = 1.0,
    batch_size: int = 16,
    epochs: int = 2,
    halt_prob_thres: float = 0.5,
    max_recurrent_steps: int = 12,
    warmup_steps: int = 2000,
    ema_decay: float = 0.999,
    patience: int = 5,
    checkpoint_dir: str = "checkpoints",
):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    ema_model = copy.deepcopy(model)
    ema_model.to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = LambdaLR(optim, lambda step: min((step + 1) / warmup_steps, 1.0))

    num_batches = train_dataset.num_groups // train_dataset.batch_size
    global_step = 0
    start_epoch = 1
    best_exact_match = -1.0
    patience_counter = 0

    # ── Resume from checkpoint ────────────────────────────────────────────
    resume_path = checkpoint_path / "best.pt"
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        best_exact_match = ckpt["metrics"]["exact_match"]
        patience_counter = ckpt["patience_counter"]
        print(f"  resumed from epoch {ckpt['epoch']}, best_exact_match={best_exact_match:.4f}, patience={patience_counter}/{patience}")

    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(train_dataset, total=num_batches, desc=f"epoch {epoch}")

        for batch in pbar:
            dataset_input = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            inputs = model.embedding(dataset_input)
            outputs, latents = model.get_initial_for_inputs(inputs)

            total_halted = 0

            for step in range(1, max_recurrent_steps + 1):
                outputs, latents, logits, halt_logits = model(inputs, latents, outputs)

                mask = (labels != IGNORE_LABEL_ID)
                loss_counts = mask.sum(-1)
                loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
                is_correct = mask & (logits.argmax(dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts

                lm_loss = (
                    stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
                ).sum()
                
                q_halt_loss = F.binary_cross_entropy_with_logits(
                    halt_logits, seq_is_correct.to(halt_logits.dtype), reduction="sum"
                )
                total_loss = lm_loss + 0.5 * q_halt_loss

                total_loss.backward()
                
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optim.step()
                optim.zero_grad(set_to_none=True)
                
                scheduler.step()
                update_ema_model(ema_model=ema_model, model=model, decay=ema_decay)

                inputs = inputs.detach()
                outputs = outputs.detach()
                latents = latents.detach()
                global_step += 1

                halt_mask = halt_logits.sigmoid() >= halt_prob_thres
                total_halted += halt_mask.sum().item()

                pbar.set_postfix(
                    loss=f"{total_loss.item():.3f}",
                    step=f"{step}/{max_recurrent_steps}",
                    active=f"{outputs.shape[0]}/{batch_size}",
                    ema=f"{ema_decay:.4f}",
                    halted=f"{total_halted}",
                )

                if not halt_mask.any():
                    continue
                
                inputs  = inputs[~halt_mask]
                outputs = outputs[~halt_mask]
                latents = latents[~halt_mask]
                labels  = labels[~halt_mask]
                
                if outputs.numel() == 0:
                    break

        # ── Validation ────────────────────────────────────────────────────
        metrics = validate(
            ema_model=ema_model,
            val_dataset=val_dataset,
            device=device,
            max_recurrent_steps=max_recurrent_steps,
        )
        print(
            f"[epoch {epoch}] "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"token_acc={metrics['token_acc']:.4f} | "
            f"path_token_acc={metrics['path_token_acc']:.4f} | "
            f"exact_match={metrics['exact_match']:.4f}"
        )

        # ── Checkpoint + Early stopping ───────────────────────────────────
        if metrics["exact_match"] > best_exact_match:
            best_exact_match = metrics["exact_match"]
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "patience_counter": patience_counter,
            }, checkpoint_path / "best.pt")

            print(f"  ✓ new best exact_match={best_exact_match:.4f}, checkpoint saved")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Restore best ──────────────────────────────────────────────────────
    best_ckpt = torch.load(checkpoint_path / "best.pt", map_location=device)
    ema_model.load_state_dict(best_ckpt["ema_model_state_dict"])
    model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Restored best checkpoint from epoch {best_ckpt['epoch']} (exact_match={best_ckpt['metrics']['exact_match']:.4f})")

    print("Training complete")
    return model, ema_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveModel on maze dataset")
    parser.add_argument("--train-inputs-path", type=str, default="data/maze-30x30-hard-1k/train/inputs.npy",
                        help="Path to training inputs .npy file")
    parser.add_argument("--train-labels-path", type=str, default="data/maze-30x30-hard-1k/train/labels.npy",
                        help="Path to training labels .npy file")
    parser.add_argument("--test-inputs-path", type=str, default="data/maze-30x30-hard-1k/test/inputs.npy",
                        help="Path to test inputs .npy file")
    parser.add_argument("--test-labels-path", type=str, default="data/maze-30x30-hard-1k/test/labels.npy",
                        help="Path to test labels .npy file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for the PuzzleDataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--max-recurrent-steps", type=int, default=16,
                        help="Maximum number of recurrent steps during training and validation")
    parser.add_argument("--mock", action="store_true",
                        help="Use nn.Identity() as the network for rapid debugging")
    return parser.parse_args()



# python train.py \
#   --train-inputs-path data/sudoku4x4/train/inputs.npy \
#   --train-labels-path data/sudoku4x4/train/labels.npy \
#   --test-inputs-path data/sudoku4x4/test/inputs.npy \
#   --test-labels-path data/sudoku4x4/test/labels.npy \
#   --batch-size 32 \
#   --epochs 10 \
#   --learning-rate 1e-4 \
#   --ema-decay 0.999 \
#   --max-recurrent-steps 16

if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Derive metadata path from training inputs path using pathlib
    data_dir = Path(args.train_inputs_path).parent
    group_indices_path = data_dir / "group_indices.npy"
    test_group_indices_path = Path(args.test_inputs_path).parent / "group_indices.npy"

    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    vocab_size = metadata["vocab_size"]
    seq_len = metadata["seq_len"]
    
    # Load dataset
    print("Loading datasets...")
    train_dataset = PuzzleDataset(
        inputs_path=args.train_inputs_path,
        labels_path=args.train_labels_path,
        group_indices_path=str(group_indices_path),
        batch_size=args.batch_size,
        seed=42,
        ignore_label_id=metadata["ignore_label_id"]
    )
    val_dataset = PuzzleDataset(
        inputs_path=args.test_inputs_path,
        labels_path=args.test_labels_path,
        group_indices_path=str(test_group_indices_path),
        batch_size=args.batch_size,
        seed=42,
        ignore_label_id=metadata["ignore_label_id"]
    )    

    num_registers = 1 # just CLS token, used to halt
    
    print(f"Vocab size: {vocab_size}, Seq len: {seq_len}")
    print(f"Train dataset: {args.train_inputs_path}")
    print(f"Test dataset: {args.test_inputs_path}")

    # Initialize model
    print("Creating model...")

    if args.mock:
        network = nn.Identity()
        print("Using mock network: nn.Identity()")
    else:
        network = nn.Sequential(
            TransformerBlock(
                num_heads=8,
                hidden_size=512,
                expansion=4.0,
                max_seq_len=seq_len + num_registers
            ),
            TransformerBlock(
                num_heads=8,
                hidden_size=512,
                expansion=4.0,
                max_seq_len=seq_len + num_registers
            )
        )

    model = TinyRecursiveModel(
        hidden_size=512,
        vocab_size=vocab_size,
        network=network,
        n=6,
        T=3,
        num_registers=num_registers,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n--- Starting Training ({args.epochs} epochs) ---")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"EMA decay: {args.ema_decay}")
    print(f"Max recurrent steps: {args.max_recurrent_steps}")
    print()
    trained_model, ema_model = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=1.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        halt_prob_thres=0.5,
        max_recurrent_steps=args.max_recurrent_steps,
        warmup_steps=100,
        ema_decay=args.ema_decay,
    )
    
    print("--- Training finished ---")