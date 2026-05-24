import argparse
import copy
import torch
import torch.nn as nn
import json

from puzzle_datasets.puzzle_dataset import PuzzleDataset, IGNORE_LABEL_ID
from model.trm import TinyRecursiveModel
from model.layers import TransformerBlock
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from pathlib import Path
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from model.stablemax_ce_loss import stablemax_cross_entropy
from evaluation import evaluate_model


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
    use_ema: bool = True,
    patience: int = 5,
    checkpoint_path: str = "checkpoints/best.pt",
):
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    model.train()

    ema_model = None
    if use_ema:
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
    resume_path = checkpoint_file
    if resume_path.exists():
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if use_ema and ema_model is not None:
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

                if use_ema and ema_model is not None:
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
                
                inputs = inputs[~halt_mask]
                outputs = outputs[~halt_mask]
                latents = latents[~halt_mask]
                labels = labels[~halt_mask]
                
                if outputs.numel() == 0:
                    break

        # ── Validation ────────────────────────────────────────────────────
        metrics = evaluate_model(
            model=model if not use_ema else ema_model, # type: ignore
            val_dataset=val_dataset,
            device=device,
            max_recurrent_steps=max_recurrent_steps,
            probabilist_trm=False
        )
        print(
            f"[epoch {epoch}] "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"token_acc={metrics['token_acc']:.4f} | "
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
                "ema_model_state_dict": (ema_model.state_dict() if use_ema else model.state_dict()), # type: ignore
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "patience_counter": patience_counter,
            }, checkpoint_file)

            print(f"  ✓ new best exact_match={best_exact_match:.4f}, checkpoint saved")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Restore best ──────────────────────────────────────────────────────
    best_ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    if use_ema and ema_model is not None:
        ema_model.load_state_dict(best_ckpt["ema_model_state_dict"])
    
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
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA updates and validate/train the base model normally")
    parser.add_argument("--max-recurrent-steps", type=int, default=16,
                        help="Maximum number of recurrent steps during training and validation")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs with no improvement before early stopping")
    parser.add_argument("--mock", action="store_true",
                        help="Use nn.Identity() as the network for rapid debugging")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt",
                        help="Path to save/load the model checkpoint")
    return parser.parse_args()



# quick demo on local CPU
# python train.py \
#   --train-inputs-path data/sudoku4x4/train/inputs.npy \
#   --train-labels-path data/sudoku4x4/train/labels.npy \
#   --test-inputs-path data/sudoku4x4/test/inputs.npy \
#   --test-labels-path data/sudoku4x4/test/labels.npy \
#   --batch-size 4 \
#   --epochs 2 \
#   --learning-rate 1e-4 \
#   --ema-decay 0.999 \
#   --max-recurrent-steps 3 \
#   --mock \
#   --no-ema \
#   --checkpoint-path checkpoints/sudoku4x4.pt


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
    print(f"EMA enabled: {not args.no_ema}")

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
    print(f"Max recurrent steps: {args.max_recurrent_steps}")

    if args.no_ema:
        print("EMA updates disabled. Validating and checkpointing the base model.")
    else:
        print(f"EMA decay: {args.ema_decay}. Validating and checkpointing the EMA model.")

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
        use_ema=not args.no_ema,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
    )
    
    print("--- Training finished ---")