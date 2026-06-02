import argparse
import copy
import torch
import torch.nn as nn
import json
import math

from puzzle_datasets.puzzle_dataset import PuzzleDataset, IGNORE_LABEL_ID
from model.trm import TinyRecursiveModel
from model.layers import TransformerBlock
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from pathlib import Path
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_


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
def val_epoch(
    model: TinyRecursiveModel,
    val_dataset: PuzzleDataset,
    device: torch.device,
    max_recurrent_steps: int,
    W_slice: int | None = None
) -> dict:
    model.eval()
    model.disable_tracking()
    
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

        logits = model.predict(dataset_input, max_recurrent_steps, W_slice)

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

    return {
        "val_loss": total_val_loss / max(total_puzzles, 1),
        "token_acc": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_puzzles  / max(total_puzzles, 1),
        "trajectories": None,
    }


def train(
    model: TinyRecursiveModel,
    train_dataset: PuzzleDataset,
    val_dataset: PuzzleDataset,
    device: torch.device,
    matryoshka_sizes: list[int] = [512],
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 16,
    epochs: int = 2,
    halt_prob_thres: float = 0.5,
    max_recurrent_steps: int = 12,
    ema_decay: float = 0.999,
    use_ema: bool = True,
    patience: int = 5,
    checkpoint_path: Path = Path("checkpoints/checkpoint.pt"),
    resume_from_checkpoint: Path | None = None,
    eval_interval: int = 1,
):
    assert matryoshka_sizes is not None and len(matryoshka_sizes) > 0

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
    scheduler = ReduceLROnPlateau(
        optim,
        mode="max",
        factor=0.5,
        patience=patience // 2,
        threshold=1e-4,
        min_lr=1e-6
    )

    num_batches = train_dataset.num_groups // train_dataset.batch_size
    global_step = 0
    start_epoch = 1
    best_exact_match = -1.0
    patience_counter = 0

    # ── Resume from checkpoint
    if resume_from_checkpoint is not None and resume_from_checkpoint.exists():
        print(f"Resuming from {resume_from_checkpoint}")
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        
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
    
    weights = [1 / (len(matryoshka_sizes) - i) for i in range(len(matryoshka_sizes))]

    for epoch in range(start_epoch, epochs + 1):
        pbar = tqdm(train_dataset, total=num_batches, desc=f"epoch {epoch}")

        for batch in pbar:
            dataset_input = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            x = model.embedding(dataset_input)
            y = torch.zeros(x.shape[0], x.shape[1], model.hidden_size, device=x.device)
            z = torch.zeros(x.shape[0], x.shape[1], model.hidden_size, device=x.device)

            total_halted = 0

            for step in range(1, max_recurrent_steps + 1):
                mask = (labels != IGNORE_LABEL_ID)

                loss_counts = mask.sum(-1)
                # loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

                y, z = model(x, y, z)

                total_loss = torch.zeros((), device=device)

                for m, w in zip(matryoshka_sizes, weights):
                    W_m = model.prediction_head.weight[:, :m]
                    Q_m = model.halt_head.weight[:, :m]

                    logits = F.linear(y[:, model.num_registers:, :m], W_m)
                    halt_logits = F.linear(y[:, 0, :m], Q_m, model.halt_head.bias).squeeze(-1)

                    is_correct = mask & (logits.argmax(dim=-1) == labels)
                    seq_is_correct = is_correct.sum(-1) == loss_counts
    
                    # lm_loss = (
                    #     stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
                    # ).mean()
                    lm_loss = F.cross_entropy(
                        logits.reshape(-1, vocab_size),
                        labels.reshape(-1),
                        ignore_index=IGNORE_LABEL_ID,
                        reduction="mean"
                    )
                    
                    q_halt_loss = F.binary_cross_entropy_with_logits(
                        halt_logits, seq_is_correct.to(halt_logits.dtype), reduction="mean"
                    )
                    total_loss += w * (lm_loss + 0.5 * q_halt_loss)

                y = y.detach()
                z = z.detach()

                total_loss.backward()
                
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optim.step()
                optim.zero_grad(set_to_none=True)
                
                if use_ema and ema_model is not None:
                    update_ema_model(ema_model=ema_model, model=model, decay=ema_decay)

                x = x.detach()
                # y = y.detach()
                # z = z.detach()
                global_step += 1

                halt_mask = halt_logits.sigmoid() >= halt_prob_thres
                total_halted += halt_mask.sum().item()

                pbar.set_postfix(
                    loss=f"{total_loss.item():.3f}",
                    step=f"{step}/{max_recurrent_steps}",
                    active=f"{y.shape[0]}/{batch_size}",
                    ema=f"{ema_decay:.4f}",
                    halted=f"{total_halted}",
                    grad=f"{grad_norm:.2f}"
                )

                if not halt_mask.any():
                    continue
                
                x = x[~halt_mask]
                y = y[~halt_mask]
                z = z[~halt_mask]
                labels = labels[~halt_mask]
                
                if y.numel() == 0:
                    break

        if epoch % eval_interval == 0:
            # ── Validation
            metrics = val_epoch(
                model=model if not use_ema else ema_model, # type: ignore
                val_dataset=val_dataset,
                device=device,
                max_recurrent_steps=max_recurrent_steps
            )

            scheduler.step(metrics["exact_match"])

            print(
                f"[epoch {epoch}] "
                f"val_loss={metrics['val_loss']:.4f} | "
                f"token_acc={metrics['token_acc']:.4f} | "
                f"exact_match={metrics['exact_match']:.4f}"
            )

            # ── Checkpoint + Early stopping
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


    print("Training complete")
    return model, ema_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train TinyRecursiveModel on maze dataset")
    parser.add_argument("--train-inputs-path", type=str, default="data/maze-30x30-hard-1k/train/inputs.npy",
                        help="Path to training inputs .npy file")
    parser.add_argument("--train-labels-path", type=str, default="data/maze-30x30-hard-1k/train/labels.npy",
                        help="Path to training labels .npy file")
    parser.add_argument("--val-inputs-path", type=str, default="data/maze-30x30-hard-1k/test/inputs.npy",
                        help="Path to val inputs .npy file")
    parser.add_argument("--val-labels-path", type=str, default="data/maze-30x30-hard-1k/test/labels.npy",
                        help="Path to val labels .npy file")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for the PuzzleDataset")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help="Weight decay coefficient for AdamW optimizer")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate")
    parser.add_argument("--no-ema", action="store_true",
                        help="Disable EMA updates and validate/train the base model normally")
    parser.add_argument("--max-recurrent-steps", type=int, default=16,
                        help="Maximum number of recurrent steps during training and validation")
    parser.add_argument("--T", type=int, default=3,
                        help="Number of high level iterations (T) (# times the model can re-process its own output)")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of low level iterations (n) (# times the model processes the input before producing an output)")
    parser.add_argument("--registers", type=int, default=16,
                        help="Number of registers to be used in the model (including CLS token)")
    parser.add_argument("--expansion", type=float, default=4.0,
                        help="Expansion factor for the hidden size in the Transformer layers")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads in the Transformer layers")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs with no improvement before early stopping")
    parser.add_argument("--mock", action="store_true",
                        help="Use nn.Identity() as the network for rapid debugging")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best.pt",
                        help="Path to save the model checkpoint")
    parser.add_argument("--resume-from-checkpoint", type=str, default="checkpoints/old-best.pt",
                        help="Load a previous checkpoint")
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="Hidden size for TransformerBlock layers and model")
    parser.add_argument("--num-transformer-blocks", type=int, default=4,
                        help="Number of TransformerBlock layers in the network")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Evaluation interval in epochs")
    parser.add_argument("--enable-matryoshka-from", type=int, default=None,
                        help="Enable Matryoshka starting from this size (inclusive) and then double to reach hidden size")
    parser.add_argument("--mlp-t", action="store_true",
                        help="Use MLP for temporal mixing instead of self-attention")
    
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
    val_group_indices_path = Path(args.val_inputs_path).parent / "group_indices.npy"

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
        inputs_path=args.val_inputs_path,
        labels_path=args.val_labels_path,
        group_indices_path=str(val_group_indices_path),
        batch_size=args.batch_size,
        seed=42,
        ignore_label_id=metadata["ignore_label_id"]
    )    

    num_registers = args.registers # just CLS token, used to halt
    
    print(f"Vocab size: {vocab_size}, Seq len: {seq_len}")
    print(f"Train dataset: {args.train_inputs_path}")
    print(f"Val dataset: {args.val_inputs_path}")
    print(f"EMA enabled: {not args.no_ema}")

    # Initialize model
    print("Creating model...")

    if args.mock:
        network = nn.Identity()
        print("Using mock network: nn.Identity()")
    else:
        blocks = []
        for _ in range(args.num_transformer_blocks):
            blocks.append(
                TransformerBlock(
                    num_heads=args.num_heads,
                    hidden_size=args.hidden_size,
                    expansion=args.expansion,
                    seq_len=seq_len,
                    num_registers=num_registers,
                    use_mlp_t=args.mlp_t
                )
            )
        network = nn.Sequential(*blocks)

    model = TinyRecursiveModel(
        hidden_size=args.hidden_size,
        vocab_size=vocab_size,
        network=network,
        n=args.n, # 6
        T=args.T, # 3
        num_registers=num_registers,
        track_latents=False
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
    
    # [16, 32..., hidden_size]
    if args.enable_matryoshka_from is not None:
        matryoshka_sizes = [
            args.enable_matryoshka_from * 2**i for i in range(int(math.log2(args.hidden_size // args.enable_matryoshka_from)) + 1)
        ]
        print(f"Matryoshka training enabled with hidden sizes: {matryoshka_sizes}")
    else:
        print("Matryoshka training disabled. Using single hidden size.")
        matryoshka_sizes = [args.hidden_size]

    trained_model, ema_model = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        matryoshka_sizes=matryoshka_sizes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        halt_prob_thres=0.5,
        max_recurrent_steps=args.max_recurrent_steps,
        ema_decay=args.ema_decay,
        use_ema=not args.no_ema,
        patience=args.patience,
        checkpoint_path=Path(args.checkpoint_path),
        resume_from_checkpoint=Path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        eval_interval=args.eval_interval
    )
    
    print("--- Training finished ---")