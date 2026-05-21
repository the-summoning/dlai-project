import copy
import torch
import torch.nn as nn
import json

from maze_dataset import MazeDataset
from model.trm import TinyRecursiveModel
from model.layers import TransformerBlock
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm
from torch.nn import functional as F
from maze_dataset import IGNORE_LABEL_ID


def s(x, epsilon=1e-30):
    return torch.where(
        x < 0,
        1 / ( 1 - x + epsilon),
        x + 1
    )


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


def train(
    model: TinyRecursiveModel,
    dataset: MazeDataset,
    device: torch.device,
    learning_rate: float = 1e-4,
    weight_decay: float = 1.0,
    batch_size: int = 16,
    epochs: int = 2,
    halt_prob_thres: float = 0.5,
    max_recurrent_steps: int = 12,
    warmup_steps: int = 2000,
    ema_decay: float = 0.999,
):
    model.to(device)
    model.train()

    # EMA model
    ema_model = copy.deepcopy(model)
    ema_model.to(device)
    ema_model.eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    optim = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = LambdaLR(
        optim,
        lambda step: min((step + 1) / warmup_steps, 1.0)
    )

    num_batches = dataset.num_groups // dataset.batch_size

    global_step = 0

    for epoch in range(1, epochs + 1):
        pbar = tqdm(
            dataset,
            total=num_batches,
            desc=f"epoch {epoch}"
        )

        for batch in pbar:
            dataset_input = batch["inputs"].to(device)
            labels = batch["labels"].to(device)

            inputs = model.embedding(dataset_input)
            outputs, latents = model.get_initial_for_inputs(inputs)

            for step in range(1, max_recurrent_steps + 1):
                outputs, latents, logits, halt_logits = model(
                    inputs,
                    latents,
                    outputs
                )

                # Correctness
                mask = (labels != IGNORE_LABEL_ID)
                loss_counts = mask.sum(-1)
                loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

                is_correct = mask & (logits.argmax(dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts

                lm_loss = (
                    stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor
                ).sum()
                q_halt_loss = F.binary_cross_entropy_with_logits(halt_logits, seq_is_correct.to(halt_logits.dtype), reduction="sum")

                total_loss = lm_loss + 0.5 * q_halt_loss
                total_loss.backward()

                optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

                # EMA update
                update_ema_model(
                    ema_model=ema_model,
                    model=model,
                    decay=ema_decay
                )

                inputs = inputs.detach()
                outputs = outputs.detach()
                latents = latents.detach()

                global_step += 1

                active = outputs.shape[0]

                pbar.set_postfix(
                    loss=f"{total_loss.item():.3f}",
                    step=f"{step}/{max_recurrent_steps}",
                    active=f"{active}/{batch_size}",
                    ema=f"{ema_decay:.4f}"
                )

                halt_mask = halt_logits.sigmoid() >= halt_prob_thres

                if not halt_mask.any():
                    continue

                inputs = inputs[~halt_mask]
                outputs = outputs[~halt_mask]
                latents = latents[~halt_mask]
                labels = labels[~halt_mask]

                if outputs.numel() == 0:
                    break

    print("Training complete")

    return model, ema_model


if __name__ == "__main__":
    # Configuration for debugging
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset paths
    data_dir = "data/maze-30x30-hard-1k/train"

    with open(f"{data_dir}/metadata.json", "r") as f:
        metadata = json.load(f)
            # Get vocab size from dataset
        vocab_size = metadata["vocab_size"]
        seq_len = metadata["seq_len"]
    
    # Load dataset
    print("Loading dataset...")
    dataset = MazeDataset(
        inputs_path=f"{data_dir}/inputs.npy",
        labels_path=f"{data_dir}/labels.npy",
        group_indices_path=f"{data_dir}/group_indices.npy",
        batch_size=8,
        seed=42,
        ignore_label_id=metadata["ignore_label_id"]
    )
    
    num_registers = 16
    
    print(f"Vocab size: {vocab_size}, Seq len: {seq_len}")
    print(f"Dataset has {dataset.num_groups} puzzle groups")

    # Initialize model
    print("Creating model...")

    # network = nn.Identity()

    network = nn.Sequential(
        TransformerBlock(
            num_heads=8,
            hidden_size=512,
            expansion=4.0,
            max_seq_len=metadata["seq_len"] + num_registers
        ),
        TransformerBlock(
            num_heads=8,
            hidden_size=512,
            expansion=4.0,
            max_seq_len=metadata["seq_len"] + num_registers
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
    
    # Start training with 3 epochs for debugging
    print("--- Starting Training (3 epochs) ---")
    trained_model, ema_model = train(
        model=model,
        dataset=dataset,
        device=device,
        learning_rate=1e-4,
        weight_decay=1.0,
        batch_size=8,
        epochs=3,
        halt_prob_thres=0.5,
        max_recurrent_steps=12,
        warmup_steps=100,
        ema_decay=0.999,
    )
    
    print("--- Training finished ---")