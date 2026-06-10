from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def compute_acceleration(y, y_prev, delta_prev, eps=1e-8):
    delta_cur = y - y_prev  # [Batch, Hidden_Dim]
    
    if delta_prev is None:
        return None, delta_cur
        
    diff = delta_cur - delta_prev  # [Batch, Hidden_Dim]
    
    acceleration = torch.norm(diff, p=2, dim=-1)
    
    norm_cur = torch.norm(delta_cur, p=2, dim=-1)
    norm_prev = torch.norm(delta_prev, p=2, dim=-1)
    acceleration = acceleration / (norm_cur + norm_prev + eps)
    
    return acceleration, delta_cur



class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        network: nn.Module,
        n: int = 6,
        T: int = 3,
        num_registers: int = 16
    ):
        super().__init__()

        self.track_latents = False

        self.n = n
        self.T = T
        
        self.num_registers = max(num_registers, 1) # at least one like [CLS], used to halt
        self.register_tokens = nn.Parameter(torch.randn(self.num_registers, hidden_size) * 1e-2)
        self.hidden_size = hidden_size

        self.network = network
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # heads
        self.prediction_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.halt_head = nn.Linear(hidden_size, 1, bias = True)

        # init weights
        self.apply(self._init_weights)

        # halt head init override
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -5)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=1.0 / (module.in_features ** 0.5)) # Truncated LeCun normal
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)    

    def enable_tracking(self):
        self.track_latents = True
        self.trajectory = {"y": [], "z": []}

    def disable_tracking(self):
        self.track_latents = False

    def track(self, tensor, latent_type: Literal["y", "z"]):
        if self.track_latents:
            self.trajectory[latent_type].append(tensor.cpu().numpy()) # type: ignore

    def embedding(self, x):
        inputs = self.embed_tokens(x)
        registers = self.register_tokens.expand(inputs.shape[0], -1, -1)

        return torch.cat([registers, inputs], dim=1)

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            z = self.network(y + z + x)
            self.track(z, "z")
        
        y = self.network(y + z)
        self.track(y, "y")

        return y, z

    # deep supervision
    def forward(self, x, y, z):
        with torch.no_grad():
            for _ in range(self.T - 1): # high level
                y, z = self.latent_recursion(x, y, z)
                
        y, z = self.latent_recursion(x, y, z)

        return y, z

    @torch.inference_mode()
    def predict(
        self,
        inputs: torch.Tensor, 
        N_supervision: int,
        W_slice: int | None = None
    ):
        W_slice = W_slice or self.prediction_head.weight.shape[1] # default to hidden size

        x = self.embedding(inputs)
        y = torch.zeros(x.shape[0], x.shape[1], self.hidden_size, device=x.device)
        z = torch.zeros(x.shape[0], x.shape[1], self.hidden_size, device=x.device)
    
        for _ in range(N_supervision):            
            y, z = self.forward(x, y, z)

        W_m = self.prediction_head.weight[:, :W_slice]

        logits = F.linear(y[:, self.num_registers:, :W_slice], W_m)

        return logits 
    

    # questa implementazione semplicemente scarta i risultati per i sample haltati
    # a livello produttivo per batch size giganti invece conviene proprio rimuoverli i sample (dynamic batch)
    @torch.inference_mode()
    def predict_with_early_stopping(
        self,
        inputs: torch.Tensor, 
        N_supervision: int,
        W_slice: int | None = None,
        tau: float = 1e-2,
        eps: float = 1e-8
    ):
        W_slice = W_slice or self.prediction_head.weight.shape[1]

        x = self.embedding(inputs)
        batch_size = x.shape[0]
        
        y = torch.zeros(batch_size, x.shape[1], self.hidden_size, device=x.device)
        z = torch.zeros(batch_size, x.shape[1], self.hidden_size, device=x.device)
        
        final_y = y.clone()
        
        halt_steps = torch.full((batch_size,), N_supervision, dtype=torch.long, device=x.device)

        delta_prev = None
        prev_small = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=x.device)

        for step in range(N_supervision):
            if not active_mask.any():
                break

            y_old = y.clone()
            
            y_next, z_next = self.forward(x, y, z)
            
            delta_cur = y_next - y_old
            
            if delta_prev is not None:
                diff_delta = delta_cur - delta_prev # [Batch, Seq_Len, Hidden_Dim]
                
                acc_norm = torch.linalg.vector_norm(diff_delta, ord=2, dim=(-2, -1))
                norm_cur = torch.linalg.vector_norm(delta_cur, ord=2, dim=(-2, -1))
                norm_prev = torch.linalg.vector_norm(delta_prev, ord=2, dim=(-2, -1))
                
                acceleration = acc_norm / (norm_cur + norm_prev + eps)

                #print(f'Step {step} -> mean acceleration {acceleration.mean(axis=0):6f}')
                
                small = acceleration < tau
                
                halt_condition = small & prev_small & active_mask
                
                if halt_condition.any():
                    mask_3d = halt_condition.unsqueeze(-1).unsqueeze(-1)
                    
                    final_y = torch.where(mask_3d, y_old, final_y)                    
                    halt_steps = torch.where(halt_condition, torch.tensor(step, dtype=torch.long, device=x.device), halt_steps)                    
                    active_mask = active_mask & ~halt_condition
                
                prev_small = torch.where(active_mask, small, prev_small)

            delta_prev = delta_cur.clone()
            
            mask_active_3d = active_mask.unsqueeze(-1).unsqueeze(-1)
            y = torch.where(mask_active_3d, y_next, y)
            z = torch.where(mask_active_3d, z_next, z)

        mask_final_3d = active_mask.unsqueeze(-1).unsqueeze(-1)
        final_y = torch.where(mask_final_3d, y, final_y)

        W_m = self.prediction_head.weight[:, :W_slice]
        logits = F.linear(final_y[:, self.num_registers:, :W_slice], W_m)

        return logits, halt_steps.cpu()

    # @torch.inference_mode()
    # def probabilistic_predict(
    #     self,
    #     inputs: torch.Tensor,
    #     N_supervision: int,
    #     rollouts: int = 32,
    #     std: float = 0.2,
    # ):
    #     B = inputs.shape[0]

    #     x = self.embedding(inputs)

    #     # (R*B, ...)
    #     x = x.repeat(rollouts, 1, 1)

    #     y, z = self.get_initial_for_inputs(x)

    #     noise = torch.randn(
    #         N_supervision,
    #         *z.shape,
    #         device=z.device,
    #         dtype=z.dtype,
    #     ).mul_(std)

    #     for i in range(N_supervision):
    #         z = z + noise[i]
    #         y, z, logits, halt_logits = self.forward(x, y, z)

    #     y = y.reshape(
    #         rollouts,
    #         B,
    #         *y.shape[1:]
    #     )

    #     z = z.reshape(
    #         rollouts,
    #         B,
    #         *z.shape[1:]
    #     )

    #     logits = logits.reshape(
    #         rollouts,
    #         B,
    #         *logits.shape[1:]
    #     )

    #     halt_logits = halt_logits.reshape(
    #         rollouts,
    #         B,
    #         *halt_logits.shape[1:]
    #     )

    #     best_idx = halt_logits.squeeze(-1).argmax(0)

    #     batch_idx = torch.arange(
    #         B,
    #         device=x.device,
    #     )

    #     return (
    #         y[best_idx, batch_idx],
    #         z[best_idx, batch_idx],
    #         logits[best_idx, batch_idx],
    #         halt_logits[best_idx, batch_idx],
    #     )