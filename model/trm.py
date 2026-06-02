from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


#   def compute_noise(self, z, halt_logits):

#          self.noise_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.GELU(),
#             nn.Linear(hidden_size * 2, hidden_size)
#         )
#         q = halt_logits.sigmoid() 
#         beta = 1.0 - q
#         sigma = F.softplus(self.noise_head(z))
#         eps = torch.randn_like(z)

#         return beta.unsqueeze(-1) * sigma * eps


# IMPARARE ZONA GAUSSIANA CORRETTA condizionata da input
# z ~ p(z|x) = N(mu(x), sigma(x)) + regularization to keep reasons of same input nearby in latent space
# then during inference I can control the spreadness and originality of the reason by moving a bit in the latent space and adding noise



class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        network: nn.Module,
        n: int = 6,
        T: int = 3,
        num_registers: int = 16,
        track_latents: bool = False,
    ):
        super().__init__()

        if track_latents:
            self.enable_tracking()
        else:
            self.disable_tracking()

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
        self.trajectory = {"y": [], "z": [], "halt_logits": []}

    def disable_tracking(self):
        self.track_latents = False

    def track(self, tensor, latent_type: Literal["y", "z", "halt_logits"]):
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
        # # recursing T−1 times to improve outputs and latents (no gradients needed)
        with torch.no_grad():
            for _ in range(self.T - 1): # high level
                y, z = self.latent_recursion(x, y, z)
        
        # recursing once to improve outputs and latents
        y, z = self.latent_recursion(x, y, z)

        return y, z # detached outside in order to use MRL

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
    
        Q_m = self.halt_head.weight[:, :W_slice]

        for _ in range(N_supervision):            
            y, z = self.forward(x, y, z)

            if self.track_latents:
                halt_logits = F.linear(y[:, 0, :W_slice], Q_m, self.halt_head.bias).squeeze(-1)
                self.track(halt_logits, "halt_logits")

        W_m = self.prediction_head.weight[:, :W_slice]

        logits = F.linear(y[:, self.num_registers:, :W_slice], W_m)

        return logits 
    

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