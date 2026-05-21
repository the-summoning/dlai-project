import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TinyRecursiveModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        network: nn.Module,
        n: int = 6,
        T: int = 3,
        num_registers: int = 16,
    ):
        super().__init__()

        self.output_init_embed = nn.Parameter(torch.randn(hidden_size) * 1e-2)
        self.latent_init_embed = nn.Parameter(torch.randn(hidden_size) * 1e-2)
        
        self.n = n
        self.T = T
        
        self.num_registers = max(num_registers, 1) # at least one like [CLS], used to halt
        self.register_tokens = nn.Parameter(torch.randn(self.num_registers, hidden_size) * 1e-2)

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

    def get_initial_for_inputs(self, inputs):
        batch_size, seq_len, _ = inputs.shape

        outputs = self.output_init_embed.expand(batch_size, seq_len, -1)  # (b n d)
        latents = self.latent_init_embed.expand(batch_size, seq_len, -1)  # (b n d)

        return outputs, latents

    def latent_recursion(self, x, y, z):
        for _ in range(self.n):
            z = self.network(y + z + x)
        
        y = self.network(y + z)

        return y, z

    def embedding(self, x):
        inputs = self.embed_tokens(x)
        registers = self.register_tokens.expand(inputs.shape[0], -1, -1)

        return torch.cat([registers, inputs], dim=1)

    # deep supervision
    def forward(self, x, y, z):
        # # recursing T−1 times to improve outputs and latents (no gradients needed)
        with torch.no_grad():
            for _ in range(self.T - 1): # high level
                y, z = self.latent_recursion(x, y, z)
        
        # recursing once to improve outputs and latents
        y, z = self.latent_recursion(x, y, z)

        logits = self.prediction_head(y[:, self.num_registers:, :])
        halt_logits = self.halt_head(y[:, 0, :]).squeeze(-1)  # (B,)

        return y.detach(), z.detach(), logits, halt_logits


    @torch.inference_mode()
    def predict(self, inputs, N_supervision):
        x = self.embedding(inputs)
        y, z = self.get_initial_for_inputs(x)

        for step in range(N_supervision):            
            y, z, predictions, halt_logits = self.forward(x, y, z)

        return y, z, predictions, halt_logits