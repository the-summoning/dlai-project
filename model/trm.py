import torch
import torch.nn as nn
from torch.nn import functional as F


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

        # self.y_init = nn.Buffer(
        #     nn.init.trunc_normal_(torch.empty(hidden_size), std=1.0),
        #     persistent=True
        # )
        # self.z_init = nn.Buffer(
        #     nn.init.trunc_normal_(torch.empty(hidden_size), std=1.0),
        #     persistent=True
        # )

        # learnable inits seem to work better
        self.y_init = nn.Parameter(torch.randn(hidden_size) * 1e-2) # nn.Parameter(nn.init.trunc_normal_(torch.empty(hidden_size), std=1e-2))
        self.z_init = nn.Parameter(torch.randn(hidden_size) * 1e-2) # nn.Parameter(nn.init.trunc_normal_(torch.empty(hidden_size), std=1e-2))

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

        y_init = self.y_init.expand(batch_size, seq_len, -1)  # (b n d)
        z_init = self.z_init.expand(batch_size, seq_len, -1)  # (b n d)

        return y_init, z_init

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

        return y.detach(), z.detach()


    @torch.inference_mode()
    def predict(self, inputs: torch.Tensor, N_supervision: int, W_slice: int | None = None):
        W_slice = W_slice or self.prediction_head.weight.shape[1] # default to hidden size\

        x = self.embedding(inputs)
        y, z = self.get_initial_for_inputs(x)

        for _ in range(N_supervision):            
            y, z = self.forward(x, y, z)

        W_m = self.prediction_head.weight[:, :W_slice]
        Q_m = self.halt_head.weight[:, :W_slice]

        logits = F.linear(y[:, self.num_registers:, :W_slice], W_m)
        halt_logits = F.linear(y[:, 0, :W_slice], Q_m, self.halt_head.bias).squeeze(-1)

        return y, z, logits, halt_logits
    

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