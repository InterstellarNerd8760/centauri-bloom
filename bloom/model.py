"""Small causal transformer. No pretrained weights or external code."""
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class Config:
    vocab_size: int = 256
    context: int = 32
    width: int = 64
    heads: int = 4
    layers: int = 2

class Bloom(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4 * config.width,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, config.layers,
                                           enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.head.weight = self.tokens.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        if not 1 <= n <= self.config.context:
            raise ValueError("sequence exceeds context")
        positions = torch.arange(n, device=tokens.device)
        hidden = self.tokens(tokens) + self.positions(positions)
        mask = torch.ones(n, n, device=tokens.device, dtype=torch.bool).triu(1)
        return self.head(self.norm(self.blocks(hidden, mask=mask)))
