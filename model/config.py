from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_len: int = 2048
    n_layers: int = 12
    n_heads: int = 6
    d_model: int = 384
    d_ff: int | None = None
    dropout: float = 0.0
    bias: bool = False
    rope: bool = True
    tied_weights: bool = False

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = int(self.d_model * 8 / 3)
            self.d_ff = ((self.d_ff + 255) // 256) * 256

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def parameter_count_estimate(self) -> int:
        embed = self.vocab_size * self.d_model
        if not self.tied_weights:
            embed += self.d_model * self.vocab_size
        attn = self.n_layers * (4 * self.d_model * self.d_model + (2 if self.bias else 0) * self.d_model)
        ffn = self.n_layers * (2 * self.d_model * self.d_ff + (2 if self.bias else 0) * (self.d_model + self.d_ff))
        norms = self.n_layers * 4 * self.d_model + 2 * self.d_model
        return embed + attn + ffn + norms


SMALL = ModelConfig(
    vocab_size=32000,
    max_seq_len=1024,
    n_layers=6,
    n_heads=6,
    d_model=384,
    dropout=0.0,
)

MEDIUM = ModelConfig(
    vocab_size=32000,
    max_seq_len=2048,
    n_layers=12,
    n_heads=8,
    d_model=512,
    dropout=0.0,
)
