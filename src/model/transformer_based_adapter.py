import torch
from torch import Tensor
from torch.nn import (
    Linear,
    Module,
    Parameter,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class TransformerBasedAdapter(Module):
    def __init__(
        self,
        *,
        input_size: int,
        prefix_length: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()

        assert prefix_length > 0

        self.prefix_length = prefix_length

        self.input_to_embedding = Linear(input_size, d_model)
        self.pseudo_word_embedding = Parameter(torch.zeros(d_model))
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(
        self,
        input: Tensor,
    ):
        return self.transformer_encoder.forward(
            torch.stack(
                [self.input_to_embedding.forward(input)]
                + [
                    self.pseudo_word_embedding.unsqueeze(0).expand(
                        input.shape[0], -1
                    )
                ]
                * (self.prefix_length - 1),
                dim=1,
            )
        )
