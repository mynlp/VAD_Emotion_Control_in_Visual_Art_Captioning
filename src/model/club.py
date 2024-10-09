from typing import cast

import pydantic
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Linear, Module, ReLU
from torch.nn import functional as F


class CLUBOutput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    loss: Tensor
    estimated_mi: Tensor
    cross_entropy_loss: Tensor


class CLUB(Module):
    """
    Re-implementation of:
    - CLUB: A Contrastive Log-ratio Upper Bound of Mutual Information
    - https://proceedings.mlr.press/v119/cheng20b.html
    The original implmentation can be found at:
    - https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()  # type: ignore

        self.loc_linear_1 = Linear(x_dim, hidden_size)
        self.loc_linear_2 = Linear(hidden_size, y_dim)
        self.logvar_linear_1 = Linear(x_dim, hidden_size)
        self.logvar_linear_2 = Linear(hidden_size, y_dim)
        self.nonlinearity = ReLU()

    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> CLUBOutput:
        """
        Parameters
        ----------
        x: Tensor (batch_size, x_dim)
        y: Tensor (batch_size, y_dim)

        Return
        ------
        Tensor (,)
        """
        batch_size, y_dim = y.shape

        # compute $\log q_{\theta} ( x_{i} | y_{j} )$.
        # $\theta$ is detached.
        crossed_log_likelihood = cast(
            Tensor,
            Normal(
                loc=F.linear(
                    self.nonlinearity.forward(
                        F.linear(
                            x,
                            self.loc_linear_1.weight.detach(),
                            self.loc_linear_1.bias.detach(),
                        )
                    ),
                    self.loc_linear_2.weight.detach(),
                    self.loc_linear_2.bias.detach(),
                ),
                scale=F.linear(
                    self.nonlinearity.forward(
                        F.linear(
                            x,
                            self.logvar_linear_1.weight.detach(),
                            self.logvar_linear_1.bias.detach(),
                        )
                    ),
                    self.logvar_linear_2.weight.detach(),
                    self.logvar_linear_2.bias.detach(),
                )
                .tanh()  # to avoid extreme value
                .mul(0.5)
                .exp(),
            ).log_prob(
                y.unsqueeze(1).expand(
                    batch_size,
                    batch_size,
                    y_dim,
                )
            ),
        ).sum(dim=-1)
        assert crossed_log_likelihood.shape == (batch_size, batch_size)

        estimated_mi = (
            crossed_log_likelihood.diagonal().mean()
            - crossed_log_likelihood.mean()
        )

        # compute $ - \log q_{\theta} ( x_{i} | y_{i} )$.
        # $x_{i}$ and $y_{i}$ are detached.
        cross_entropy_loss = (
            F.gaussian_nll_loss(
                input=self.loc_linear_2.forward(
                    self.nonlinearity.forward(
                        self.loc_linear_1.forward(x.detach())
                    )
                ),
                target=y.detach(),
                var=self.logvar_linear_2.forward(
                    self.nonlinearity.forward(
                        self.logvar_linear_1.forward(x.detach())
                    )
                )
                .tanh()  # to avoid extreme value
                .exp(),
                full=True,  # for interpretability of loss
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )

        return CLUBOutput(
            loss=estimated_mi + cross_entropy_loss,
            estimated_mi=estimated_mi,
            cross_entropy_loss=cross_entropy_loss,
        )
