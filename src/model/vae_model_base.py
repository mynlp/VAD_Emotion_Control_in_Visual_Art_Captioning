import abc
from typing import Any

import pydantic
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput

from ..datamodule.data import LoadedData


class VAEModelBaseConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    lr: float
    num_warmup_steps: int
    num_training_steps: int


class VAEModelBaseOutput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    loss: Tensor


class VAEModelBase(LightningModule, abc.ABC):
    @abc.abstractproperty
    def config(self) -> VAEModelBaseConfig:
        raise NotImplementedError()

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        batch: LoadedData,
    ) -> VAEModelBaseOutput:
        raise NotImplementedError()

    @abc.abstractmethod
    def generate(
        self,
        *,
        image_feature: Tensor,
        generation_config: GenerationConfig,
        valence: float | None = None,
        arousal: float | None = None,
        dominance: float | None = None,
    ) -> tuple[GenerateOutput, Tensor, Tensor, Tensor]:
        raise NotImplementedError()

    def training_step(
        self,
        batch: LoadedData,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        output = self.forward(batch)

        for k, v in output.model_dump().items():
            if isinstance(v, Tensor):
                self.log(
                    f"train/{k}",
                    v.float().mean(),
                    batch_size=batch.shape[0],
                )

        return output.loss.mean()

    def validation_step(
        self,
        batch: LoadedData,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        output = self.forward(batch)

        for k, v in output.model_dump().items():
            if isinstance(v, Tensor):
                self.log(
                    f"val/{k}",
                    v.float().mean(),
                    batch_size=batch.shape[0],
                )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.lr,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_training_steps,
        )

        return [optimizer], [scheduler]
