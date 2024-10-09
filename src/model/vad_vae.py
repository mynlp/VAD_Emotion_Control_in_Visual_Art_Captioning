import itertools
from typing import Any, Literal, Sequence

import loguru
import pydantic
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from torch.nn import Linear
from torch.nn import functional as F
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartForConditionalGeneration,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaConfig,
    RobertaModel,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from ..datamodule.data import LoadedData
from .club import CLUB


class VADVARConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    encoder_model_name: Literal["roberta-base", "roberta-large"]
    encoder_hidden_size: int
    decoder_hidden_size: int

    dim_valence: int
    dim_arousal: int
    dim_dominance: int
    dim_content: int

    kl_weight: float
    mi_weight: float
    info_weight: float

    lr: float
    num_warmup_steps: int
    num_training_steps: int


class VADVAEOutput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )
    loss: Tensor
    reconstruction_loss: Tensor
    info_loss_valence: Tensor
    info_loss_arousal: Tensor
    info_loss_dominance: Tensor
    info_loss_content: Tensor
    mi_va: Tensor
    mi_vd: Tensor
    mi_ad: Tensor
    kl_valence: Tensor
    kl_arousal: Tensor
    kl_dominance: Tensor
    kl_content: Tensor
    z: Tensor
    z_valence: Tensor
    z_arousal: Tensor
    z_dominance: Tensor


class VADVAE(LightningModule):
    def __init__(
        self,
        *,
        encoder_model_name: Literal[
            "roberta-base",
            "roberta-large",
        ],
        decoder_model_name: Literal[
            "facebook/bart-base",
            "facebook/bart-large",
        ],
        dim_valence: int,
        dim_arousal: int,
        dim_dominance: int,
        dim_content: int,
        dim_content_feature: int,
        kl_weight: float,
        mi_weight: float,
        info_weight: float,
        lr: float,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> None:
        super().__init__()
        loguru.logger.info(f"Load {encoder_model_name} as an encoder")
        encoder = RobertaModel.from_pretrained(encoder_model_name)
        assert isinstance(encoder, RobertaModel)
        encoder_config = encoder.config
        assert isinstance(encoder_config, RobertaConfig)
        encoder_hidden_size = encoder_config.hidden_size

        loguru.logger.info(f"Load {decoder_model_name} as an decoder")
        decoder = BartForConditionalGeneration.from_pretrained(
            decoder_model_name
        )
        assert isinstance(decoder, BartForConditionalGeneration)
        decoder_config = decoder.config
        assert isinstance(decoder_config, BartConfig)
        decoder_hidden_size = int(decoder_config.hidden_size)

        self.encoder = encoder
        self.decoder = decoder

        self.loc: dict[Literal["v", "a", "d", "c"], Linear] = {
            "v": Linear(encoder_hidden_size, dim_valence),
            "a": Linear(encoder_hidden_size, dim_arousal),
            "d": Linear(encoder_hidden_size, dim_dominance),
            "c": Linear(encoder_hidden_size, dim_content),
        }
        self.logstd: dict[Literal["v", "a", "d", "c"], Linear] = {
            "v": Linear(encoder_hidden_size, dim_valence),
            "a": Linear(encoder_hidden_size, dim_arousal),
            "d": Linear(encoder_hidden_size, dim_dominance),
            "c": Linear(encoder_hidden_size, dim_content),
        }
        self.info_predictor: dict[Literal["v", "a", "d", "c"], Linear] = {
            "v": Linear(dim_valence, 1),
            "a": Linear(dim_arousal, 1),
            "d": Linear(dim_dominance, 1),
            "c": Linear(dim_content, dim_content_feature),
        }
        self.club: dict[
            tuple[Literal["v", "a", "d", "c"], Literal["v", "a", "d", "c"]],
            CLUB,
        ] = {
            (x, y): CLUB(x_dim, y_dim, max(x_dim, y_dim))
            for (x, x_dim), (y, y_dim) in itertools.combinations(
                iterable=(
                    ("v", dim_valence),
                    ("a", dim_arousal),
                    ("d", dim_dominance),
                    ("c", dim_content),
                ),
                r=2,
            )
        }

        for k, v in self.loc.items():
            self.add_module(f"mu_{k}", v)
        for k, v in self.logstd.items():
            self.add_module(f"logvar_{k}", v)
        for k, v in self.info_predictor.items():
            self.add_module(f"info_predictor_{k}", v)
        for (x, y), v in self.club.items():
            self.add_module(f"logvar_{x}{y}", v)

        self.latent_to_hidden = Linear(
            dim_valence + dim_arousal + dim_dominance + dim_content,
            decoder_hidden_size,
        )

        self.config = VADVARConfig(
            encoder_model_name=encoder_model_name,
            encoder_hidden_size=encoder_hidden_size,
            decoder_hidden_size=decoder_hidden_size,
            dim_valence=dim_valence,
            dim_arousal=dim_arousal,
            dim_dominance=dim_dominance,
            dim_content=dim_content,
            kl_weight=kl_weight,
            mi_weight=mi_weight,
            info_weight=info_weight,
            lr=lr,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def forward(
        self,
        batch: LoadedData,
    ) -> VADVAEOutput:
        batch_size = batch.shape[0]

        encoder_output = self.encoder.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            return_dict=True,
        )
        assert isinstance(
            encoder_output, BaseModelOutputWithPoolingAndCrossAttentions
        ), type(encoder_output)

        encoder_hidden_state = encoder_output.last_hidden_state[:, 0]

        assert encoder_hidden_state.shape == (
            batch_size,
            self.config.encoder_hidden_size,
        )

        loc: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: v.forward(encoder_hidden_state) for k, v in self.loc.items()
        }

        logstd: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: v.forward(encoder_hidden_state) for k, v in self.logstd.items()
        }

        normal: dict[Literal["v", "a", "d", "c"], Normal] = {
            k: Normal(loc[k], logstd[k].exp()) for k in ("v", "a", "d", "c")
        }

        z: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: v.rsample() for k, v in normal.items()
        }

        kldiv: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: kl_divergence(
                v, Normal(torch.zeros_like(v.loc), torch.ones_like(v.scale))
            ).sum(dim=-1)
            for k, v in normal.items()
        }

        club_mi: dict[
            tuple[
                Literal["v", "a", "d", "c"],
                Literal["v", "a", "d", "c"],
            ],
            Tensor,
        ] = {(x, y): v.forward(z[x], z[y]) for (x, y), v in self.club.items()}

        info_loss: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: self.info_predictor[k]
            .forward(z[k])
            .squeeze(-1)
            .sigmoid()
            .sub(score)
            .square()
            .where(mask, 0.0)
            .sum()
            .div(batch.mask_valence.long().sum().clamp_min(1.0))
            for k, score, mask in (
                ("v", batch.score_valence, batch.mask_valence),
                ("a", batch.score_arousal, batch.mask_arousal),
                ("d", batch.score_dominance, batch.mask_dominance),
            )
        }
        info_loss["c"] = (
            self.info_predictor["c"]
            .forward(z["c"])
            .sub(batch.image_feature)
            .square()
            .sum(dim=-1)
            .where(batch.mask_image_feature, 0.0)
            .sum()
            .div(batch.mask_image_feature.long().sum().clamp_min(1.0))
        )

        decoder_output = self.decoder.forward(  # type: ignore
            encoder_outputs=[
                self.latent_to_hidden.forward(  # type: ignore
                    torch.cat(tuple(z.values()), dim=1)
                ).unsqueeze(1)
            ],
            labels=batch.input_ids.where(  # type: ignore
                batch.input_ids.ne(batch.pad_token_id), -100
            ),
            return_dict=True,
        )
        assert isinstance(decoder_output, Seq2SeqLMOutput)
        assert decoder_output.loss is not None

        reconstruction_loss = decoder_output.loss
        assert reconstruction_loss.dim() == 0

        reconstruction_loss = reconstruction_loss.mul(
            batch.attention_mask.long().sum()
        ).div(batch.attention_mask.shape[0])
        # cf. https://github.com/huggingface/transformers/blob/c48787f347bd604f656c2cfff730e029c8f8c1fe/src/transformers/models/bart/modeling_bart.py#L1750

        loss = (
            reconstruction_loss.add(
                torch.stack(tuple(kldiv.values())).sum(dim=0).mean(),
                alpha=self.config.kl_weight,
            )
            .add(
                torch.stack(tuple(club_mi.values())).sum(dim=0).mean(),
                alpha=self.config.mi_weight,
            )
            .add(
                torch.stack(tuple(info_loss.values())).sum(dim=0).mean(),
                alpha=self.config.info_weight,
            )
        )

        return VADVAEOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            info_loss_valence=info_loss["v"],
            info_loss_arousal=info_loss["a"],
            info_loss_dominance=info_loss["d"],
            info_loss_content=info_loss["c"],
            mi_va=club_mi.get(
                ("v", "a"),
                club_mi.get(("a", "v"), torch.as_tensor(float("nan"))),
            ),
            mi_vd=club_mi.get(
                ("v", "d"),
                club_mi.get(("d", "v"), torch.as_tensor(float("nan"))),
            ),
            mi_ad=club_mi.get(
                ("a", "d"),
                club_mi.get(("d", "a"), torch.as_tensor(float("nan"))),
            ),
            kl_valence=kldiv["v"],
            kl_arousal=kldiv["a"],
            kl_dominance=kldiv["d"],
            kl_content=kldiv["c"],
            z=torch.cat(tuple(z.values()), dim=1),
            z_valence=z["v"],
            z_arousal=z["a"],
            z_dominance=z["d"],
        )

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

    @torch.no_grad()
    def generate(
        self,
        *,
        image_feature: Tensor,
        generation_config: GenerationConfig,
        prompt_ids: Sequence[int] | None = None,
        valence: float | None = None,
        arousal: float | None = None,
        dominance: float | None = None,
    ) -> GenerateOutput:
        """
        Method for VAD-conditioned image captioning.

        Parameter
        ---------
        image_feature: Tensor of size (batch_size, feature_dim)
            User-specified image feature.
        generation_config: Generation config
            Configuration object for generation.
        valence: float | None
            User-specified valence score.
        arousal: float | None
            User-specified arousal score.
        dominance: float | None
            User-specified dominance score.

        Return
        ------
        VADVAEGenerationOutput
        """
        assert valence is None or 0 < valence < 1
        assert arousal is None or 0 < arousal < 1
        assert dominance is None or 0 < dominance < 1
        assert image_feature.dim() == 2

        if valence is None:
            valence = 0.5
        if arousal is None:
            arousal = 0.5
        if dominance is None:
            dominance = 0.5

        batch_size = image_feature.shape[0]

        logit: dict[Literal["v", "a", "d"], Tensor] = {
            k: (v.log() - (1 - v).log()).reshape(1, 1).expand(batch_size, 1)
            for k, v in (
                ("v", torch.as_tensor(valence, device=self.device)),
                ("a", torch.as_tensor(arousal, device=self.device)),
                ("d", torch.as_tensor(dominance, device=self.device)),
            )
        }

        z: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: F.linear(
                v.sub(self.info_predictor[k].bias),
                torch.pinverse(self.info_predictor[k].weight),
            )
            for k, v in (
                ("v", logit["v"]),
                ("a", logit["a"]),
                ("d", logit["d"]),
                ("c", image_feature),
            )
        }

        h = self.latent_to_hidden.forward(
            torch.cat(tuple(z[k] for k in ("v", "a", "d", "c")), dim=1)
        ).unsqueeze(1)

        if prompt_ids is None:
            inputs = None
        else:
            inputs = (
                torch.as_tensor(prompt_ids, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        generated = self.decoder.generate(
            inputs=inputs,
            generation_config=generation_config,
            encoder_outputs=BaseModelOutput(last_hidden_state=h),
        )
        assert isinstance(generated, GenerateOutput), generated

        return generated

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
