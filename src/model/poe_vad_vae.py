import itertools
from typing import Literal, Sequence

import loguru
import torch
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from torch.nn import Linear
from torch.nn import functional as F
from transformers import (  # type: ignore
    BartConfig,
    BartForConditionalGeneration,
    GPT2Config,
    GPT2LMHeadModel,
    T5Config,
    T5ForConditionalGeneration,
)
from transformers.generation.configuration_utils import (  # type: ignore
    GenerationConfig,
)
from transformers.generation.utils import GenerateOutput  # type: ignore
from transformers.modeling_outputs import (  # type: ignore
    BaseModelOutput,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
)

from ..datamodule.data import LoadedData
from .club import CLUB, CLUBOutput
from .transformer_based_adapter import TransformerBasedAdapter
from .vae_model_base import (
    VAEModelBase,
    VAEModelBaseConfig,
    VAEModelBaseOutput,
)


class POEVADVAEConfig(VAEModelBaseConfig):
    decoder_hidden_size: int
    freeze_decoder: bool

    dim_valence: int
    dim_arousal: int
    dim_dominance: int
    dim_content: int

    prefix_length: int

    kl_weight: float
    mi_weight: float
    info_weight: float


class POEVADVAEOutput(VAEModelBaseOutput):
    text_reconstruction_loss: Tensor
    image_reconstruction_loss: Tensor
    info_loss_valence: Tensor
    info_loss_arousal: Tensor
    info_loss_dominance: Tensor
    mi_va: Tensor
    mi_vd: Tensor
    mi_vc: Tensor
    mi_ad: Tensor
    mi_ac: Tensor
    mi_dc: Tensor
    kl_v: Tensor
    kl_a: Tensor
    kl_d: Tensor
    kl_c: Tensor
    z: Tensor
    z_v: Tensor
    z_a: Tensor
    z_d: Tensor
    z_c: Tensor


def _compute_poe_loc(
    *,
    locs: Sequence[Tensor],
    scales: Sequence[Tensor],
    shape: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> Tensor:
    assert len(locs) == len(scales)

    variances = [x.square() for x in scales]

    match len(locs):
        case 0:
            assert shape is not None
            return torch.zeros(shape, device=device)
        case 1:
            return locs[0] / (variances[0] + 1).sqrt()
        case 2:
            return (locs[0] * variances[1] + locs[1] * variances[0]) / (
                (variances[0] + 1) * (variances[1] + 1) - 1
            ).clamp_min(torch.finfo(locs[0].dtype).eps)
        case _:
            raise NotImplementedError()


def _compute_poe_scale(
    *,
    locs: Sequence[Tensor],
    scales: Sequence[Tensor],
    shape: Sequence[int] | None = None,
    device: torch.device | None = None,
) -> Tensor:
    assert len(locs) == len(scales)

    variances = [x.square() for x in scales]

    match len(locs):
        case 0:
            assert shape is not None
            return torch.ones(shape, device=device)
        case 1:
            return scales[0] / (variances[0] + 1).sqrt()
        case 2:
            return (
                scales[0]
                * scales[1]
                / ((variances[0] + 1) * (variances[1] + 1) - 1)
                .clamp_min(torch.finfo(scales[0].dtype).eps)
                .sqrt()
            )
        case _:
            raise NotImplementedError()


class POEVADVAE(VAEModelBase):
    @property
    def config(self):
        return self.__config

    def __init__(
        self,
        *,
        decoder_model_name: Literal[
            "facebook/bart-base",
            "facebook/bart-large",
            "t5-base",
            "t5-large",
            "gpt2",
            "gpt2-large",
        ],
        freeze_decoder: bool,
        dim_valence: int,
        dim_arousal: int,
        dim_dominance: int,
        dim_content: int,
        dim_text_feature: int,
        dim_image_feature: int,
        kl_weight: float,
        mi_weight: float,
        info_weight: float,
        prefix_length: int,
        lr: float,
        num_warmup_steps: int,
        num_training_steps: int,
    ) -> None:
        super().__init__()

        loguru.logger.info(f"Load {decoder_model_name} as an decoder")
        match decoder_model_name:
            case "facebook/bart-base" | "facebook/bart-large":
                decoder = BartForConditionalGeneration.from_pretrained(
                    decoder_model_name,
                )
                assert isinstance(decoder, BartForConditionalGeneration)
                assert isinstance(decoder.config, BartConfig)
            case "t5-base" | "t5-large":
                decoder = T5ForConditionalGeneration.from_pretrained(
                    decoder_model_name,
                )
                assert isinstance(decoder, T5ForConditionalGeneration)
                assert isinstance(decoder.config, T5Config)
            case "gpt2" | "gpt2-large":
                decoder = GPT2LMHeadModel.from_pretrained(
                    decoder_model_name,
                )
                assert isinstance(decoder, GPT2LMHeadModel)
                assert isinstance(decoder.config, GPT2Config)

        if freeze_decoder:
            for param in decoder.parameters():
                param.requires_grad = False

        decoder_hidden_size = int(decoder.config.hidden_size)

        self.decoder = decoder
        self.image_feature_decoder_loc = Linear(
            dim_valence + dim_arousal + dim_dominance + dim_content,
            dim_image_feature,
        )
        self.image_feature_decoder_logscale = Linear(
            dim_valence + dim_arousal + dim_dominance + dim_content,
            dim_image_feature,
        )

        self.adapter = TransformerBasedAdapter(
            input_size=dim_valence + dim_arousal + dim_dominance + dim_content,
            prefix_length=prefix_length,
            d_model=decoder_hidden_size,
        )

        self.text_feature_to_loc: dict[Literal["v", "a", "d", "c"], Linear] = {
            "v": Linear(dim_text_feature, dim_valence),
            "a": Linear(dim_text_feature, dim_arousal),
            "d": Linear(dim_text_feature, dim_dominance),
            "c": Linear(dim_text_feature, dim_content),
        }
        self.text_feature_to_logscale: dict[
            Literal["v", "a", "d", "c"], Linear
        ] = {
            "v": Linear(dim_text_feature, dim_valence),
            "a": Linear(dim_text_feature, dim_arousal),
            "d": Linear(dim_text_feature, dim_dominance),
            "c": Linear(dim_text_feature, dim_content),
        }
        self.image_feature_to_loc: dict[
            Literal["v", "a", "d", "c"], Linear
        ] = {
            "v": Linear(dim_image_feature, dim_valence),
            "a": Linear(dim_image_feature, dim_arousal),
            "d": Linear(dim_image_feature, dim_dominance),
            "c": Linear(dim_image_feature, dim_content),
        }
        self.image_feature_to_logscale: dict[
            Literal["v", "a", "d", "c"], Linear
        ] = {
            "v": Linear(dim_image_feature, dim_valence),
            "a": Linear(dim_image_feature, dim_arousal),
            "d": Linear(dim_image_feature, dim_dominance),
            "c": Linear(dim_image_feature, dim_content),
        }
        self.info_predictor: dict[Literal["v", "a", "d"], Linear] = {
            "v": Linear(dim_valence, 1),
            "a": Linear(dim_arousal, 1),
            "d": Linear(dim_dominance, 1),
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

        for k, v in self.text_feature_to_loc.items():
            self.add_module(f"text_feature_to_loc_{k}", v)
        for k, v in self.text_feature_to_logscale.items():
            self.add_module(f"text_feature_to_logscale_{k}", v)
        for k, v in self.image_feature_to_loc.items():
            self.add_module(f"image_feature_to_loc_{k}", v)
        for k, v in self.image_feature_to_logscale.items():
            self.add_module(f"image_feature_to_logscale_{k}", v)
        for k, v in self.info_predictor.items():
            self.add_module(f"info_predictor_{k}", v)
        for (x, y), v in self.club.items():
            self.add_module(f"logvar_{x}{y}", v)

        self.__config = POEVADVAEConfig(
            decoder_hidden_size=decoder_hidden_size,
            freeze_decoder=freeze_decoder,
            dim_valence=dim_valence,
            dim_arousal=dim_arousal,
            dim_dominance=dim_dominance,
            dim_content=dim_content,
            prefix_length=prefix_length,
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
    ) -> POEVADVAEOutput:
        # v: valence
        # a: arousal
        # d: dominance
        # c: content

        latents: dict[Literal["v", "a", "d", "c"], Tensor] = {}
        kl_divs: dict[Literal["v", "a", "d", "c"], Tensor] = {}

        text_is_given = batch.mask_text_feature
        image_is_given = batch.mask_image_feature
        both_text_and_image_are_given = text_is_given.logical_and(
            image_is_given
        )

        for latent_block_name in ("v", "a", "d", "c"):
            text_loc = self.text_feature_to_loc[latent_block_name].forward(
                F.layer_norm(
                    batch.text_feature,
                    (batch.text_feature.shape[-1],),
                )
            )
            text_scale = (
                self.text_feature_to_logscale[latent_block_name]
                .forward(
                    F.layer_norm(
                        batch.text_feature,
                        (batch.text_feature.shape[-1],),
                    )
                )
                .exp()
            )

            image_loc = self.image_feature_to_loc[latent_block_name].forward(
                F.layer_norm(
                    batch.image_feature,
                    (batch.image_feature.shape[-1],),
                )
            )
            image_scale = (
                self.image_feature_to_logscale[latent_block_name]
                .forward(
                    F.layer_norm(
                        batch.image_feature,
                        (batch.image_feature.shape[-1],),
                    )
                )
                .exp()
            )

            assert (
                text_loc.shape
                == text_scale.shape
                == image_loc.shape
                == image_scale.shape
            )

            loc = torch.where(
                both_text_and_image_are_given.unsqueeze(-1).expand_as(
                    text_loc
                ),
                _compute_poe_loc(
                    locs=(text_loc, image_loc),
                    scales=(text_scale, image_scale),
                ),
                torch.where(
                    text_is_given.unsqueeze(-1).expand_as(text_loc),
                    _compute_poe_loc(
                        locs=(text_loc,),
                        scales=(text_scale,),
                    ),
                    torch.where(
                        image_is_given.unsqueeze(-1).expand_as(image_loc),
                        _compute_poe_loc(
                            locs=(image_loc,),
                            scales=(image_scale,),
                        ),
                        _compute_poe_loc(
                            locs=(),
                            scales=(),
                            shape=image_loc.shape,
                            device=self.device,
                        ),
                    ),
                ),
            )

            scale = torch.where(
                both_text_and_image_are_given.unsqueeze(-1).expand_as(
                    text_loc
                ),
                _compute_poe_scale(
                    locs=(text_loc, image_loc),
                    scales=(text_scale, image_scale),
                ),
                torch.where(
                    text_is_given.unsqueeze(-1).expand_as(text_loc),
                    _compute_poe_scale(
                        locs=(text_loc,),
                        scales=(text_scale,),
                    ),
                    torch.where(
                        image_is_given.unsqueeze(-1).expand_as(image_loc),
                        _compute_poe_scale(
                            locs=(image_loc,),
                            scales=(image_scale,),
                        ),
                        _compute_poe_scale(
                            locs=(),
                            scales=(),
                            shape=image_loc.shape,
                            device=self.device,
                        ),
                    ),
                ),
            )

            variational_posterior = Normal(loc, scale)
            sample = variational_posterior.rsample()  # type: ignore
            assert isinstance(sample, Tensor)
            latents[latent_block_name] = sample
            kl_divs[latent_block_name] = kl_divergence(
                variational_posterior,
                Normal(torch.zeros_like(loc), torch.ones_like(scale)),
            ).sum(dim=-1)

        club_mis: dict[
            tuple[
                Literal["v", "a", "d", "c"],
                Literal["v", "a", "d", "c"],
            ],
            CLUBOutput,
        ] = {
            (x, y): v.forward(latents[x], latents[y])
            for (x, y), v in self.club.items()
        }

        info_loss: dict[Literal["v", "a", "d"], Tensor] = {
            k: torch.where(
                mask,
                self.info_predictor[k]
                .forward(latents[k])
                .squeeze(-1)
                .sub(score.logit(torch.finfo(torch.float).eps))
                .square(),
                0.0,
            )
            .sum()
            .div(mask.long().sum().clamp_min(1.0))
            for k, score, mask in (
                ("v", batch.score_valence, batch.mask_valence),
                ("a", batch.score_arousal, batch.mask_arousal),
                ("d", batch.score_dominance, batch.mask_dominance),
            )
        }

        concatenated_latent = torch.cat(
            [latents[k] for k in ("v", "a", "d", "c")], dim=1
        )

        match self.decoder:
            case (
                BartForConditionalGeneration()
                | T5ForConditionalGeneration() as seq2seq
            ):
                decoder_output = seq2seq.forward(  # type: ignore
                    encoder_outputs=[
                        self.adapter.forward(  # type: ignore
                            concatenated_latent,
                        )
                    ],
                    labels=torch.where(  # type: ignore
                        batch.attention_mask.bool(), batch.input_ids, -100
                    ),
                    return_dict=True,
                )
                assert isinstance(decoder_output, Seq2SeqLMOutput)
                assert decoder_output.loss is not None
            case GPT2LMHeadModel() as gpt2:
                inputs_embeds = torch.cat(
                    [
                        self.adapter.forward(concatenated_latent),
                        gpt2.resize_token_embeddings().forward(
                            batch.input_ids
                        ),
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        torch.ones(
                            size=(
                                batch.input_ids.shape[0],
                                self.config.prefix_length,
                            ),
                            device=batch.input_ids.device,
                        ),
                        batch.attention_mask,
                    ],
                    dim=1,
                )
                labels = torch.cat(
                    [
                        torch.full(
                            size=(
                                batch.input_ids.shape[0],
                                self.config.prefix_length,
                            ),
                            fill_value=-100,
                            device=batch.input_ids.device,
                        ),
                        torch.where(
                            batch.attention_mask.bool(),
                            batch.input_ids,
                            -100,
                        ),
                    ],
                    dim=1,
                )

                assert (
                    inputs_embeds.shape[:2]
                    == attention_mask.shape
                    == labels.shape
                ), (
                    inputs_embeds.shape,
                    attention_mask.shape,
                    labels.shape,
                )

                decoder_output = gpt2.forward(  # type: ignore
                    inputs_embeds=inputs_embeds,  # type: ignore
                    attention_mask=attention_mask,  # type: ignore
                    labels=labels,  # type: ignore
                    return_dict=True,
                )
                assert isinstance(
                    decoder_output,
                    CausalLMOutputWithCrossAttentions,
                )
                assert decoder_output.loss is not None

        text_reconstruction_loss = (
            decoder_output.loss.nan_to_num()
            * batch.attention_mask.long().sum()
            / max(batch.attention_mask.shape[0], 1)
        )
        assert text_reconstruction_loss.dim() == 0

        image_reconstruction_loss = torch.where(
            batch.mask_image_feature,
            F.gaussian_nll_loss(
                input=self.image_feature_decoder_loc.forward(
                    concatenated_latent
                ),
                target=batch.image_feature,
                var=self.image_feature_decoder_logscale.forward(
                    concatenated_latent
                )
                .mul(2)
                .exp(),
                full=True,  # for interpretability of loss.
                reduction="none",
            ).sum(dim=-1),
            0.0,
        ).sum() / batch.mask_image_feature.long().sum().clamp_min(1.0)

        reconstruction_loss = (
            text_reconstruction_loss + image_reconstruction_loss
        )

        loss = (
            reconstruction_loss.add(
                torch.stack(tuple(kl_divs.values())).sum(dim=0).mean(),
                alpha=self.config.kl_weight,
            )
            .add(
                torch.stack([x.loss for x in club_mis.values()])
                .sum(dim=0)
                .mean(),
                alpha=self.config.mi_weight,
            )
            .add(
                torch.stack(tuple(info_loss.values())).sum(dim=0).mean(),
                alpha=self.config.info_weight,
            )
        )

        return POEVADVAEOutput(
            loss=loss,
            text_reconstruction_loss=text_reconstruction_loss,
            image_reconstruction_loss=image_reconstruction_loss,
            info_loss_valence=info_loss["v"],
            info_loss_arousal=info_loss["a"],
            info_loss_dominance=info_loss["d"],
            mi_va=(
                club_mis["v", "a"]
                if ("v", "a") in club_mis
                else club_mis["a", "v"]
            ).estimated_mi,
            mi_vd=(
                club_mis["v", "d"]
                if ("v", "d") in club_mis
                else club_mis["d", "v"]
            ).estimated_mi,
            mi_vc=(
                club_mis["v", "c"]
                if ("v", "c") in club_mis
                else club_mis["c", "v"]
            ).estimated_mi,
            mi_ad=(
                club_mis["a", "d"]
                if ("a", "d") in club_mis
                else club_mis["d", "a"]
            ).estimated_mi,
            mi_ac=(
                club_mis["a", "c"]
                if ("a", "c") in club_mis
                else club_mis["a", "c"]
            ).estimated_mi,
            mi_dc=(
                club_mis["d", "c"]
                if ("d", "c") in club_mis
                else club_mis["c", "d"]
            ).estimated_mi,
            kl_v=kl_divs["v"],
            kl_a=kl_divs["a"],
            kl_d=kl_divs["d"],
            kl_c=kl_divs["c"],
            z=torch.cat(tuple(latents.values()), dim=1),
            z_v=latents["v"],
            z_a=latents["a"],
            z_d=latents["d"],
            z_c=latents["c"],
        )

    @torch.no_grad()  # type: ignore
    def generate(
        self,
        *,
        image_feature: Tensor,
        generation_config: GenerationConfig,
        valence: float | None = None,
        arousal: float | None = None,
        dominance: float | None = None,
    ) -> tuple[GenerateOutput, Tensor, Tensor, Tensor]:
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
        generate_output: GenerateOutput
        v_predicted: Tensor
        a_predicted: Tensor
        d_predicted: Tensor
        """
        assert valence is None or 0 < valence < 1
        assert arousal is None or 0 < arousal < 1
        assert dominance is None or 0 < dominance < 1
        assert image_feature.dim() == 2

        loc: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: v.forward(image_feature)
            for k, v in self.image_feature_to_loc.items()
        }
        scale: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: v.forward(image_feature).exp()
            for k, v in self.image_feature_to_logscale.items()
        }

        z: dict[Literal["v", "a", "d", "c"], Tensor] = {
            k: loc[k] / (scale[k].square() + 1).sqrt()
            for k in ("v", "a", "d", "c")
        }

        v_predicted = (
            self.info_predictor["v"].forward(z["v"]).squeeze(-1).sigmoid()
        )
        a_predicted = (
            self.info_predictor["a"].forward(z["a"]).squeeze(-1).sigmoid()
        )
        d_predicted = (
            self.info_predictor["d"].forward(z["d"]).squeeze(-1).sigmoid()
        )

        batch_size = image_feature.shape[0]

        if valence is not None:
            v_manipulated = torch.as_tensor(valence, device=self.device)
            z["v"] = F.linear(
                (torch.logit(v_manipulated) - self.info_predictor["v"].bias)
                .unsqueeze(0)
                .expand(batch_size, 1),
                torch.pinverse(self.info_predictor["v"].weight),
            )
        if arousal is not None:
            a_manipulated = torch.as_tensor(arousal, device=self.device)
            z["a"] = F.linear(
                (torch.logit(a_manipulated) - self.info_predictor["a"].bias)
                .unsqueeze(0)
                .expand(batch_size, 1),
                torch.pinverse(self.info_predictor["a"].weight),
            )
        if dominance is not None:
            d_manipulated = torch.as_tensor(dominance, device=self.device)
            z["d"] = F.linear(
                (torch.logit(d_manipulated) - self.info_predictor["d"].bias)
                .unsqueeze(0)
                .expand(batch_size, 1),
                torch.pinverse(self.info_predictor["d"].weight),
            )

        h = self.adapter.forward(
            torch.cat([z[k] for k in ("v", "a", "d", "c")], dim=1)
        )
        assert h.shape == (
            batch_size,
            self.config.prefix_length,
            self.config.decoder_hidden_size,
        )

        match self.decoder:
            case (
                BartForConditionalGeneration()
                | T5ForConditionalGeneration() as seq2seq
            ):
                generate_output = seq2seq.generate(
                    generation_config=generation_config,
                    encoder_outputs=BaseModelOutput(last_hidden_state=h),
                )
            case GPT2LMHeadModel() as gpt2:
                generate_output = gpt2.generate(
                    generation_config=generation_config,
                    inputs_embeds=h,
                )
        assert isinstance(generate_output, GenerateOutput)

        return generate_output, v_predicted, a_predicted, d_predicted
