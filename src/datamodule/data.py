import itertools
from pathlib import Path
from typing import Any, Sequence

import pydantic
import torch
from torch import Tensor
from typing_extensions import Self


class LoadedData(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    input_ids: Tensor
    attention_mask: Tensor
    score_valence: Tensor
    score_arousal: Tensor
    score_dominance: Tensor
    text_feature: Tensor
    image_feature: Tensor
    mask_valence: Tensor
    mask_arousal: Tensor
    mask_dominance: Tensor
    mask_text_feature: Tensor
    mask_image_feature: Tensor
    pad_token_id: int
    image_path: tuple[Path, ...]

    @classmethod
    def collate_fn(
        cls,
        data: Sequence[Self],
    ) -> Self:
        pad_token_id_set = set(x.pad_token_id for x in data)
        if len(pad_token_id_set) == 1:
            pad_token_id = pad_token_id_set.pop()
        else:
            raise RuntimeError()

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x.input_ids for x in data],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x.attention_mask for x in data],
            batch_first=True,
            padding_value=0,
        )

        score_valence = torch.stack([x.score_valence for x in data])
        score_arousal = torch.stack([x.score_arousal for x in data])
        score_dominance = torch.stack([x.score_dominance for x in data])
        text_feature = torch.stack([x.text_feature for x in data])
        image_feature = torch.stack([x.image_feature for x in data])

        mask_valence = torch.stack([x.mask_valence for x in data])
        mask_arousal = torch.stack([x.mask_arousal for x in data])
        mask_dominance = torch.stack([x.mask_dominance for x in data])
        mask_text_feature = torch.stack([x.mask_text_feature for x in data])
        mask_image_feature = torch.stack([x.mask_image_feature for x in data])

        return cls(
            input_ids=input_ids,
            attention_mask=attention_mask,
            score_valence=score_valence,
            score_arousal=score_arousal,
            score_dominance=score_dominance,
            text_feature=text_feature,
            image_feature=image_feature,
            mask_valence=mask_valence,
            mask_arousal=mask_arousal,
            mask_dominance=mask_dominance,
            mask_text_feature=mask_text_feature,
            mask_image_feature=mask_image_feature,
            pad_token_id=pad_token_id,
            image_path=tuple(
                itertools.chain.from_iterable(x.image_path for x in data)
            ),
        )

    def to(
        self,
        device: torch.device,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        return self.__class__(
            input_ids=self.input_ids.to(device, *args, **kwargs),
            attention_mask=self.attention_mask.to(device, *args, **kwargs),
            score_valence=self.score_valence.to(device, *args, **kwargs),
            score_arousal=self.score_arousal.to(device, *args, **kwargs),
            score_dominance=self.score_dominance.to(device, *args, **kwargs),
            text_feature=self.text_feature.to(device, *args, **kwargs),
            image_feature=self.image_feature.to(device, *args, **kwargs),
            mask_valence=self.mask_valence.to(device, *args, **kwargs),
            mask_arousal=self.mask_arousal.to(device, *args, **kwargs),
            mask_dominance=self.mask_dominance.to(device, *args, **kwargs),
            mask_text_feature=self.mask_text_feature.to(
                device, *args, **kwargs
            ),
            mask_image_feature=self.mask_image_feature.to(
                device, *args, **kwargs
            ),
            pad_token_id=self.pad_token_id,
            image_path=self.image_path,
        )

    @property
    def shape(self):
        return self.input_ids.shape
