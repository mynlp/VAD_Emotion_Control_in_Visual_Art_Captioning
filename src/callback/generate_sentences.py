from pathlib import Path
from typing import Sequence

import loguru
import polars as pl
from lightning import Callback, LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers.generation.configuration_utils import (  # type: ignore
    GenerationConfig,
)
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore

from ..datamodule.data import LoadedData
from ..datamodule.datamodule_artemis import ArtemisDataset
from ..model.poe_vad_vae import POEVADVAE


class GenerateSentences(Callback):
    def __init__(
        self,
        dataloader: DataLoader[LoadedData],
        dirpath: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        beam_size: int,
        do_sample: bool,
        top_p: float,
        temperature: float,
        vads: Sequence[tuple[float | None, float | None, float | None]],
        every_n_epoch: int = 10,
    ) -> None:
        super().__init__()

        assert isinstance(dataloader.dataset, ArtemisDataset)

        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.dirpath = dirpath
        self.tokenizer = tokenizer
        self.vads = vads
        self.every_n_epoch = every_n_epoch
        self.epoch = 0

        self.generation_config = GenerationConfig(
            max_length=max_length,
            early_stopping=beam_size > 1,
            num_beams=beam_size,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        if (self.epoch + 1) % self.every_n_epoch == 0:
            self._dump(trainer, pl_module)
        self.epoch += 1

    def _dump(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        loguru.logger.info("Generate sentences")

        assert isinstance(pl_module, POEVADVAE)

        valences_manipulated: list[float | None] = []
        arousals_manipulated: list[float | None] = []
        dominances_manipulated: list[float | None] = []
        valences_predicted: list[float] = []
        arousals_predicted: list[float] = []
        dominances_predicted: list[float] = []

        image_paths: list[str] = []
        sentences: list[str] = []

        for v, a, d in self.vads:
            for batch in self.dataloader:
                assert isinstance(batch, LoadedData)
                assert batch.mask_image_feature.all().item()
                batch = batch.to(pl_module.device)

                generated, v_predicted, a_predicted, d_predicted = (
                    pl_module.generate(
                        image_feature=batch.image_feature,
                        generation_config=self.generation_config,
                        valence=v,
                        arousal=a,
                        dominance=d,
                    )
                )

                valences_manipulated.extend([v] * batch.shape[0])
                arousals_manipulated.extend([a] * batch.shape[0])
                dominances_manipulated.extend([d] * batch.shape[0])
                valences_predicted.extend(v_predicted.tolist())  # type: ignore
                arousals_predicted.extend(a_predicted.tolist())  # type: ignore
                dominances_predicted.extend(
                    d_predicted.tolist()  # type: ignore
                )
                sentences.extend(
                    self.tokenizer.batch_decode(  # type: ignore
                        generated.sequences,
                        skip_special_tokens=True,
                    )
                )
                image_paths.extend(p.as_posix() for p in batch.image_path)

        loguru.logger.info("Save sentences")

        if not self.dirpath.exists():
            self.dirpath.mkdir(parents=True)

        pl.DataFrame(
            {
                "image_path": image_paths,
                "valence_predicted": valences_predicted,
                "arousal_predicted": arousals_predicted,
                "dominance_predicted": dominances_predicted,
                "valence_manipulated": valences_manipulated,
                "arousal_manipulated": arousals_manipulated,
                "dominance_manipulated": dominances_manipulated,
                "sentence": sentences,
                "reference": [
                    str(x)
                    for x in self.dataset.df.get_column("utterance").to_list()
                ]
                * len(self.vads),
            },
        ).sort(  # type: ignore
            "image_path",
        ).write_csv(
            self.dirpath / (f"generate_epoch{self.epoch}.csv")
        )
        loguru.logger.info("Saved")
