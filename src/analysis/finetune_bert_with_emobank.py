import datetime
import typing
from pathlib import Path
from typing import Literal

import polars as pl
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from tap import Tap
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer


def collate_fn(
    data: typing.Sequence[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [x[0] for x in data], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x[1] for x in data], batch_first=True, padding_value=0
    )
    v = torch.stack([x[2] for x in data])
    a = torch.stack([x[3] for x in data])
    d = torch.stack([x[4] for x in data])
    return input_ids, attention_mask, v, a, d


class EmoBankDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        df: pl.DataFrame,
    ) -> None:
        super().__init__()
        self.df = df

    def __getitem__(
        self,
        index: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        named_row = self.df.row(index, named=True)
        return (
            torch.as_tensor(named_row["input_ids"]),
            torch.as_tensor(named_row["attention_mask"]),
            torch.as_tensor(named_row["V"]),
            torch.as_tensor(named_row["A"]),
            torch.as_tensor(named_row["D"]),
        )

    def __len__(self):
        return len(self.df)


class EmoBankDataModule(LightningDataModule):
    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(
        self,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        return DataLoader(
            EmoBankDataset(self.df.filter(pl.col("split") == "train")),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
        )

    def val_dataloader(
        self,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        return DataLoader(
            EmoBankDataset(self.df.filter(pl.col("split") == "dev")),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(
        self,
    ) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        return DataLoader(
            EmoBankDataset(self.df.filter(pl.col("split") == "test")),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


class BERTBasedEmotionRegression(LightningModule):
    def __init__(
        self,
        bert_model: BertModel,
        target_emotion_dimension: Literal["V", "A", "D"],
        lr: float,
    ) -> None:
        super().__init__()
        self.bert_model = bert_model
        self.fc = torch.nn.Linear(
            typing.cast(BertConfig, bert_model.config).hidden_size,
            1,
        )
        self.target_emotion_dimension: Literal["V", "A", "D"] = (
            target_emotion_dimension
        )
        self.lr = lr

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        bert_output = self.bert_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        assert isinstance(
            bert_output, BaseModelOutputWithPoolingAndCrossAttentions
        )
        prediction = self.fc.forward(
            bert_output.last_hidden_state[:, 0]
        ).squeeze(-1)
        return prediction

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> Tensor:
        input_ids, attention_mask, v, a, d = batch
        prediction = self.forward(input_ids, attention_mask)
        loss = (
            (
                prediction
                - {"V": v, "A": a, "D": d}[self.target_emotion_dimension]
            )
            .square()
            .mean()
        )
        self.log("trian/loss", loss, batch_size=input_ids.shape[0])
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> None:
        input_ids, attention_mask, v, a, d = batch
        prediction = self.forward(input_ids, attention_mask)
        loss = (
            (
                prediction
                - {"V": v, "A": a, "D": d}[self.target_emotion_dimension]
            )
            .square()
            .mean()
        )
        self.log("val/loss", loss, batch_size=input_ids.shape[0])

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return AdamW(self.parameters(), lr=self.lr)


class ArgumentParser(Tap):
    bert_model_name: Literal["bert-base-uncased"] = "bert-base-uncased"
    emobank_path: Path = Path("EmoBank") / "corpus" / "emobank.csv"
    target_emotion_dimension: Literal["V", "A", "D"] = "V"
    batch_size: int = 128
    lr: float = 1e-5
    num_workers: int = 2

    save_dir: Path = Path("save_dir")
    name: str = "bert-emobank"
    version: str = datetime.datetime.now().isoformat()

    random_seed: int = 43


def main():
    args = ArgumentParser(explicit_bool=True).parse_args()

    seed_everything(args.random_seed)

    bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        args.bert_model_name
    )
    assert isinstance(bert_tokenizer, BertTokenizer)

    bert_model: BertModel = BertModel.from_pretrained(args.bert_model_name)
    assert isinstance(bert_model, BertModel)

    emobank_df = pl.read_csv(args.emobank_path)
    batch_encoding = bert_tokenizer(
        emobank_df.get_column("text").to_list(),
        add_special_tokens=True,
        padding=False,
        truncation=True,
    )
    emobank_df = emobank_df.with_columns(  # type: ignore
        pl.Series("input_ids", batch_encoding["input_ids"]),
        pl.Series("attention_mask", batch_encoding["attention_mask"]),
    )

    datamodule = EmoBankDataModule(
        df=emobank_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = BERTBasedEmotionRegression(
        bert_model=bert_model,
        target_emotion_dimension=args.target_emotion_dimension,
        lr=args.lr,
    )

    Trainer(
        logger=CSVLogger(
            save_dir=args.save_dir,
            name=args.name,
            version=args.version,
        ),
        callbacks=[
            ModelCheckpoint(
                filename=f"bert-emobank-{args.target_emotion_dimension}",
                monitor="val/loss",
                mode="min",
            )
        ],
    ).fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
