import random
import re
from pathlib import Path
from typing import Literal, Sequence, cast

import loguru
import polars as pl
import torch
import tqdm
from lightning.pytorch import LightningDataModule
from PIL import Image
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer  # type: ignore

from .data import LoadedData

ART_STYLE = "art_style"
PAINTING = "painting"
EMOTION = "emotion"
UTTERANCE = "utterance"

IMAGE_PATH = "image_path"
SPLIT = "split"

INPUT_IDS = "input_ids"
ATTENTION_MASK = "attention_mask"
TEXT_FEATURE = "text_feature"
IMAGE_FEATURE = "image_feature"

VALENCE = "Valence"
AROUSAL = "Arousal"
DOMINANCE = "Dominance"

CATEGORY_TO_VALENCE = {
    "amusement": 0.929,
    "anger": 0.167,
    "awe": 0.469,
    "contentment": 0.875,
    "disgust": 0.052,
    "excitement": 0.896,
    "fear": 0.073,
    "sadness": 0.052,
}
CATEGORY_TO_AROUSAL = {
    "amusement": 0.837,
    "anger": 0.865,
    "awe": 0.740,
    "contentment": 0.610,
    "disgust": 0.775,
    "excitement": 0.684,
    "fear": 0.840,
    "sadness": 0.288,
}
CATEGORY_TO_DOMINANCE = {
    "amusement": 0.803,
    "anger": 0.657,
    "awe": 0.300,
    "contentment": 0.782,
    "disgust": 0.317,
    "excitement": 0.731,
    "fear": 0.293,
    "sadness": 0.164,
}


class ArtemisDataset(Dataset[LoadedData]):
    def __init__(
        self,
        df: pl.DataFrame,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        self.__df = df
        self.__pad_token_id = pad_token_id

    @property
    def df(self):
        return self.__df

    def __getitem__(
        self,
        index: int,
    ) -> LoadedData:
        row = self.__df.row(index, named=True)

        true_as_tensor = torch.as_tensor(True)

        return LoadedData(
            input_ids=torch.as_tensor(row[INPUT_IDS]),
            attention_mask=torch.as_tensor(row[ATTENTION_MASK]),
            score_valence=torch.as_tensor(row[VALENCE]),
            score_arousal=torch.as_tensor(row[AROUSAL]),
            score_dominance=torch.as_tensor(row[DOMINANCE]),
            text_feature=torch.as_tensor(row[TEXT_FEATURE]),
            image_feature=torch.as_tensor(row[IMAGE_FEATURE]),
            mask_valence=torch.as_tensor(row[VALENCE] != -100),
            mask_arousal=torch.as_tensor(row[AROUSAL] != -100),
            mask_dominance=torch.as_tensor(row[DOMINANCE] != -100),
            mask_text_feature=true_as_tensor,
            mask_image_feature=true_as_tensor,
            pad_token_id=self.__pad_token_id,
            image_path=(Path(row[IMAGE_PATH]),),
        )

    def __len__(self):
        return len(self.__df)


class NRCVADDataset(Dataset[LoadedData]):
    def __init__(
        self,
        df: pl.DataFrame,
        pad_token_id: int,
        dim_image_feature: int,
    ) -> None:
        super().__init__()

        self.__df = df
        self.__pad_token_id = pad_token_id
        self.__dim_image_feature = dim_image_feature

    @property
    def df(self):
        return self.__df

    def __getitem__(
        self,
        index: int,
    ) -> LoadedData:
        row = self.__df.row(index, named=True)

        return LoadedData(
            input_ids=torch.as_tensor(row[INPUT_IDS]),
            attention_mask=torch.as_tensor(row[ATTENTION_MASK]),
            score_valence=torch.as_tensor(row[VALENCE]),
            score_arousal=torch.as_tensor(row[AROUSAL]),
            score_dominance=torch.as_tensor(row[DOMINANCE]),
            text_feature=torch.as_tensor(row[TEXT_FEATURE]),
            image_feature=torch.full(
                size=(self.__dim_image_feature,),
                fill_value=-100.0,
            ),
            mask_valence=torch.as_tensor(row[VALENCE] != -100),
            mask_arousal=torch.as_tensor(row[AROUSAL] != -100),
            mask_dominance=torch.as_tensor(row[DOMINANCE] != -100),
            mask_text_feature=torch.as_tensor(True),
            mask_image_feature=torch.as_tensor(False),
            pad_token_id=self.__pad_token_id,
            image_path=(Path("."),),
        )

    def __len__(self):
        return len(self.__df)


class DViSADataset(Dataset[LoadedData]):
    def __init__(
        self,
        df: pl.DataFrame,
        pad_token_id: int,
        dim_text_feature: int,
    ) -> None:
        super().__init__()

        self.__df = df
        self.__pad_token_id = pad_token_id
        self.__dim_text_feature = dim_text_feature

    @property
    def df(self):
        return self.__df

    def __getitem__(
        self,
        index: int,
    ) -> LoadedData:
        row = self.__df.row(index, named=True)

        return LoadedData(
            input_ids=torch.as_tensor([self.__pad_token_id], dtype=torch.long),
            attention_mask=torch.as_tensor([0], dtype=torch.long),
            score_valence=torch.as_tensor(row[VALENCE]),
            score_arousal=torch.as_tensor(row[AROUSAL]),
            score_dominance=torch.as_tensor(row[DOMINANCE]),
            text_feature=torch.full(
                size=(self.__dim_text_feature,),
                fill_value=-100.0,
            ),
            image_feature=torch.as_tensor(row[IMAGE_FEATURE]),
            mask_valence=torch.as_tensor(row[VALENCE] != -100),
            mask_arousal=torch.as_tensor(row[AROUSAL] != -100),
            mask_dominance=torch.as_tensor(row[DOMINANCE] != -100),
            mask_text_feature=torch.as_tensor(False),
            mask_image_feature=torch.as_tensor(True),
            pad_token_id=self.__pad_token_id,
            image_path=(Path("."),),
        )

    def __len__(self):
        return len(self.__df)


class ConcatDataset(Dataset[LoadedData]):
    def __init__(
        self,
        *datasets: ArtemisDataset | NRCVADDataset | DViSADataset,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.lengths = tuple(len(d) for d in datasets)

    def __getitem__(
        self,
        index: int,
    ) -> LoadedData:
        dataset_index = 0
        datapoint_index = index
        for length in self.lengths:
            if datapoint_index < length:
                break
            else:
                datapoint_index -= length
                dataset_index += 1

        return self.datasets[dataset_index][datapoint_index]

    def __len__(self):
        return sum(self.lengths)


def _check_and_modify_path(path: str | Path):
    path = Path(path)
    if not path.exists():
        loguru.logger.warning(f"Image file {path} not found.")
        possible_paths = tuple(
            path.parent.glob(
                re.sub(
                    r"\*+",
                    "*",
                    "".join(c if c.isascii() else "*" for c in path.name),
                )
            )
        )
        assert len(possible_paths) > 0, path
        loguru.logger.warning(
            f"Perhaps it is {possible_paths}? "
            f"Regard {possible_paths[0]} as a correct path."
        )
        path = possible_paths[0]
    return path.as_posix()


class ArtemisDataModule(LightningDataModule):
    __artemis_df: pl.DataFrame | None = None
    __nrc_vad_df: pl.DataFrame | None = None
    __dvisa_df: pl.DataFrame | None = None

    __artemis_dataset_creation_completed: bool = False
    __nrc_vad_dataset_creation_completed: bool = False
    __dvisa_dataset_creation_completed: bool = False

    @property
    def pad_token_id(self):
        assert self.__artemis_dataset_creation_completed
        assert self.__nrc_vad_dataset_creation_completed
        assert self.__dvisa_dataset_creation_completed
        assert isinstance(self.tokenizer.pad_token_id, int)
        return self.tokenizer.pad_token_id

    @property
    def dim_text_feature(self):
        assert self.__artemis_dataset_creation_completed
        assert self.__nrc_vad_dataset_creation_completed
        assert self.__dvisa_dataset_creation_completed
        assert self.__artemis_df is not None
        return len(self.__artemis_df.row(0, named=True)[TEXT_FEATURE])

    @property
    def dim_image_feature(self):
        assert self.__artemis_dataset_creation_completed
        assert self.__nrc_vad_dataset_creation_completed
        assert self.__dvisa_dataset_creation_completed
        assert self.__artemis_df is not None
        return len(self.__artemis_df.row(0, named=True)[IMAGE_FEATURE])

    @property
    def num_steps_per_epoch(self):
        assert self.__artemis_dataset_creation_completed
        assert self.__nrc_vad_dataset_creation_completed
        assert self.__dvisa_dataset_creation_completed
        assert self.__artemis_df is not None

        train_dataset_size = len(
            self.__artemis_df.filter(pl.col(SPLIT) == "train")
        )
        if self.__nrc_vad_df is not None:
            train_dataset_size += len(
                self.__nrc_vad_df.filter(pl.col(SPLIT) == "train")
            )
        if self.__dvisa_df is not None:
            train_dataset_size += len(
                self.__dvisa_df.filter(pl.col(SPLIT) == "train")
            )

        return train_dataset_size // self.batch_size

    @property
    def artemis_data_prompt_ids(self) -> list[int] | None:
        if self.artemis_data_prompt == "":
            return None
        return self.tokenizer.encode(self.artemis_data_prompt)

    def __init__(
        self,
        *,
        tokenizer: PreTrainedTokenizer,
        artemis_data_path: Path,
        artemis_data_sep: Literal["\t", ","],
        use_artemis_categorical_emotions: bool,
        nrc_vad_lexicon_data_path: Path,
        nrc_vad_lexicon_data_sep: Literal["\t", ","],
        dvisa_data_path: Path | None,
        dvisa_data_sep: Literal["\t", ","],
        wikiart_dirpath: Path,
        batch_size: int,
        num_workers: int,
        random_seed: int | None,
        data_split_strategy: Literal["none", "image"],
        val_ratio: float,
        test_ratio: float,
        image_model_name: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ],
        batch_size_for_clip: int,
        batch_size_for_generation: int,
        debug_mode: bool,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.artemis_data_path = artemis_data_path
        self.artemis_data_sep = artemis_data_sep
        self.use_artemis_categorical_emotions = (
            use_artemis_categorical_emotions
        )

        self.nrc_vad_lexicon_data_path = nrc_vad_lexicon_data_path
        self.nrc_vad_lexicon_data_sep = nrc_vad_lexicon_data_sep

        self.dvisa_data_path = dvisa_data_path
        self.dvisa_data_sep = dvisa_data_sep

        self.wikiart_dirpath = wikiart_dirpath

        self.image_model_name: Literal[
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-336",
        ] = image_model_name

        self.batch_size_for_clip = batch_size_for_clip
        self.batch_size_for_generation = batch_size_for_generation
        self.random_seed = random_seed

        self.data_split_strategy: Literal[
            "none",
            "image",
        ] = data_split_strategy
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.debug_mode = debug_mode

    def setup(self, stage: str | None = None) -> None:
        if not self.__artemis_dataset_creation_completed:
            self.__create_artemis_datasets()
        if not self.__nrc_vad_dataset_creation_completed:
            self.__create_nrc_vad_datasets()
        if not self.__dvisa_dataset_creation_completed:
            self.__create_dvisa_datasets()

    def train_dataloader(self) -> DataLoader[LoadedData]:
        assert self.__artemis_df is not None

        datasets: list[ArtemisDataset | NRCVADDataset | DViSADataset] = [
            ArtemisDataset(
                self.__artemis_df.filter(pl.col(SPLIT) == "train"),
                self.pad_token_id,
            )
        ]
        if self.__nrc_vad_df is not None:
            datasets.append(
                NRCVADDataset(
                    self.__nrc_vad_df.filter(pl.col(SPLIT) == "train"),
                    self.pad_token_id,
                    self.dim_image_feature,
                )
            )
        if self.__dvisa_df is not None:
            datasets.append(
                DViSADataset(
                    self.__dvisa_df.filter(pl.col(SPLIT) == "train"),
                    self.pad_token_id,
                    self.dim_text_feature,
                )
            )

        return DataLoader(
            dataset=ConcatDataset(*datasets),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=LoadedData.collate_fn,
            shuffle=True,
        )

    def val_dataloader(
        self,
    ) -> tuple[DataLoader[LoadedData], ...]:
        assert self.__artemis_df is not None
        dataloaders: list[DataLoader[LoadedData]] = [
            DataLoader(
                dataset=ArtemisDataset(
                    self.__artemis_df.filter(pl.col(SPLIT) == "val"),
                    self.pad_token_id,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=LoadedData.collate_fn,
            )
        ]
        if self.__nrc_vad_df is not None:
            dataloaders.append(
                DataLoader(
                    dataset=NRCVADDataset(
                        self.__nrc_vad_df.filter(pl.col(SPLIT) == "val"),
                        self.pad_token_id,
                        self.dim_image_feature,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=LoadedData.collate_fn,
                )
            )
        if self.__dvisa_df is not None:
            dataloaders.append(
                DataLoader(
                    dataset=DViSADataset(
                        self.__dvisa_df.filter(pl.col(SPLIT) == "val"),
                        self.pad_token_id,
                        self.dim_text_feature,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=LoadedData.collate_fn,
                )
            )

        return tuple(dataloaders)

    def test_dataloader(
        self,
    ) -> tuple[DataLoader[LoadedData], ...]:
        assert self.__artemis_df is not None
        dataloaders: list[DataLoader[LoadedData]] = [
            DataLoader(
                dataset=ArtemisDataset(
                    self.__artemis_df.filter(pl.col(SPLIT) == "test"),
                    self.pad_token_id,
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=LoadedData.collate_fn,
            )
        ]
        if self.__nrc_vad_df is not None:
            dataloaders.append(
                DataLoader(
                    dataset=NRCVADDataset(
                        self.__nrc_vad_df.filter(pl.col(SPLIT) == "test"),
                        self.pad_token_id,
                        self.dim_image_feature,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=LoadedData.collate_fn,
                )
            )
        if self.__dvisa_df is not None:
            dataloaders.append(
                DataLoader(
                    dataset=DViSADataset(
                        self.__dvisa_df.filter(pl.col(SPLIT) == "test"),
                        self.pad_token_id,
                        self.dim_text_feature,
                    ),
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=LoadedData.collate_fn,
                )
            )
        return tuple(dataloaders)

    def val_dataloader_for_image2text_generation(
        self,
    ) -> DataLoader[LoadedData]:
        assert self.__artemis_df is not None
        return DataLoader(
            dataset=ArtemisDataset(
                self.__artemis_df.filter(pl.col(SPLIT) == "val")
                .group_by(IMAGE_PATH)  # type: ignore
                .agg(
                    pl.col(UTTERANCE),
                    pl.col(INPUT_IDS).first(),
                    pl.col(ATTENTION_MASK).first(),
                    pl.col(TEXT_FEATURE).first(),
                    pl.col(IMAGE_FEATURE).first(),
                    pl.col(VALENCE).first(),
                    pl.col(AROUSAL).first(),
                    pl.col(DOMINANCE).first(),
                ),
                self.pad_token_id,
            ),
            batch_size=self.batch_size_for_generation,
            num_workers=self.num_workers,
            collate_fn=LoadedData.collate_fn,
        )

    def test_dataloader_for_image2text_generation(
        self,
    ) -> DataLoader[LoadedData]:
        assert self.__artemis_df is not None
        return DataLoader(
            dataset=ArtemisDataset(
                self.__artemis_df.filter(pl.col(SPLIT) == "test")
                .group_by(IMAGE_PATH)  # type: ignore
                .agg(
                    pl.col(UTTERANCE),
                    pl.col(INPUT_IDS).first(),
                    pl.col(ATTENTION_MASK).first(),
                    pl.col(TEXT_FEATURE).first(),
                    pl.col(IMAGE_FEATURE).first(),
                    pl.col(VALENCE).first(),
                    pl.col(AROUSAL).first(),
                    pl.col(DOMINANCE).first(),
                ),
                self.pad_token_id,
            ),
            batch_size=self.batch_size_for_generation,
            num_workers=self.num_workers,
            collate_fn=LoadedData.collate_fn,
        )

    @torch.no_grad()  # type: ignore
    def __compute_clip_image_features(
        self,
        image_paths: Sequence[str],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        from transformers import CLIPImageProcessor, CLIPModel  # type: ignore

        loguru.logger.info(f"Load {self.image_model_name}")
        processor = CLIPImageProcessor.from_pretrained(  # type: ignore
            self.image_model_name,
        )
        model = CLIPModel.from_pretrained(  # type: ignore
            self.image_model_name,
        )
        assert isinstance(processor, CLIPImageProcessor)
        assert isinstance(model, CLIPModel)

        class TemporaryDataset(Dataset[Tensor]):
            def __getitem__(self, index: int) -> Tensor:
                processed = processor.preprocess(  # type: ignore
                    Image.open(image_paths[index]),
                    return_tensors="pt",
                )
                pixel_values = processed["pixel_values"]  # type: ignore
                assert isinstance(pixel_values, Tensor)
                return pixel_values[0]

            def __len__(self):
                return len(image_paths)

        loguru.logger.info("Compute image features...")
        loguru.logger.info(f"device: {device}")
        model.eval()
        model.to(device)  # type: ignore

        features: list[list[float]] = []
        for batch in tqdm.tqdm(
            iterable=DataLoader(
                dataset=TemporaryDataset(),
                batch_size=self.batch_size_for_clip,
                shuffle=False,
                num_workers=self.num_workers,
            ),
            desc="Image Feature",
        ):
            batch: Tensor = batch.to(device)
            feature = model.get_image_features(
                pixel_values=cast(FloatTensor, batch)
            )
            assert feature.dim() == 2, feature.shape
            features.extend(feature.tolist())  # type: ignore

        return features

    @torch.no_grad()  # type: ignore
    def __compute_clip_text_features(
        self,
        texts: list[str],
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> list[list[float]]:
        from transformers import CLIPModel, CLIPTokenizer  # type: ignore

        loguru.logger.info(f"Load {self.image_model_name}")
        tokenizer = CLIPTokenizer.from_pretrained(  # type: ignore
            self.image_model_name
        )
        model = CLIPModel.from_pretrained(  # type: ignore
            self.image_model_name,
        )
        assert isinstance(tokenizer, CLIPTokenizer)
        assert isinstance(model, CLIPModel)

        loguru.logger.info("Compute text features...")
        loguru.logger.info(f"device: {device}")
        model.eval()
        model.to(device)  # type: ignore

        features: list[list[float]] = []
        for index in tqdm.trange(
            0,
            len(texts),
            self.batch_size_for_clip,
            desc="Text Feature",
        ):
            feature = model.get_text_features(
                **tokenizer(  # type: ignore
                    texts[index : index + self.batch_size_for_clip],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
            )
            assert feature.dim() == 2, feature.shape
            features.extend(feature.tolist())  # type: ignore

        return features

    def __create_artemis_datasets(self):
        loguru.logger.info("Load ArtEmis data...")
        df = pl.read_csv(
            self.artemis_data_path,
            separator=self.artemis_data_sep,
        )

        if self.debug_mode:
            loguru.logger.info("DEBUG MODE: use only first 100 examples.")
            df = df.head(n=100)

        df = df.with_columns(
            (
                self.wikiart_dirpath.as_posix()
                + "/"
                + pl.col(ART_STYLE)
                + "/"
                + pl.col(PAINTING)
                + ".jpg"
            ).alias(IMAGE_PATH),
        )

        loguru.logger.info("Check and modify image paths...")
        df = df.with_columns(
            pl.col(IMAGE_PATH).map_elements(
                {
                    str(p): _check_and_modify_path(p)
                    for p in df.get_column(IMAGE_PATH).unique()
                }.__getitem__
            )
        )

        loguru.logger.info("Tokenize ArtEmis utterances...")
        utterances: list[str] = df.get_column(UTTERANCE).to_list()
        tokenized = self.tokenizer(utterances)

        loguru.logger.info("Compute text features...")
        text_features = self.__compute_clip_text_features(utterances)

        loguru.logger.info("Compute image features...")
        image_paths: list[str] = df.get_column(IMAGE_PATH).unique().to_list()
        image_features = self.__compute_clip_image_features(image_paths)

        df = df.with_columns(
            pl.lit(pl.Series(INPUT_IDS, tokenized[INPUT_IDS])),
            pl.lit(pl.Series(ATTENTION_MASK, tokenized[ATTENTION_MASK])),
            pl.lit(pl.Series(TEXT_FEATURE, text_features)),
            pl.col(IMAGE_PATH)
            .map_elements(dict(zip(image_paths, image_features)).__getitem__)
            .alias(IMAGE_FEATURE),
        )

        if self.use_artemis_categorical_emotions:
            df = df.with_columns(
                pl.col(EMOTION)
                .map_elements(
                    lambda x: CATEGORY_TO_VALENCE.get(x, -100),
                    return_dtype=pl.Float32,
                )
                .alias(VALENCE),
                pl.col(EMOTION)
                .map_elements(
                    lambda x: CATEGORY_TO_AROUSAL.get(x, -100),
                    return_dtype=pl.Float32,
                )
                .alias(AROUSAL),
                pl.col(EMOTION)
                .map_elements(
                    lambda x: CATEGORY_TO_DOMINANCE.get(x, -100),
                    return_dtype=pl.Float32,
                )
                .alias(DOMINANCE),
            )
        else:
            df = df.with_columns(
                pl.lit(-100).alias(VALENCE),
                pl.lit(-100).alias(AROUSAL),
                pl.lit(-100).alias(DOMINANCE),
            )
        match self.data_split_strategy:
            case "none":
                num_full_samples = len(df)
                num_test_samples = int(num_full_samples * self.test_ratio)
                num_val_samples = int(num_full_samples * self.val_ratio)
                num_train_samples = (
                    num_full_samples - num_val_samples - num_test_samples
                )

                splits = (
                    (["train"] * num_train_samples)
                    + (["val"] * num_val_samples)
                    + (["test"] * num_test_samples)
                )
                random.Random(self.random_seed).shuffle(splits)

                df = df.with_columns(pl.lit(pl.Series(SPLIT, splits)))

            case "image":
                num_full_images = len(image_paths)
                num_test_images = int(num_full_images * self.test_ratio)
                num_val_images = int(num_full_images * self.val_ratio)
                num_train_images = (
                    num_full_images - num_val_images - num_test_images
                )

                splits = (
                    (["train"] * num_train_images)
                    + (["val"] * num_val_images)
                    + (["test"] * num_test_images)
                )
                random.Random(self.random_seed).shuffle(splits)

                df = df.with_columns(
                    pl.col(IMAGE_PATH)
                    .map_elements(dict(zip(image_paths, splits)).__getitem__)
                    .alias(SPLIT)
                )

        loguru.logger.info(df.head())

        self.__artemis_df = df
        self.__artemis_dataset_creation_completed = True

    def __create_nrc_vad_datasets(self):
        loguru.logger.info("Load NRC VAD data...")
        df = pl.read_csv(
            self.nrc_vad_lexicon_data_path,
            separator=self.nrc_vad_lexicon_data_sep,
        )

        if self.debug_mode:
            loguru.logger.info("DEBUG MODE: use only first 100 examples.")
            df = df.head(n=100)

        df = df.sample(fraction=1, shuffle=True, seed=self.random_seed)

        loguru.logger.info("Tokenize NRC VAD data...")
        words = df.get_column("Word").to_list()
        tokenized = self.tokenizer(words)

        loguru.logger.info("Compute word features...")
        word_features = self.__compute_clip_text_features(words)

        df = df.with_columns(
            pl.lit(pl.Series(INPUT_IDS, tokenized[INPUT_IDS])),
            pl.lit(pl.Series(ATTENTION_MASK, tokenized[ATTENTION_MASK])),
            pl.lit(pl.Series(TEXT_FEATURE, word_features)),
        )

        num_full_samples = len(df)
        num_test_samples = int(num_full_samples * self.test_ratio)
        num_val_samples = int(num_full_samples * self.val_ratio)
        num_train_samples = (
            num_full_samples - num_val_samples - num_test_samples
        )

        splits = (
            (["train"] * num_train_samples)
            + (["val"] * num_val_samples)
            + (["test"] * num_test_samples)
        )
        random.Random(self.random_seed).shuffle(splits)

        df = df.with_columns(pl.lit(pl.Series(SPLIT, splits)))
        self.__nrc_vad_df = df
        self.__nrc_vad_dataset_creation_completed = True

    def __create_dvisa_datasets(self) -> None:
        if self.dvisa_data_path is None:
            loguru.logger.info("DViSA not provided. Exit this function.")
            self.__dvisa_dataset_creation_completed = True
            return None

        assert self.__artemis_df is not None

        dvisa_temp_df = (
            pl.read_csv(  # type: ignore
                self.dvisa_data_path,
                separator=self.dvisa_data_sep,
            )
            .unique("artwork")
            .with_columns(
                pl.col("artwork").str.replace(".jpg", "").alias(PAINTING),
                pl.col("final_vad").cast(pl.List).list.get(0).alias(VALENCE),
                pl.col("final_vad").cast(pl.List).list.get(1).alias(AROUSAL),
                pl.col("final_vad").cast(pl.List).list.get(2).alias(DOMINANCE),
            )
        )
        painting_to_valence: dict[str, float] = {
            k: v
            for k, v in zip(
                dvisa_temp_df.get_column(PAINTING),
                dvisa_temp_df.get_column(VALENCE),
            )
            if isinstance(v, float)
        }
        painting_to_arousal: dict[str, float] = {
            k: v
            for k, v in zip(
                dvisa_temp_df.get_column(PAINTING),
                dvisa_temp_df.get_column(AROUSAL),
            )
            if isinstance(v, float)
        }
        painting_to_dominance: dict[str, float] = {
            k: v
            for k, v in zip(
                dvisa_temp_df.get_column(PAINTING),
                dvisa_temp_df.get_column(DOMINANCE),
            )
            if isinstance(v, float)
        }

        self.__dvisa_df = (
            self.__artemis_df.unique(PAINTING)
            .with_columns(  # type: ignore
                pl.col(PAINTING)
                .map_elements(lambda x: painting_to_valence.get(x, -100))
                .alias(VALENCE),
                pl.col(PAINTING)
                .map_elements(lambda x: painting_to_arousal.get(x, -100))
                .alias(AROUSAL),
                pl.col(PAINTING)
                .map_elements(lambda x: painting_to_dominance.get(x, -100))
                .alias(DOMINANCE),
            )
            .select(
                (
                    IMAGE_PATH,
                    IMAGE_FEATURE,
                    VALENCE,
                    AROUSAL,
                    DOMINANCE,
                    SPLIT,
                )
            )
        )
        self.__dvisa_dataset_creation_completed = True
