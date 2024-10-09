import datetime
import os
from pathlib import Path
from typing import Literal

import loguru
import torch
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.loggers import CSVLogger
from tap import Tap
from transformers import (  # type: ignore
    BartTokenizer,
    GPT2Tokenizer,
    T5Tokenizer,
)

from ..callback.generate_sentences import GenerateSentences
from ..datamodule.datamodule_artemis import ArtemisDataModule
from ..model.poe_vad_vae import POEVADVAE


class ArgumentParser(Tap):
    artemis_data_path: Path = Path("data") / "artemis_dataset_original.tsv"
    artemis_data_sep: Literal["\t", ","] = "\t"
    use_artemis_categorical_emotions: bool = False

    nrc_vad_lexicon_data_path: Path = (
        Path("VAD-VAE") / "NRC-VAD" / "NRC-VAD-Lexicon.txt"
    )
    nrc_vad_lexicon_data_sep: Literal["\t", ","] = "\t"

    dvisa_data_path: Path = Path("D-ViSA") / "D-ViSA.csv"
    dvisa_data_sep: Literal["\t", ","] = ","

    wikiart_dirpath: Path = Path("wikiart")

    batch_size: int = 32
    batch_size_for_clip: int = 128
    batch_size_for_generation: int = 256
    num_workers: int = (lambda x: 0 if x is None else x)(os.cpu_count())

    data_split_strategy: Literal["none", "image"] = "image"
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    decoder_model_name: Literal[
        "facebook/bart-base",
        "facebook/bart-large",
        "t5-base",
        "t5-large",
        "gpt2",
        "gpt2-large",
    ] = "t5-base"
    freeze_decoder: bool = False
    image_model_name: Literal[
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-large-patch14-336",
    ] = "openai/clip-vit-large-patch14"
    random_seed: int = 2023
    debug_mode: bool = False
    dim_valence: int = 8
    dim_arousal: int = 8
    dim_dominance: int = 8
    dim_content: int = 488
    prefix_length: int = 8
    kl_weight: float = 1
    mi_weight: float = 1
    info_weight: float = 1
    save_dir: Path = Path("save_dir")
    name: str = "vad-vae"
    version: str = datetime.datetime.now().isoformat()
    max_epochs: int = 10
    lr: float = 1e-5
    num_warmup_steps: int = 1000
    gradient_clip_val: float = 100

    max_length_for_inference: int = 50
    beam_size_for_beam_search: int = 3
    top_p_for_nucleus_sampling: float = 0.9
    temperature_for_inference: float = 1

    enable_checkpointing: bool = False
    enable_progress_bar: bool = False
    generate_sentences_every_n_epoch: int = 10


def main():
    args = ArgumentParser(explicit_bool=True).parse_args()

    loguru.logger.info(args)

    loguru.logger.info("Seed everything.")
    seed_everything(args.random_seed)

    loguru.logger.info(f"Load {args.decoder_model_name} Tokenizer...")
    match args.decoder_model_name:
        case "facebook/bart-base" | "facebook/bart-large":
            tokenizer = BartTokenizer.from_pretrained(  # type: ignore
                args.decoder_model_name,
            )
            assert isinstance(tokenizer, BartTokenizer)
        case "t5-base" | "t5-large":
            tokenizer = T5Tokenizer.from_pretrained(  # type: ignore
                args.decoder_model_name,
            )
            assert isinstance(tokenizer, T5Tokenizer)
        case "gpt2" | "gpt2-large":
            tokenizer = GPT2Tokenizer.from_pretrained(  # type: ignore
                args.decoder_model_name
            )
            assert isinstance(tokenizer, GPT2Tokenizer)
            tokenizer.pad_token_id = tokenizer.eos_token_id

    loguru.logger.info("Create Lightning DataModule...")
    datamodule = ArtemisDataModule(
        tokenizer=tokenizer,
        artemis_data_path=args.artemis_data_path,
        artemis_data_sep=args.artemis_data_sep,
        use_artemis_categorical_emotions=args.use_artemis_categorical_emotions,
        nrc_vad_lexicon_data_path=args.nrc_vad_lexicon_data_path,
        nrc_vad_lexicon_data_sep=args.nrc_vad_lexicon_data_sep,
        dvisa_data_path=args.dvisa_data_path,
        dvisa_data_sep=args.dvisa_data_sep,
        wikiart_dirpath=args.wikiart_dirpath,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_seed=args.random_seed,
        data_split_strategy=args.data_split_strategy,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        image_model_name=args.image_model_name,
        batch_size_for_clip=args.batch_size_for_clip,
        batch_size_for_generation=args.batch_size_for_generation,
        debug_mode=args.debug_mode,
    )
    datamodule.setup()

    loguru.logger.info("Create Lightning Module...")
    model = POEVADVAE(
        decoder_model_name=args.decoder_model_name,
        freeze_decoder=args.freeze_decoder,
        dim_valence=args.dim_valence,
        dim_arousal=args.dim_arousal,
        dim_dominance=args.dim_dominance,
        dim_content=args.dim_content,
        dim_text_feature=datamodule.dim_text_feature,
        dim_image_feature=datamodule.dim_image_feature,
        prefix_length=args.prefix_length,
        kl_weight=args.kl_weight,
        mi_weight=args.mi_weight,
        info_weight=args.info_weight,
        lr=args.lr,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_epochs * datamodule.num_steps_per_epoch,
    )

    loguru.logger.info("Create Trainer...")
    trainer_logger = CSVLogger(
        save_dir=args.save_dir,
        name=args.name,
        version=args.version,
    )

    log_dir = Path(trainer_logger.log_dir)

    callbacks: list[Callback] = []

    test_dataloader_for_image2text_generation = (
        datamodule.test_dataloader_for_image2text_generation()
    )
    callbacks.append(
        GenerateSentences(
            dataloader=test_dataloader_for_image2text_generation,
            dirpath=log_dir / "generate_test_beam",
            tokenizer=tokenizer,
            max_length=args.max_length_for_inference,
            beam_size=args.beam_size_for_beam_search,
            do_sample=False,
            top_p=1.0,
            temperature=args.temperature_for_inference,
            vads=[
                # No manipulation:
                (None, None, None),
                # Valence manipulation:
                (0.1, None, None),
                (0.5, None, None),
                (0.9, None, None),
                # Arousal manipulation:
                (None, 0.1, None),
                (None, 0.5, None),
                (None, 0.9, None),
                # Dominance manipulation:
                (None, None, 0.1),
                (None, None, 0.5),
                (None, None, 0.9),
            ],
            every_n_epoch=args.generate_sentences_every_n_epoch,
        )
    )
    callbacks.append(
        GenerateSentences(
            dataloader=test_dataloader_for_image2text_generation,
            dirpath=log_dir / "generate_test_nucleus",
            tokenizer=tokenizer,
            max_length=args.max_length_for_inference,
            beam_size=1,
            do_sample=True,
            top_p=args.top_p_for_nucleus_sampling,
            temperature=args.temperature_for_inference,
            vads=[
                # No manipulation:
                (None, None, None),
                # Valence manipulation:
                (0.1, None, None),
                (0.5, None, None),
                (0.9, None, None),
                # Arousal manipulation:
                (None, 0.1, None),
                (None, 0.5, None),
                (None, 0.9, None),
                # Dominance manipulation:
                (None, None, 0.1),
                (None, None, 0.5),
                (None, None, 0.9),
            ],
            every_n_epoch=args.generate_sentences_every_n_epoch,
        )
    )

    trainer = Trainer(
        logger=trainer_logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        enable_checkpointing=args.enable_checkpointing,
        enable_progress_bar=args.enable_progress_bar,
    )

    log_dir.mkdir(exist_ok=True, parents=True)
    args.save(log_dir / "args.json")  # type: ignore

    loguru.logger.info("Start fitting...")
    torch.set_float32_matmul_precision("high")
    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    main()
