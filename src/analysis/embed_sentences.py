from pathlib import Path
from typing import Literal

import loguru
import polars as pl
import torch
import tqdm
from tap import Tap
from torch import Tensor
from transformers import BertModel, BertTokenizer  # type: ignore
from transformers.modeling_outputs import (  # type: ignore
    BaseModelOutputWithPoolingAndCrossAttentions,
)


class ArgumentParser(Tap):
    data_path: Path
    output_path: Path = Path("temp.csv")

    bert: Literal["bert-base-uncased"] = "bert-base-uncased"
    batch_size: int = 256
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def main():
    loguru.logger.info("Parse command-line arguments.")
    args = ArgumentParser().parse_args()

    df = pl.read_csv(args.data_path)
    loguru.logger.info(f"Show the first few data:\n{df}")

    sentences: list[str] = df.get_column("sentence").to_list()

    loguru.logger.info("Compute BERT embeddings.")
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert)  # type: ignore
    assert isinstance(bert_tokenizer, BertTokenizer)
    bert_model = BertModel.from_pretrained(args.bert)  # type: ignore
    assert isinstance(bert_model, BertModel)

    bert_embedding_list: list[Tensor] = []
    bert_model.eval()
    bert_model.to(args.device)  # type: ignore

    with torch.no_grad():
        for index in tqdm.trange(0, len(sentences), args.batch_size):
            bert_output = bert_model.forward(
                **bert_tokenizer(  # type: ignore
                    sentences[index : index + args.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(args.device),
                return_dict=True,
            )
            assert isinstance(
                bert_output,
                BaseModelOutputWithPoolingAndCrossAttentions,
            )
            bert_embedding_list.append(
                bert_output.last_hidden_state[:, 0].cpu()
            )

    bert_embeddings = torch.cat(bert_embedding_list, dim=0)

    loguru.logger.info("Save CSV file.")
    df = df.with_columns(  # type: ignore
        pl.lit(
            pl.Series(
                "bert",
                [
                    repr(x.tolist()).replace(" ", "")  # type: ignore
                    for x in bert_embeddings.unbind(dim=0)
                ],
            )
        ),
    )
    df.write_csv(args.output_path)


if __name__ == "__main__":
    main()
