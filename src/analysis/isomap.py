import ast
from pathlib import Path
from typing import Literal

import loguru
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import Isomap
from tap import Tap


class ArgumentParser(Tap):
    data_path: Path
    save_dir: Path = Path("visualization")
    use_bert_embedding: bool = True
    use_tfidf_embedding: bool = True
    n_jobs: int = -1
    alpha: float = 0.1
    debug_mode: bool = False


def apply_isomap(
    df: pl.DataFrame,
    key: Literal["bert", "tfidf"],
    n_jobs: int,
) -> pl.DataFrame:
    match key:
        case "bert":
            return df.with_columns(  # type: ignore
                pl.lit(
                    pl.Series(
                        "isomap_" + key,
                        Isomap(n_components=2, n_jobs=n_jobs)
                        .fit_transform(
                            [ast.literal_eval(x) for x in df.get_column(key)]
                        )
                        .tolist(),
                    )
                )
            )
        case "tfidf":
            return df.with_columns(  # type: ignore
                pl.lit(
                    pl.Series(
                        "isomap_" + key,
                        Isomap(n_components=2, n_jobs=n_jobs).fit_transform(
                            TfidfVectorizer().fit_transform(  # type: ignore
                                df.get_column("sentence").to_list()
                            )
                        ),
                    )
                )
            )


def visualize_isomap_vad_manipulated(
    df: pl.DataFrame,
    emb_type: Literal["bert", "tfidf"],
    emo_dim: Literal["valence", "arousal", "dominance"],
    save_dir: Path,
    alpha: float,
    high: float = 0.9,
    med: float = 0.5,
    low: float = 0.1,
):
    match emo_dim:
        case "valence":
            df = df.filter(  # type: ignore
                pl.col("arousal_manipulated").is_null()
                & pl.col("dominance_manipulated").is_null()
            )
        case "arousal":
            df = df.filter(  # type: ignore
                pl.col("valence_manipulated").is_null()
                & pl.col("dominance_manipulated").is_null()
            )
        case "dominance":
            df = df.filter(  # type: ignore
                pl.col("valence_manipulated").is_null()
                & pl.col("arousal_manipulated").is_null()
            )

    EMO_DIM_MANIPULATED = emo_dim + "_manipulated"
    df_high = df.filter(pl.col(EMO_DIM_MANIPULATED) == high)  # type: ignore
    df_med = df.filter(pl.col(EMO_DIM_MANIPULATED) == med)  # type: ignore
    df_low = df.filter(pl.col(EMO_DIM_MANIPULATED) == low)  # type: ignore

    assert len(df_high) > 0 and len(df_med) > 0 and len(df_low) > 0

    ISOMAP_EMB_TYPE = "isomap_" + emb_type
    data_high = np.array(df_high.get_column(ISOMAP_EMB_TYPE).to_list())
    data_med = np.array(df_med.get_column(ISOMAP_EMB_TYPE).to_list())
    data_low = np.array(df_low.get_column(ISOMAP_EMB_TYPE).to_list())

    assert data_high.ndim == 2 and data_high.shape[1] == 2
    assert data_med.ndim == 2 and data_med.shape[1] == 2
    assert data_low.ndim == 2 and data_low.shape[1] == 2

    # cmap = plt.get_cmap("viridis")
    # color_high = cmap(0.9)
    # color_med = cmap(0.5)
    # color_low = cmap(0.1)
    color_high = "#FF800E"
    color_med = "#595959"
    color_low = "#006BA4"

    plt.clf()
    plt.scatter(  # type: ignore
        data_low[:, 0],
        data_low[:, 1],
        marker=".",
        linewidths=0,
        alpha=alpha,
        label=f"$\\text{{{emo_dim}}}={low}$",
        color=color_low,
    )
    plt.scatter(  # type: ignore
        data_med[:, 0],
        data_med[:, 1],
        marker=".",
        linewidths=0,
        alpha=alpha,
        label=f"$\\text{{{emo_dim}}}={med}$",
        color=color_med,
    )
    plt.scatter(  # type: ignore
        data_high[:, 0],
        data_high[:, 1],
        marker=".",
        linewidths=0,
        alpha=alpha,
        label=f"$\\text{{{emo_dim}}}={high}$",
        color=color_high,
    )
    plt.scatter(  # type: ignore
        [data_high.mean()],
        [data_high.mean()],
        s=[200],
        marker="*",
        color=color_high,
        edgecolor="black",
    )
    plt.scatter(  # type: ignore
        [data_med.mean()],
        [data_med.mean()],
        s=[200],
        marker="*",
        color=color_med,
        edgecolor="black",
    )
    plt.scatter(  # type: ignore
        [data_low.mean()],
        [data_low.mean()],
        marker="*",
        s=[200],
        color=color_low,
        edgecolor="black",
    )

    legend = plt.legend(markerscale=4)
    for lh in legend.legend_handles:
        if lh is not None:
            lh.set_alpha(1)

    plt.savefig(  # type: ignore
        save_dir
        / f"isomap_visualization_{emb_type}_{emo_dim}_manipulated.png",
        bbox_inches="tight",
    )


def visualize_isomap_no_manipulation(
    df: pl.DataFrame,
    emb_type: Literal["bert", "tfidf"],
    emo_dim: Literal["valence", "arousal", "dominance"],
    save_dir: Path,
    alpha: float,
):
    df = df.filter(  # type: ignore
        pl.col("valence_manipulated").is_null()
        & pl.col("arousal_manipulated").is_null()
        & pl.col("dominance_manipulated").is_null()
    )

    xy = np.array(df.get_column("isomap_" + emb_type).to_list())
    assert xy.ndim == 2 and xy.shape[1] == 2
    x = xy[:, 0]
    y = xy[:, 1]
    c = np.array(df.get_column(emo_dim + "_predicted").to_list())
    assert c.ndim == 1

    plt.clf()
    plt.scatter(  # type: ignore
        x,
        y,
        c=c,
        # cmap=plt.get_cmap("viridis"),
        marker=".",
        linewidths=0,
        alpha=alpha,
        vmin=0.45,
        vmax=0.55,
    )
    plt.colorbar(label=f"$\\text{{{emo_dim}}}$").solids.set_alpha(1)
    # plt.legend().get_frame().set_alpha(1.0)
    plt.savefig(  # type: ignore
        save_dir / f"isomap_visualization_{emb_type}_{emo_dim}_predicted.png",
        bbox_inches="tight",
    )


def main():
    loguru.logger.info("Parse command-line arguments.")

    args = ArgumentParser(explicit_bool=True).parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    loguru.logger.info("Read csv.")
    df = pl.read_csv(args.data_path)

    if args.debug_mode:
        loguru.logger.warning("Debug mode.")
        df = df.head(n=2000)

    loguru.logger.info(f"Show the first few data:\n{df}")

    if args.use_bert_embedding:
        loguru.logger.info("Compute isomap for bert embeddings.")
        assert "bert" in df
        df = apply_isomap(df, "bert", args.n_jobs)

        loguru.logger.info("Visualize isomap for bert embeddings.")
        visualize_isomap_vad_manipulated(
            df, "bert", "valence", args.save_dir, args.alpha
        )
        visualize_isomap_vad_manipulated(
            df, "bert", "arousal", args.save_dir, args.alpha
        )
        visualize_isomap_vad_manipulated(
            df, "bert", "dominance", args.save_dir, args.alpha
        )

        visualize_isomap_no_manipulation(
            df, "bert", "valence", args.save_dir, args.alpha
        )
        visualize_isomap_no_manipulation(
            df, "bert", "arousal", args.save_dir, args.alpha
        )
        visualize_isomap_no_manipulation(
            df, "bert", "dominance", args.save_dir, args.alpha
        )

    if args.use_tfidf_embedding:
        loguru.logger.info("Compute isomap for tfidf embeddings.")
        df = apply_isomap(df, "tfidf", args.n_jobs)

        loguru.logger.info("Visualize isomap for tfidf embeddings.")
        visualize_isomap_vad_manipulated(
            df, "tfidf", "valence", args.save_dir, args.alpha
        )
        visualize_isomap_vad_manipulated(
            df, "tfidf", "arousal", args.save_dir, args.alpha
        )
        visualize_isomap_vad_manipulated(
            df, "tfidf", "dominance", args.save_dir, args.alpha
        )

        visualize_isomap_no_manipulation(
            df, "tfidf", "valence", args.save_dir, args.alpha
        )
        visualize_isomap_no_manipulation(
            df, "tfidf", "arousal", args.save_dir, args.alpha
        )
        visualize_isomap_no_manipulation(
            df, "tfidf", "dominance", args.save_dir, args.alpha
        )

    loguru.logger.info("Finish.")


if __name__ == "__main__":
    main()
