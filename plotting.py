import csv
import os
from collections import defaultdict
from pathlib import Path

import plotly
import plotly.graph_objects as go
import plotly.subplots


def set_up_figure(x_axis_name: str, num_cols: int) -> go.Figure:
    x_axis_title = {"epoch": "Training epochs", "time": "Wall-clock time (s)"}

    fig = plotly.subplots.make_subplots(rows=1, cols=num_cols)
    fig.update_xaxes(title_text=x_axis_title[x_axis_name])

    big_font = dict(family="Arial", size=18, color="black")
    small_font = dict(family="Arial", size=12, color="black")
    fig.update_layout(
        font=big_font,
        plot_bgcolor="white",
        margin=dict(r=20, t=20, b=10),
        uirevision=True,
        hovermode="x unified",
        legend=dict(font=big_font),
    )

    fig.update_xaxes(
        zeroline=False,
        linecolor="black",
        gridcolor="rgb(200,200,200)",
        griddash="5px,2px",
        ticks="outside",
        tickfont=small_font,
        title_font=big_font,
        mirror=True,
        tickcolor="black",
    )
    fig.update_yaxes(
        zeroline=False,
        linecolor="black",
        gridcolor="rgb(200,200,200)",
        griddash="5px,2px",
        ticks="outside",
        tickfont=small_font,
        title_font=big_font,
        mirror=True,
        tickcolor="black",
    )

    return fig


def create_classifier_figure(x_axis_name: str) -> go.Figure:
    fig = set_up_figure(x_axis_name=x_axis_name, num_cols=2)
    fig.update_yaxes(title_text="Cross-entropy loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    colors = plotly.colors.colorbrewer.Set1

    log_dir = Path("runs/classifier")
    csv_paths = log_dir.rglob("log.csv")
    csv_paths = sorted(csv_paths, key=lambda path: os.path.getmtime(path.parent))

    for color_idx, path in enumerate(csv_paths):
        test_name = str(path.relative_to(log_dir).parent)

        data = defaultdict(list)
        with open(path, "r") as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                for key, value in row.items():
                    data[key].append(float(value))

        xs = data[x_axis_name]
        for col, subplot_name in enumerate(["loss", "acc"], 1):
            for mode, dash in [("train", "dot"), ("test", "solid")]:
                trace_name = f"{test_name}/{mode}"
                ys = data[f"{mode}_{subplot_name}"]
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        line=dict(dash=dash, color=colors[color_idx % len(colors)]),
                        mode="lines",
                        name=trace_name,
                        legendgroup=test_name,
                        showlegend=col == 1,
                    ),
                    1,
                    col,
                )

    return fig


def create_autoencoder_figure(x_axis_name: str) -> go.Figure:
    fig = set_up_figure(x_axis_name=x_axis_name, num_cols=1)
    fig.update_yaxes(title_text="L2 loss", row=1, col=1)

    colors = plotly.colors.colorbrewer.Set1

    log_dir = Path("runs/autoencoder")
    csv_paths = log_dir.rglob("log.csv")
    csv_paths = sorted(csv_paths, key=lambda path: os.path.getmtime(path.parent))

    for color_idx, path in enumerate(csv_paths):
        test_name = str(path.relative_to(log_dir).parent)

        data = defaultdict(list)
        with open(path, "r") as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                for key, value in row.items():
                    data[key].append(float(value))

        xs = data[x_axis_name]
        for mode, dash in [("train", "dot"), ("test", "solid")]:
            trace_name = f"{test_name}/{mode}"
            ys = data[f"{mode}_loss"]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    line=dict(dash=dash, color=colors[color_idx % len(colors)]),
                    mode="lines",
                    name=trace_name,
                    legendgroup=test_name,
                ),
                1,
                1,
            )

    return fig


def main() -> None:
    save_dir = Path("figures")
    fig_ctrs = {
        "classifier": create_classifier_figure,
        "autoencoder": create_autoencoder_figure,
    }
    for model_name, fig_ctr in fig_ctrs.items():
        model_save_dir = save_dir / model_name
        model_save_dir.mkdir(parents=True, exist_ok=True)
        for x_axis_name in ["epoch", "time"]:
            fig = fig_ctr(x_axis_name)
            fig.write_html(model_save_dir / f"{x_axis_name}.html")
            fig.write_image(
                model_save_dir / f"{x_axis_name}.png", width=1500, height=700
            )


if __name__ == "__main__":
    main()
