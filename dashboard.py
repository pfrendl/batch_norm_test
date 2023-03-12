import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from plotting import create_autoencoder_figure, create_classifier_figure

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        "model",
        dcc.RadioItems(
            ["classifier", "autoencoder"],
            "classifier",
            id="model-type",
        ),
        "x axis",
        dcc.RadioItems(
            ["epochs", "time"],
            "epochs",
            id="x-axis",
        ),
        dcc.Graph(id="live-update-graph", style={"height": "80vh"}),
        dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),
    ],
)


@app.callback(
    Output("live-update-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("model-type", "value"),
    Input("x-axis", "value"),
)
def update_graph_live(
    n_intervals: int, model_type_radio_item: str, x_axis_radio_item: str
):
    fig_ctr = {
        "classifier": create_classifier_figure,
        "autoencoder": create_autoencoder_figure,
    }[model_type_radio_item]
    x_axis_name = {"epochs": "epoch", "time": "time"}[x_axis_radio_item]
    fig = fig_ctr(x_axis_name)
    fig.update_layout(uirevision=model_type_radio_item + x_axis_radio_item)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
