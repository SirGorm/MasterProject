import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output


'''
Use in browser to visulase signals

http://127.0.0.1:8050/

'''
# --- Konfigurasjon ---
FOLDER_PATH = r"C:/MasterProject/VS_Camera/SignalRecording/Results/Squat/recording_001"
MARKER_FILE = os.path.join(FOLDER_PATH, "m.json")

# --- Last inn CSV-filer ---
csv_files = [f for f in os.listdir(FOLDER_PATH) if f.endswith(".csv")]

# --- Les markører ---
def load_markers():
    if os.path.exists(MARKER_FILE):
        with open(MARKER_FILE, "r") as f:
            data = json.load(f)
        return data.get("markers", [])
    return []

markers = load_markers()

# --- Konverter korte fargekoder til CSS-navn ---
COLOR_MAP = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "y": "yellow",
    "m": "magenta",
    "c": "cyan",
    "k": "black"
}

# --- Dash app ---
app = Dash(__name__)
app.title = "Signal Viewer v5"

app.layout = html.Div([
    html.H1("📈 Biologisk Signal Viewer v5", style={'textAlign': 'center'}),

    html.Label("Velg én eller flere CSV-filer:"),
    dcc.Dropdown(
        id="file-dropdown",
        options=[{"label": f, "value": f} for f in csv_files],
        multi=True,
        value=[csv_files[0]] if csv_files else [],
        style={"width": "60%", "margin-bottom": "20px"}
    ),

    # Avhukinger
    html.Div([
        dcc.Checklist(
            id="options-toggle",
            options=[
                {"label": " Vis hele signalet", "value": "full"},
                {"label": " Vis markører", "value": "markers"}
            ],
            value=["markers"],
            style={"margin-bottom": "20px"}
        )
    ]),

    html.Label("Naviger i signalet:"),
    dcc.Slider(
        id="window-slider",
        min=0,
        max=10,
        step=1,
        value=0,
        tooltip={"placement": "bottom", "always_visible": True},
        updatemode='drag'
    ),
    html.Div(id="slider-info", style={"margin": "10px 0"}),

    dcc.Graph(id="signal-plot", style={"height": "70vh"}),

    html.H3("📊 Signalstatistikk (for valgt tidsvindu eller hele signalet)"),
    html.Div(id="stats-output", style={
        "backgroundColor": "#f4f4f4",
        "padding": "10px",
        "borderRadius": "8px",
        "width": "60%"
    })
])

# --- Oppdater sliderens maksimum ---
@app.callback(
    Output("window-slider", "max"),
    Input("file-dropdown", "value")
)
def update_slider_range(selected_files):
    if not selected_files:
        return 10
    df = pd.read_csv(os.path.join(FOLDER_PATH, selected_files[0]))
    total_points = len(df)
    window_size = 10000
    return max(1, total_points // window_size)

# --- Hovedcallback ---
@app.callback(
    [Output("signal-plot", "figure"),
     Output("stats-output", "children"),
     Output("slider-info", "children")],
    [Input("file-dropdown", "value"),
     Input("window-slider", "value"),
     Input("options-toggle", "value")]
)
def update_plot(selected_files, window_index, options):
    if not selected_files:
        return go.Figure(), "Ingen data valgt", ""

    window_size = 10000
    show_full = "full" in options
    show_markers = "markers" in options
    fig = go.Figure()
    stats_text = []
    slider_info = "Viser hele signalet" if show_full else f"Vindu {window_index}"

    # --- Legg til signaler ---
    for file in selected_files:
        path = os.path.join(FOLDER_PATH, file)
        df = pd.read_csv(path)

        cols = df.columns.tolist()
        if len(cols) < 2:
            time = np.arange(len(df))
            signal_cols = [cols[0]]
        else:
            time = df[cols[0]]
            signal_cols = cols[1:]

        # Klipp vindu / hele
        if show_full:
            df_window = df
        else:
            start = window_index * window_size
            end = start + window_size
            df_window = df.iloc[start:end]

        for col in signal_cols:
            fig.add_trace(go.Scatter(
                x=df_window[cols[0]],
                y=df_window[col],
                mode='lines',
                name=f"{file} - {col}"
            ))

        for col in signal_cols:
            stats_text.append(
                html.P(f"{file} | {col}: "
                       f"Mean={df_window[col].mean():.3f}, "
                       f"Std={df_window[col].std():.3f}, "
                       f"Min={df_window[col].min():.3f}, "
                       f"Max={df_window[col].max():.3f}")
            )

    # --- Legg til markører ---
    if show_markers:
        for m in markers:
            t = m.get("time")
            label = m.get("label", "")
            color = COLOR_MAP.get(m.get("color", "black"), "black")
            fig.add_vline(
                x=t,
                line=dict(color=color, dash="dash", width=2),
                annotation_text=label,
                annotation_position="top left"
            )

    fig.update_layout(
        title="Signaloversikt",
        xaxis_title="Tid / Indeks",
        yaxis_title="Amplitude",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, stats_text, slider_info


if __name__ == "__main__":
    app.run(debug=True)
    #print("Use link to see results: http://127.0.0.1:8050/")
