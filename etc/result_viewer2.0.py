import os
import json
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, ALL, ctx
from dash.exceptions import PreventUpdate


'''
Use in browser to visualise signals

http://127.0.0.1:8050/

Browse through all exercise folders and recordings in the dataset.
Structure: dataset / exercise / person_id / recording_NNN /
'''

# --- Konfigurasjon ---
DATASET_ROOT = r"C:\Users\skogl\Downloads\eirikgsk\Master_git\dataset"

COLORS = ["r", "g", "b", "y", "m", "c"]
COLOR_MAP = {
    "r": "red", "g": "green", "b": "blue", "y": "yellow",
    "m": "magenta", "c": "cyan", "k": "black"
}


# --- Hjelpefunksjoner ---
def list_subdirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def get_csv_files(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(f for f in os.listdir(folder_path) if f.endswith(".csv"))


def get_folder(exercise, person, recording):
    if exercise and person and recording:
        return os.path.join(DATASET_ROOT, exercise, person, recording)
    return None


def load_markers(folder_path):
    for name in ["markers.json", "m.json"]:
        path = os.path.join(folder_path, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("markers", [])
    return []


def save_markers(folder_path, markers):
    path = os.path.join(folder_path, "markers.json")
    with open(path, "w") as f:
        json.dump({"markers": markers, "total_markers": len(markers)}, f, indent=2)


def build_marker_table(markers):
    if not markers:
        return html.P("Ingen markorer", style={"color": "#888"})
    header = html.Tr([
        html.Th("Time (s)", style={"width": "25%"}),
        html.Th("Label", style={"width": "25%"}),
        html.Th("Color", style={"width": "20%"}),
        html.Th("", style={"width": "15%"}),
    ])
    rows = []
    for i, m in enumerate(markers):
        rows.append(html.Tr([
            html.Td(dcc.Input(
                id={"type": "marker-time", "index": i},
                type="number", step=0.01,
                value=round(m.get("time", 0), 3),
                style={"width": "100%"}
            )),
            html.Td(dcc.Input(
                id={"type": "marker-label", "index": i},
                type="text",
                value=m.get("label", ""),
                style={"width": "100%"}
            )),
            html.Td(dcc.Dropdown(
                id={"type": "marker-color", "index": i},
                options=[{"label": COLOR_MAP[c], "value": c} for c in COLORS],
                value=m.get("color", "r"),
                clearable=False,
                style={"width": "100%"}
            )),
            html.Td(html.Button(
                "Slett", id={"type": "marker-delete", "index": i},
                n_clicks=0,
                style={"backgroundColor": "#e44", "color": "white", "border": "none",
                       "padding": "4px 10px", "borderRadius": "4px", "cursor": "pointer"}
            )),
        ]))
    return html.Table([html.Thead(header), html.Tbody(rows)],
                      style={"width": "100%", "borderCollapse": "collapse"})


# --- Initialverdier ---
exercises = list_subdirs(DATASET_ROOT)
init_exercise = exercises[0] if exercises else ""
persons = list_subdirs(os.path.join(DATASET_ROOT, init_exercise)) if init_exercise else []
init_person = persons[0] if persons else ""
recordings = list_subdirs(os.path.join(DATASET_ROOT, init_exercise, init_person)) if init_person else []
init_recording = recordings[0] if recordings else ""
init_folder = get_folder(init_exercise, init_person, init_recording)
init_csvs = get_csv_files(init_folder) if init_folder else []

# --- Dash app ---
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Signal Viewer v5"

app.layout = html.Div([
    # Hidden store for markers state
    dcc.Store(id="markers-store", data=[]),

    html.H1("Signal Viewer v5", style={'textAlign': 'center'}),

    # --- Folder selectors ---
    html.Div([
        html.Div([
            html.Label("Exercise:"),
            dcc.Dropdown(id="exercise-dropdown",
                         options=[{"label": e, "value": e} for e in exercises],
                         value=init_exercise, clearable=False, style={"width": "100%"}),
        ], style={"width": "22%", "display": "inline-block", "marginRight": "15px"}),
        html.Div([
            html.Label("Person:"),
            dcc.Dropdown(id="person-dropdown",
                         options=[{"label": p, "value": p} for p in persons],
                         value=init_person, clearable=False, style={"width": "100%"}),
        ], style={"width": "22%", "display": "inline-block", "marginRight": "15px"}),
        html.Div([
            html.Label("Recording:"),
            dcc.Dropdown(id="recording-dropdown",
                         options=[{"label": r, "value": r} for r in recordings],
                         value=init_recording, clearable=False, style={"width": "100%"}),
        ], style={"width": "22%", "display": "inline-block"}),
    ], style={"marginBottom": "20px"}),

    html.Div(id="folder-path-display", style={
        "backgroundColor": "#eef", "padding": "6px 12px", "borderRadius": "4px",
        "marginBottom": "15px", "fontFamily": "monospace", "fontSize": "13px"
    }),

    html.Label("Velg en eller flere CSV-filer:"),
    dcc.Dropdown(id="file-dropdown",
                 options=[{"label": f, "value": f} for f in init_csvs],
                 multi=True, value=[init_csvs[0]] if init_csvs else [],
                 style={"width": "60%", "marginBottom": "20px"}),

    html.Div([
        dcc.Checklist(id="options-toggle", options=[
            {"label": " Vis hele signalet", "value": "full"},
            {"label": " Vis markorer", "value": "markers"}
        ], value=["markers"], style={"marginBottom": "20px"})
    ]),

    html.Label("Naviger i signalet:"),
    dcc.Slider(id="window-slider", min=0, max=10, step=1, value=0,
               tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'),
    html.Div(id="slider-info", style={"margin": "10px 0"}),

    dcc.Graph(id="signal-plot", style={"height": "70vh"}),

    html.H3("Signalstatistikk"),
    html.Div(id="stats-output", style={
        "backgroundColor": "#f4f4f4", "padding": "10px", "borderRadius": "8px", "width": "60%"
    }),

    # --- Marker editor section ---
    html.Hr(style={"marginTop": "30px"}),
    html.H3("Marker Editor"),

    html.Div([
        html.Div([
            html.Label("Klikk i plottet for aa legge til en marker, eller legg til manuelt:"),
            html.Div([
                dcc.Input(id="new-marker-time", type="number", placeholder="Tid (s)",
                          step=0.01, style={"width": "120px", "marginRight": "8px"}),
                dcc.Input(id="new-marker-label", type="text", placeholder="Label",
                          style={"width": "120px", "marginRight": "8px"}),
                dcc.Dropdown(id="new-marker-color",
                             options=[{"label": COLOR_MAP[c], "value": c} for c in COLORS],
                             value="r", clearable=False,
                             style={"width": "120px", "display": "inline-block", "marginRight": "8px"}),
                html.Button("Legg til", id="add-marker-btn", n_clicks=0,
                            style={"backgroundColor": "#4a4", "color": "white", "border": "none",
                                   "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px"}),
        ]),
    ]),

    html.Div(id="marker-table-container"),

    html.Div([
        html.Button("Lagre markorer", id="save-markers-btn", n_clicks=0,
                    style={"backgroundColor": "#36c", "color": "white", "border": "none",
                           "padding": "10px 24px", "borderRadius": "4px", "cursor": "pointer",
                           "fontSize": "15px", "marginTop": "10px", "marginRight": "10px"}),
        html.Span(id="save-status", style={"color": "#888"}),
    ]),
])


# ===== Folder navigation callbacks =====

@app.callback(
    [Output("person-dropdown", "options"), Output("person-dropdown", "value")],
    Input("exercise-dropdown", "value")
)
def update_persons(exercise):
    if not exercise:
        return [], None
    plist = list_subdirs(os.path.join(DATASET_ROOT, exercise))
    return [{"label": p, "value": p} for p in plist], (plist[0] if plist else None)


@app.callback(
    [Output("recording-dropdown", "options"), Output("recording-dropdown", "value")],
    [Input("exercise-dropdown", "value"), Input("person-dropdown", "value")]
)
def update_recordings(exercise, person):
    if not exercise or not person:
        return [], None
    recs = list_subdirs(os.path.join(DATASET_ROOT, exercise, person))
    return [{"label": r, "value": r} for r in recs], (recs[0] if recs else None)


@app.callback(
    [Output("file-dropdown", "options"), Output("file-dropdown", "value"),
     Output("folder-path-display", "children")],
    [Input("exercise-dropdown", "value"), Input("person-dropdown", "value"),
     Input("recording-dropdown", "value")]
)
def update_files(exercise, person, recording):
    if not exercise or not person or not recording:
        return [], [], "Ingen mappe valgt"
    folder = get_folder(exercise, person, recording)
    csvs = get_csv_files(folder)
    return [{"label": f, "value": f} for f in csvs], ([csvs[0]] if csvs else []), \
        f"{exercise} / {person} / {recording}"


@app.callback(
    Output("window-slider", "max"),
    [Input("file-dropdown", "value"), Input("exercise-dropdown", "value"),
     Input("person-dropdown", "value"), Input("recording-dropdown", "value")]
)
def update_slider_range(selected_files, exercise, person, recording):
    if not selected_files or not exercise or not person or not recording:
        return 10
    folder = get_folder(exercise, person, recording)
    try:
        df = pd.read_csv(os.path.join(folder, selected_files[0]))
        return max(1, len(df) // 10000)
    except Exception:
        return 10


# ===== Load markers when folder changes =====

@app.callback(
    Output("markers-store", "data"),
    [Input("exercise-dropdown", "value"), Input("person-dropdown", "value"),
     Input("recording-dropdown", "value")]
)
def load_markers_into_store(exercise, person, recording):
    folder = get_folder(exercise, person, recording)
    if not folder:
        return []
    return load_markers(folder)


# ===== Add marker: from button or by clicking in plot =====

@app.callback(
    Output("markers-store", "data", allow_duplicate=True),
    [Input("add-marker-btn", "n_clicks"), Input("signal-plot", "clickData")],
    [State("new-marker-time", "value"), State("new-marker-label", "value"),
     State("new-marker-color", "value"), State("markers-store", "data")],
    prevent_initial_call=True
)
def add_marker(btn_clicks, click_data, time_val, label_val, color_val, markers):
    trigger = ctx.triggered_id
    markers = list(markers) if markers else []

    if trigger == "add-marker-btn":
        if time_val is None:
            raise PreventUpdate
        label = label_val or f"M{len(markers) + 1}"
        markers.append({"time": round(time_val, 3), "label": label, "color": color_val or "r"})
    elif trigger == "signal-plot" and click_data:
        t = click_data["points"][0]["x"]
        label = f"M{len(markers) + 1}"
        markers.append({"time": round(t, 3), "label": label, "color": color_val or "r"})

    markers.sort(key=lambda m: m["time"])
    return markers


# ===== Delete marker =====

@app.callback(
    Output("markers-store", "data", allow_duplicate=True),
    Input({"type": "marker-delete", "index": ALL}, "n_clicks"),
    State("markers-store", "data"),
    prevent_initial_call=True
)
def delete_marker(n_clicks_list, markers):
    if not markers or not any(n_clicks_list):
        raise PreventUpdate
    trigger = ctx.triggered_id
    if trigger and "index" in trigger:
        idx = trigger["index"]
        if 0 <= idx < len(markers):
            markers = [m for i, m in enumerate(markers) if i != idx]
    return markers


# ===== Update marker time/label/color from table =====

@app.callback(
    Output("markers-store", "data", allow_duplicate=True),
    [Input({"type": "marker-time", "index": ALL}, "value"),
     Input({"type": "marker-label", "index": ALL}, "value"),
     Input({"type": "marker-color", "index": ALL}, "value")],
    State("markers-store", "data"),
    prevent_initial_call=True
)
def update_marker_fields(times, labels, colors, markers):
    if not markers:
        raise PreventUpdate
    updated = []
    for i, m in enumerate(markers):
        updated.append({
            "time": round(times[i], 3) if i < len(times) and times[i] is not None else m["time"],
            "label": labels[i] if i < len(labels) and labels[i] is not None else m["label"],
            "color": colors[i] if i < len(colors) and colors[i] is not None else m["color"],
        })
    return updated


# ===== Render marker table =====

@app.callback(
    Output("marker-table-container", "children"),
    Input("markers-store", "data")
)
def render_marker_table(markers):
    return build_marker_table(markers or [])


# ===== Save markers to file =====

@app.callback(
    Output("save-status", "children"),
    Input("save-markers-btn", "n_clicks"),
    [State("markers-store", "data"), State("exercise-dropdown", "value"),
     State("person-dropdown", "value"), State("recording-dropdown", "value")],
    prevent_initial_call=True
)
def save_markers_to_file(n_clicks, markers, exercise, person, recording):
    folder = get_folder(exercise, person, recording)
    if not folder:
        return "Ingen mappe valgt"
    save_markers(folder, markers or [])
    return f"Lagret {len(markers or [])} markorer til markers.json"


# ===== Main plot + stats =====

@app.callback(
    [Output("signal-plot", "figure"), Output("stats-output", "children"),
     Output("slider-info", "children")],
    [Input("file-dropdown", "value"), Input("window-slider", "value"),
     Input("options-toggle", "value"), Input("exercise-dropdown", "value"),
     Input("person-dropdown", "value"), Input("recording-dropdown", "value"),
     Input("markers-store", "data")]
)
def update_plot(selected_files, window_index, options, exercise, person, recording, markers):
    if not selected_files or not exercise or not person or not recording:
        return go.Figure(), "Ingen data valgt", ""

    folder = get_folder(exercise, person, recording)
    window_size = 10000
    show_full = "full" in options
    show_markers = "markers" in options
    fig = go.Figure()
    stats_text = []
    slider_info = "Viser hele signalet" if show_full else f"Vindu {window_index}"

    for file in selected_files:
        path = os.path.join(folder, file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        signal_cols = [cols[0]] if len(cols) < 2 else cols[1:]

        df_window = df if show_full else df.iloc[window_index * window_size:(window_index + 1) * window_size]

        for col in signal_cols:
            fig.add_trace(go.Scatter(x=df_window[cols[0]], y=df_window[col],
                                     mode='lines', name=f"{file} - {col}"))
        for col in signal_cols:
            stats_text.append(html.P(
                f"{file} | {col}: Mean={df_window[col].mean():.3f}, "
                f"Std={df_window[col].std():.3f}, Min={df_window[col].min():.3f}, "
                f"Max={df_window[col].max():.3f}"))

    if show_markers and markers:
        for m in markers:
            t = m.get("time")
            label = m.get("label", "")
            color = COLOR_MAP.get(m.get("color", "r"), "red")
            fig.add_vline(x=t, line=dict(color=color, dash="dash", width=2),
                          annotation_text=label, annotation_position="top left")

    fig.update_layout(
        title=f"{exercise} / {person} / {recording}",
        xaxis_title="Tid / Indeks", yaxis_title="Amplitude",
        hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig, stats_text, slider_info


if __name__ == "__main__":
    app.run(debug=True)
