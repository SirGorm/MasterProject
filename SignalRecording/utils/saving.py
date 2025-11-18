import csv
import json
from pathlib import Path
from datetime import datetime
import logging


def get_next_recording_folder(base_dir):
    base = Path(base_dir)
    base.mkdir(exist_ok=True)

    i = 1
    while True:
        folder = base / f"recording_{i:03d}"
        if not folder.exists():
            folder.mkdir()
            return folder
        i += 1


def save_recorded_data(recorded_data, sampling_rates, markers, base_dir, recording_start_time, recording_start_offset):
    recording_dir = get_next_recording_folder(base_dir)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save channel CSV files
    for channel, data in recorded_data.items():
        if data:
            filename = recording_dir / f"biopoint_{channel}.csv"
            fs = sampling_rates[channel.split('_')[0]]

            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Time (s)", channel, f"Sampling Rate: {fs} Hz"])
                for i, value in enumerate(data):
                    writer.writerow([i / fs, value])

    # Metadata
    metadata_file = recording_dir / "metadata.json"
    metadata = {
        "timestamp": timestamp_str,
        "recording_start": datetime.fromtimestamp(
            recording_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "sampling_rates": sampling_rates,
        "total_samples": {k: len(v) for k, v in recorded_data.items()},
        "duration_seconds": {k: len(v) / sampling_rates.get(k.split('_')[0], 1)
                             for k, v in recorded_data.items()}
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Markers
    markers_file = recording_dir / "markers.json"
    marker_list = [
        {"time": t - recording_start_offset, "label": label, "color": color}
        for t, label, color in markers
    ]

    with open(markers_file, 'w') as f:
        json.dump({"markers": marker_list, "total_markers": len(marker_list)}, f, indent=2)

    return recording_dir
