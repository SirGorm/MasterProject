"""
Automatic rep/phase marker detection from combined accelerometer signal.

Detects peaks and valleys in the smoothed accelerometer signal to place
phase transition markers for each rep.

Marker pattern:  start → [phase transitions per rep] → stop

Usage:
    python auto_markers.py                                    # interactive mode
    python auto_markers.py -e Squat -p Person01 -s Set01      # specific set
    python auto_markers.py --all                              # process all sets
    python auto_markers.py --all --no-plot                    # batch mode, no plots
"""

import os
import json
import argparse
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

DEFAULT_DATASET_ROOT = r"C:\Users\skogl\Downloads\eirikgsk\Master_git\dataset"
COMBINED_FILENAME = "biopoint_a_combined.csv"
MARKERS_FILENAME = "markers.json"

# --- Default detection parameters (tunable per exercise) ---
EXERCISE_PARAMS = {
    "Squat": {
        "smooth_window": 15,        # smoothing filter order (samples)
        "cutoff_hz": 2.0,           # low-pass cutoff frequency
        "min_rep_seconds": 1.5,     # minimum time between peaks (seconds)
        "prominence": 1.0,          # minimum peak prominence
        "use_peaks": True,          # detect peaks (concentric push)
        "use_valleys": True,        # detect valleys (eccentric bottom)
    },
    "Benchpress": {
        "smooth_window": 15,
        "cutoff_hz": 2.0,
        "min_rep_seconds": 1.5,
        "prominence": 1.0,
        "use_peaks": True,
        "use_valleys": True,
    },
    "Pullup": {
        "smooth_window": 15,
        "cutoff_hz": 2.0,
        "min_rep_seconds": 1.5,
        "prominence": 1.0,
        "use_peaks": True,
        "use_valleys": True,
    },
    "Deadlift": {
        "smooth_window": 15,
        "cutoff_hz": 2.0,
        "min_rep_seconds": 1.5,
        "prominence": 1.0,
        "use_peaks": True,
        "use_valleys": True,
    },
}

DEFAULT_PARAMS = {
    "smooth_window": 15,
    "cutoff_hz": 2.0,
    "min_rep_seconds": 1.5,
    "prominence": 1.0,
    "use_peaks": True,
    "use_valleys": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-detect rep markers from accelerometer")
    parser.add_argument("--dataset_root", "-d", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--exercise", "-e", type=str, default=None, help="Exercise name (e.g. Squat)")
    parser.add_argument("--person", "-p", type=str, default=None, help="Person folder (e.g. Person01)")
    parser.add_argument("--set", "-s", type=str, default=None, help="Set folder (e.g. Set01)")
    parser.add_argument("--all", action="store_true", help="Process all sets in dataset")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot preview (batch mode)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing markers with >2 entries")
    parser.add_argument("--prominence", type=float, default=None, help="Override peak prominence threshold")
    parser.add_argument("--min-rep-time", type=float, default=None, help="Override min seconds between reps")
    parser.add_argument("--cutoff", type=float, default=None, help="Override low-pass cutoff frequency (Hz)")
    return parser.parse_args()


def load_signal(filepath):
    """Load combined accelerometer CSV. Returns (times, values, sampling_rate)."""
    times, values = [], []
    sampling_rate = 50  # default
    with open(filepath, "r") as f:
        header = f.readline()
        # Parse sampling rate from header
        parts = header.strip().split(",")
        for part in parts:
            if "Sampling Rate" in part:
                sr_str = part.replace("Sampling Rate:", "").replace("Hz", "").strip()
                sampling_rate = int(float(sr_str))
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            times.append(float(p[0]))
            values.append(float(p[1]))
    return np.array(times), np.array(values), sampling_rate


def load_markers(filepath):
    """Load existing markers.json."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def smooth_signal(values, sampling_rate, cutoff_hz=2.0):
    """Apply Butterworth low-pass filter."""
    nyquist = sampling_rate / 2.0
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist - 1
    b, a = butter(3, cutoff_hz / nyquist, btype='low')
    return filtfilt(b, a, values)


def detect_reps(times, smoothed, sampling_rate, params):
    """Detect peaks and/or valleys in smoothed signal. Returns sorted marker times."""
    min_distance = int(params["min_rep_seconds"] * sampling_rate)
    prominence = params["prominence"]

    marker_times = []
    marker_types = []  # 'peak' or 'valley'

    if params["use_peaks"]:
        peaks, properties = find_peaks(smoothed, distance=min_distance, prominence=prominence)
        for idx in peaks:
            marker_times.append(times[idx])
            marker_types.append("peak")

    if params["use_valleys"]:
        # Find valleys by inverting the signal
        valleys, properties = find_peaks(-smoothed, distance=min_distance, prominence=prominence)
        for idx in valleys:
            marker_times.append(times[idx])
            marker_types.append("valley")

    # Sort by time
    if marker_times:
        order = np.argsort(marker_times)
        marker_times = [marker_times[i] for i in order]
        marker_types = [marker_types[i] for i in order]

    return marker_times, marker_types


def build_markers_json(start_time, stop_time, rep_marker_times, rep_marker_types):
    """Build markers list in the expected format."""
    markers = [{"time": start_time, "label": "start", "color": "r"}]

    for t, mtype in zip(rep_marker_times, rep_marker_types):
        if t <= start_time or t >= stop_time:
            continue
        # Alternate colors for visual distinction
        color = "b" if mtype == "peak" else "y"
        markers.append({"time": t, "label": mtype, "color": color})

    markers.append({"time": stop_time, "label": "stop", "color": "g"})

    return {"markers": markers, "total_markers": len(markers)}


def plot_preview(times, raw, smoothed, markers_data, title, start_time, stop_time):
    """Show interactive matplotlib plot for verification."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Crop to exercise region with some padding
    pad = 2.0
    mask = (times >= start_time - pad) & (times <= stop_time + pad)

    ax.plot(times[mask], raw[mask], alpha=0.3, color='gray', label='Raw signal')
    ax.plot(times[mask], smoothed[mask], color='blue', linewidth=1.5, label='Smoothed')

    # Plot detected markers
    for m in markers_data["markers"]:
        t = m["time"]
        label = m["label"]
        color_map = {"r": "red", "g": "green", "b": "blue", "y": "orange"}
        c = color_map.get(m["color"], "black")

        ax.axvline(x=t, color=c, linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(t, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 5,
                label, rotation=90, ha='right', va='top', fontsize=8, color=c)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s², gravity removed)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def process_set(set_path, exercise_name, params, args):
    """Process a single set folder."""
    combined_path = os.path.join(set_path, COMBINED_FILENAME)
    markers_path = os.path.join(set_path, MARKERS_FILENAME)

    if not os.path.isfile(combined_path):
        return "no_signal"

    # Load existing markers for start/stop times
    existing = load_markers(markers_path)
    if existing is None:
        print(f"  No markers.json found, skipping (need start/stop)")
        return "no_markers"

    existing_markers = existing.get("markers", [])

    # Check if already has intermediate markers
    if len(existing_markers) > 2 and not args.overwrite:
        print(f"  Already has {len(existing_markers)} markers, skipping (use --overwrite)")
        return "skipped"

    # Find start and stop times
    start_time = None
    stop_time = None
    for m in existing_markers:
        if m["label"] == "start":
            start_time = m["time"]
        elif m["label"] == "stop":
            stop_time = m["time"]

    if start_time is None or stop_time is None:
        # Use first and last marker as fallback
        if len(existing_markers) >= 2:
            start_time = existing_markers[0]["time"]
            stop_time = existing_markers[-1]["time"]
        else:
            print(f"  Cannot determine start/stop times, skipping")
            return "no_start_stop"

    # Load signal
    times, values, sr = load_signal(combined_path)

    # Smooth
    cutoff = params["cutoff_hz"]
    smoothed = smooth_signal(values, sr, cutoff)

    # Crop to exercise region for detection
    exercise_mask = (times >= start_time) & (times <= stop_time)
    ex_times = times[exercise_mask]
    ex_smoothed = smoothed[exercise_mask]

    # Detect reps
    rep_times, rep_types = detect_reps(ex_times, ex_smoothed, sr, params)

    if not rep_times:
        print(f"  No reps detected, try adjusting parameters")
        return "no_reps"

    n_peaks = sum(1 for t in rep_types if t == "peak")
    n_valleys = sum(1 for t in rep_types if t == "valley")
    print(f"  Detected: {n_peaks} peaks, {n_valleys} valleys ({len(rep_times)} transitions)")

    # Build markers
    markers_data = build_markers_json(start_time, stop_time, rep_times, rep_types)

    # Preview plot
    label = os.path.basename(os.path.dirname(set_path)) + "/" + os.path.basename(set_path)
    title = f"{exercise_name} - {label} ({n_peaks} peaks, {n_valleys} valleys)"

    if not args.no_plot:
        plot_preview(times, values, smoothed, markers_data, title, start_time, stop_time)

        response = input("  Save these markers? (y/n/q=quit): ").strip().lower()
        if response == "q":
            return "quit"
        if response != "y":
            return "rejected"

    # Save
    with open(markers_path, "w") as f:
        json.dump(markers_data, f, indent=2)
    print(f"  Saved {len(markers_data['markers'])} markers to {markers_path}")
    return "saved"


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return

    print(f"Dataset root: {dataset_root}")
    print("=" * 60)

    counts = {"saved": 0, "skipped": 0, "no_signal": 0, "no_markers": 0,
              "no_reps": 0, "rejected": 0, "no_start_stop": 0}

    for exercise in sorted(os.listdir(dataset_root)):
        if args.exercise and exercise != args.exercise:
            continue
        exercise_path = os.path.join(dataset_root, exercise)
        if not os.path.isdir(exercise_path):
            continue

        # Get params for this exercise
        params = EXERCISE_PARAMS.get(exercise, DEFAULT_PARAMS).copy()
        # Apply CLI overrides
        if args.prominence is not None:
            params["prominence"] = args.prominence
        if args.min_rep_time is not None:
            params["min_rep_seconds"] = args.min_rep_time
        if args.cutoff is not None:
            params["cutoff_hz"] = args.cutoff

        for person in sorted(os.listdir(exercise_path)):
            if args.person and person != args.person:
                continue
            person_path = os.path.join(exercise_path, person)
            if not os.path.isdir(person_path):
                continue

            for set_name in sorted(os.listdir(person_path)):
                if args.set and set_name != args.set:
                    continue
                set_path = os.path.join(person_path, set_name)
                if not os.path.isdir(set_path):
                    continue

                print(f"\n{exercise}/{person}/{set_name}")
                result = process_set(set_path, exercise, params, args)

                if result == "quit":
                    print("\nQuitting.")
                    return

                if result in counts:
                    counts[result] += 1

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Saved:          {counts['saved']}")
    print(f"  Skipped:        {counts['skipped']} (already have markers)")
    print(f"  Rejected:       {counts['rejected']} (user said no)")
    print(f"  No signal:      {counts['no_signal']}")
    print(f"  No markers:     {counts['no_markers']}")
    print(f"  No reps found:  {counts['no_reps']}")
    print("Done.")


if __name__ == "__main__":
    main()
