"""
Visualize phase classifications across a full exercise session.

Picks a random session, loads all biosignals, creates 2-second windows,
classifies each as rest/concentric/eccentric using rule-based joint kinematics,
and plots the full-length signals with phase-colored backgrounds.

Usage:
    python -m ml.v3.visualize_phases
    python -m ml.v3.visualize_phases --exercise Squat --session Set01
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple

from ml.v3.config import CONFIG, SIGNALS
from ml.v3.data.preprocessing import JointProcessor, DataPreprocessor
from ml.v3.data.validate_data import DataValidator


# Phase colors
PHASE_COLORS = {
    'rest':       '#AAAAAA',   # gray
    'concentric': '#4CAF50',   # green
    'eccentric':  '#F44336',   # red
    'unknown':    '#FFC107',   # amber
}


def load_session_signals(session_path: Path) -> Dict[str, Dict]:
    """Load all enabled biosignal CSVs for a session."""
    signals = {}
    for sig_name, sig_cfg in SIGNALS.items():
        if not sig_cfg.enabled or sig_cfg.file_name.endswith('.json'):
            continue
        sig_path = session_path / sig_cfg.file_name
        if sig_path.exists():
            try:
                df = pd.read_csv(sig_path)
                signals[sig_name] = {
                    'time': df.iloc[:, 0].values,
                    'data': df.iloc[:, 1].values,
                    'sampling_rate': sig_cfg.sampling_rate,
                }
            except Exception:
                pass
    return signals


def classify_windows(
    joint_data: Dict,
    start_time: float,
    end_time: float,
    window_sec: float,
    exercise_type: str,
) -> List[Dict]:
    """Create windows and classify each phase from joint kinematics."""
    jp = JointProcessor()
    windows = []
    t = start_time
    while t + window_sec <= end_time:
        phase = jp.detect_phase(joint_data, t, t + window_sec, exercise_type)
        windows.append({
            'start': t,
            'end': t + window_sec,
            'phase': phase,
        })
        t += window_sec  # non-overlapping
    return windows


def plot_session(
    signals: Dict[str, Dict],
    windows: List[Dict],
    exercise_type: str,
    session_id: str,
    markers: List[Dict] = None,
    output_path: Path = None,
):
    """
    Plot all signals in subplots with phase-colored backgrounds.

    Each subplot shows one signal modality. The background of each 2-second
    window is shaded by its phase classification.
    """
    # Filter to signals that have data
    active_signals = {k: v for k, v in signals.items() if len(v['data']) > 0}
    n_signals = len(active_signals)
    if n_signals == 0:
        print("No signals to plot!")
        return

    fig, axes = plt.subplots(n_signals, 1, figsize=(18, 3 * n_signals), sharex=True)
    if n_signals == 1:
        axes = [axes]

    fig.suptitle(
        f'Phase Classification — {exercise_type} / {session_id}\n'
        f'({len(windows)} windows, {sum(1 for w in windows if w["phase"]=="concentric")} concentric, '
        f'{sum(1 for w in windows if w["phase"]=="eccentric")} eccentric, '
        f'{sum(1 for w in windows if w["phase"]=="rest")} rest, '
        f'{sum(1 for w in windows if w["phase"]=="unknown")} unknown)',
        fontsize=14, fontweight='bold'
    )

    for ax_idx, (sig_name, sig_info) in enumerate(active_signals.items()):
        ax = axes[ax_idx]
        time = sig_info['time']
        data = sig_info['data']

        # Plot signal
        ax.plot(time, data, linewidth=0.5, color='#333333', alpha=0.8)
        ax.set_ylabel(sig_name, fontsize=11, fontweight='bold')

        # Shade phase backgrounds
        for w in windows:
            color = PHASE_COLORS.get(w['phase'], '#FFC107')
            ax.axvspan(w['start'], w['end'], alpha=0.25, color=color, linewidth=0)

        # Draw window boundaries as thin dotted lines
        for w in windows:
            ax.axvline(w['start'], color='#999999', linewidth=0.3, linestyle=':')

        # Add markers (peaks/valleys) if available
        if markers:
            for m in markers:
                t_marker = m.get('time', 0)
                label = m.get('label', '').lower()
                if label == 'valley':
                    ax.axvline(t_marker, color='orange', linewidth=1.0, linestyle='--', alpha=0.7)
                elif label == 'peak':
                    ax.axvline(t_marker, color='blue', linewidth=1.0, linestyle='--', alpha=0.5)

        ax.grid(True, alpha=0.2)

    # Legend
    legend_patches = [
        mpatches.Patch(color=PHASE_COLORS['rest'], alpha=0.4, label='Rest'),
        mpatches.Patch(color=PHASE_COLORS['eccentric'], alpha=0.4, label='Eccentric'),
        mpatches.Patch(color=PHASE_COLORS['concentric'], alpha=0.4, label='Concentric'),
        mpatches.Patch(color=PHASE_COLORS['unknown'], alpha=0.4, label='Unknown'),
    ]
    if markers and any(m.get('label', '').lower() == 'valley' for m in markers):
        legend_patches.append(plt.Line2D([0], [0], color='orange', linestyle='--', label='Valley (rep)'))
        legend_patches.append(plt.Line2D([0], [0], color='blue', linestyle='--', label='Peak'))
    axes[-1].legend(handles=legend_patches, loc='lower right', fontsize=9, ncol=3)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    import json

    parser = argparse.ArgumentParser(description='Visualize phase classifications')
    parser.add_argument('--exercise', type=str, default=None,
                        help='Exercise type (Squat, Benchpress, Deadlift). Random if not set.')
    parser.add_argument('--session', type=str, default=None,
                        help='Session folder name (e.g. Set01). Random if not set.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path. Shows plot interactively if not set.')
    args = parser.parse_args()

    # Validate dataset to get available sessions
    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    validator.validate_all()
    valid_sessions = validator.get_valid_sessions()

    if not valid_sessions:
        print("No valid sessions found!")
        return

    # Filter by exercise if specified
    if args.exercise:
        filtered = [(ex, sid, p) for ex, sid, p in valid_sessions if ex.lower() == args.exercise.lower()]
        if not filtered:
            print(f"No sessions found for exercise '{args.exercise}'")
            print(f"Available: {set(ex for ex, _, _ in valid_sessions)}")
            return
        valid_sessions = filtered

    # Filter by session if specified
    if args.session:
        filtered = [(ex, sid, p) for ex, sid, p in valid_sessions if args.session.lower() in sid.lower()]
        if not filtered:
            print(f"No sessions found matching '{args.session}'")
            return
        valid_sessions = filtered

    # Pick a random session
    exercise, session_id, session_path = random.choice(valid_sessions)
    print(f"\nSelected: {exercise} / {session_id}")
    print(f"Path: {session_path}")

    # Load joint data
    jp = JointProcessor()
    joint_path = session_path / CONFIG.data.joints_file
    joint_data = jp.load_joint_data(joint_path) if joint_path.exists() else None

    if not joint_data:
        print("No joint data available for phase detection!")
        return

    # Load markers
    markers_path = session_path / CONFIG.data.markers_file
    markers = []
    if markers_path.exists():
        with open(markers_path, 'r') as f:
            markers_data = json.load(f)
        markers = markers_data.get('markers', [])

    # Find exercise time range from markers
    start_marker = next((m for m in markers if m.get('label', '').lower() == 'start'), None)
    stop_marker = next((m for m in markers if m.get('label', '').lower() == 'stop'), None)

    if start_marker:
        exercise_start = start_marker['time']
    else:
        exercise_start = 0.0

    if stop_marker:
        exercise_end = stop_marker['time']
    else:
        # Estimate from joint data
        frames = joint_data.get('frames', [])
        if frames:
            first_ts = frames[0].get('timestamp_usec', 0)
            last_ts = frames[-1].get('timestamp_usec', 0)
            exercise_end = (last_ts - first_ts) / 1e6
        else:
            exercise_end = 60.0

    print(f"Exercise window: {exercise_start:.1f}s - {exercise_end:.1f}s ({exercise_end - exercise_start:.1f}s)")

    # Load signals
    signals = load_session_signals(session_path)
    print(f"Loaded signals: {list(signals.keys())}")

    # Classify windows
    window_sec = CONFIG.data.time_window_sec
    windows = classify_windows(joint_data, exercise_start, exercise_end, window_sec, exercise)

    from collections import Counter
    phase_counts = Counter(w['phase'] for w in windows)
    print(f"Windows: {len(windows)}, Phases: {dict(phase_counts)}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        safe_session_id = session_id.replace('/', '_').replace('\\', '_')
        output_path = CONFIG.output.plots_dir / f'phase_visualization_{exercise}_{safe_session_id}.png'

    # Plot
    plot_session(signals, windows, exercise, session_id, markers, output_path)


if __name__ == '__main__':
    main()
