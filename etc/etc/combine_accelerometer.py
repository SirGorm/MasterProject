"""
Traverse the entire dataset and combine biopoint_ax, biopoint_ay, biopoint_az
into a single gravity-removed accelerometer magnitude signal.

Output: biopoint_a_combined.csv in each recording folder.

Usage:
    python combine_accelerometer.py
    python combine_accelerometer.py --dataset_root "C:\path\to\data"
    python combine_accelerometer.py --dry-run
"""

import os
import argparse
import numpy as np

GRAVITY = 9.80665  # m/s²
SAMPLING_RATE = 50  # Hz
OUTPUT_FILENAME = "biopoint_a_combined.csv"
AXIS_FILES = ["biopoint_ax.csv", "biopoint_ay.csv", "biopoint_az.csv"]

DEFAULT_DATASET_ROOT = r"c:\Users\skogl\Downloads\eirikgsk\Master_git\dataset"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine ax/ay/az accelerometer files and remove gravity"
    )
    parser.add_argument(
        "--dataset_root", "-d", type=str, default=DEFAULT_DATASET_ROOT,
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be done, without writing files"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing combined files"
    )
    return parser.parse_args()


def load_axis_csv(filepath):
    """Load a single-axis accelerometer CSV. Returns (times, values)."""
    times = []
    values = []
    with open(filepath, "r") as f:
        header = f.readline()  # skip header: "Time (s),ax,Sampling Rate: 50 Hz"
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            times.append(float(parts[0]))
            values.append(float(parts[1]))
    return np.array(times), np.array(values)


def save_combined_csv(filepath, times, values, sampling_rate):
    """Save combined signal in the same CSV format as other signals."""
    with open(filepath, "w") as f:
        f.write(f"Time (s),a_combined,Sampling Rate: {sampling_rate} Hz\n")
        for t, v in zip(times, values):
            f.write(f"{t},{v}\n")


def process_recording(recording_path, dry_run=False, overwrite=False):
    """Process a single recording folder: combine ax/ay/az and remove gravity."""
    # Check if all axis files exist
    axis_paths = []
    for axis_file in AXIS_FILES:
        path = os.path.join(recording_path, axis_file)
        if not os.path.isfile(path):
            return None  # missing axis file, skip silently
        axis_paths.append(path)

    output_path = os.path.join(recording_path, OUTPUT_FILENAME)

    if os.path.isfile(output_path) and not overwrite:
        return "skipped"

    if dry_run:
        print(f"  [DRY RUN] Would create: {output_path}")
        return "dry_run"

    # Load all three axes
    times_x, ax = load_axis_csv(axis_paths[0])
    times_y, ay = load_axis_csv(axis_paths[1])
    times_z, az = load_axis_csv(axis_paths[2])

    # Verify same length
    min_len = min(len(ax), len(ay), len(az))
    if len(ax) != len(ay) or len(ay) != len(az):
        print(f"  WARNING: Axis length mismatch in {recording_path}: "
              f"ax={len(ax)}, ay={len(ay)}, az={len(az)}. Truncating to {min_len}.")
        ax = ax[:min_len]
        ay = ay[:min_len]
        az = az[:min_len]
        times_x = times_x[:min_len]

    # Compute magnitude and remove gravity
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    gravity_removed = magnitude - GRAVITY

    save_combined_csv(output_path, times_x, gravity_removed, SAMPLING_RATE)
    return "created"


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return

    print(f"Dataset root: {dataset_root}")
    print(f"Output filename: {OUTPUT_FILENAME}")
    print(f"Gravity constant: {GRAVITY} m/s²")
    if args.dry_run:
        print("MODE: Dry run (no files will be written)")
    print("=" * 60)

    counts = {"created": 0, "skipped": 0, "dry_run": 0, "missing": 0}

    for exercise in sorted(os.listdir(dataset_root)):
        exercise_path = os.path.join(dataset_root, exercise)
        if not os.path.isdir(exercise_path):
            continue

        for person in sorted(os.listdir(exercise_path)):
            person_path = os.path.join(exercise_path, person)
            if not os.path.isdir(person_path):
                continue

            for recording in sorted(os.listdir(person_path)):
                recording_path = os.path.join(person_path, recording)
                if not os.path.isdir(recording_path):
                    continue

                result = process_recording(
                    recording_path,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite
                )

                label = f"{exercise}/{person}/{recording}"

                if result == "created":
                    print(f"  Created: {label}")
                    counts["created"] += 1
                elif result == "skipped":
                    print(f"  Skipped (exists): {label}")
                    counts["skipped"] += 1
                elif result == "dry_run":
                    counts["dry_run"] += 1
                else:
                    counts["missing"] += 1

    print("=" * 60)
    print("Summary:")
    print(f"  Created:  {counts['created']}")
    print(f"  Skipped:  {counts['skipped']} (already exist, use --overwrite)")
    print(f"  Missing:  {counts['missing']} (no ax/ay/az files)")
    if args.dry_run:
        print(f"  Dry run:  {counts['dry_run']}")
    print("Done.")


if __name__ == "__main__":
    main()
