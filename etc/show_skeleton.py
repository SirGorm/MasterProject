"""
3D skeleton playback using Open3D from Azure Kinect joint data.

Usage:
    python show_skeleton.py --exercise Squat --person Person01 --recording 001
    python show_skeleton.py --dataset C:\\path\\to\\data --exercise Warmup
    python show_skeleton.py --exercise Squat  # plays all persons/recordings
"""

import os
import json
import argparse
import open3d as o3d
import numpy as np
import time


def parse_args():
    default_dataset = os.environ.get(
        "STRENGTH_DATASET_PATH",
        os.path.join(os.path.dirname(__file__), "..", "dataset")
    )
    parser = argparse.ArgumentParser(
        description="3D skeleton playback using Open3D"
    )
    parser.add_argument("--dataset", "-d", type=str, default=default_dataset,
                        help="Root directory of dataset")
    parser.add_argument("--exercise", "-e", type=str, default="Warmup",
                        help="Exercise name (or None for all)")
    parser.add_argument("--person", "-p", type=str, default=None,
                        help="Person name (or omit for all)")
    parser.add_argument("--recording", "-r", type=str, default=None,
                        help="Recording ID (or omit for all)")
    parser.add_argument("--no-timestamp", action="store_true",
                        help="Use fixed pause instead of timestamps")
    parser.add_argument("--pause", type=float, default=0.05,
                        help="Pause time in seconds (when --no-timestamp)")
    return parser.parse_args()


def recording_folder_name(rec_id):
    return f"recording_{int(rec_id):03d}"


def main():
    args = parse_args()

    use_timestamp = not args.no_timestamp
    pause_time = args.pause
    recording_name = recording_folder_name(args.recording) if args.recording else None

    # Find chosen joint data
    json_files = []

    for exercise in os.listdir(args.dataset):
        if args.exercise and exercise != args.exercise:
            continue

        exercise_path = os.path.join(args.dataset, exercise)
        if not os.path.isdir(exercise_path):
            continue

        for person in os.listdir(exercise_path):
            if args.person and person != args.person:
                continue

            person_path = os.path.join(exercise_path, person)
            if not os.path.isdir(person_path):
                continue

            for recording in os.listdir(person_path):
                if recording_name and recording != recording_name:
                    continue

                recording_path = os.path.join(person_path, recording)
                json_path = os.path.join(recording_path, "joint_data.json")

                if os.path.isfile(json_path):
                    json_files.append(json_path)

    if not json_files:
        raise RuntimeError("No joint_data.json found for chosen exercise/person/recording")

    print(f"Found {len(json_files)} files")

    # --- Open3D visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    points = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()

    prev_timestamp = None
    geometry_added = False

    # --- Play back all found files ---
    for file_path in json_files:

        print(f"\nPlayback of file: {file_path}")

        with open(file_path) as f:
            data = json.load(f)

        joint_names = data["joint_names"]
        bone_list = data["bone_list"]
        frames = data["frames"]

        prev_timestamp = None

        for frame in frames:
            bodies = frame["bodies"]
            if not bodies:
                continue

            joints = bodies[0]["joint_positions"]
            pts = np.array(joints)
            pts[:, 1] *= -1

            points.points = o3d.utility.Vector3dVector(pts)

            line_indices = []
            for a, b in bone_list:
                i = joint_names.index(a)
                j = joint_names.index(b)
                line_indices.append([i, j])

            lines.points = points.points
            lines.lines = o3d.utility.Vector2iVector(line_indices)

            if not geometry_added:
                vis.add_geometry(lines)
                geometry_added = True

            vis.update_geometry(lines)
            vis.poll_events()
            vis.update_renderer()

            # --- Playback timing ---
            if use_timestamp:
                timestamp = frame["timestamp_usec"] / 1_000_000
                if prev_timestamp is not None:
                    delay = timestamp - prev_timestamp
                    if delay > 0:
                        time.sleep(delay)
                prev_timestamp = timestamp
            else:
                time.sleep(pause_time)

    vis.destroy_window()


if __name__ == "__main__":
    main()
