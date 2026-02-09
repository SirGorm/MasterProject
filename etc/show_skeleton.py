"""
Playback joint data from dataset using Open3D.
Usage example:
python show_skeleton.py -ex (Exercise) -p (person id) -s (set id) -d (dataset root / optional)
"""
import os
import json
import argparse
import open3d as o3d
import numpy as np
import time
import re

# ----------------------------------------------------
# Argument parsing
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Playback joint data")
    parser.add_argument("-ex", "--exercise", type=str, required=True, help="Exercise name (e.g. Squat)")
    parser.add_argument("-p", "--person", type=int, required=True, help="Person number (e.g. 1)")
    parser.add_argument("-s", "--set", dest="set_id", type=int, required=True, help="Set number (e.g. 1)")
    parser.add_argument("-d", "--dataset_root", type=str, default=None, help="Dataset root path")
    return parser.parse_args()

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def extract_number(name):
    """Extract first number found in a string"""
    match = re.search(r"\d+", name)
    return int(match.group()) if match else None

def find_matching_folder(parent, target_number):
    """
    Find folder in parent whose name contains target_number
    (case-insensitive, flexible naming)
    """
    for folder in os.listdir(parent):
        folder_path = os.path.join(parent, folder)
        if not os.path.isdir(folder_path):
            continue

        number = extract_number(folder)
        if number == target_number:
            return folder_path

    return None

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    args = parse_args()

    DATASET_ROOT = args.dataset_root or r"C:\Users\skogl\Downloads\eirikgsk\MasterProject\data"

    exercise_path = os.path.join(DATASET_ROOT, args.exercise)
    if not os.path.isdir(exercise_path):
        raise RuntimeError(f"Exercise not found: {exercise_path}")

    person_path = find_matching_folder(exercise_path, args.person)
    if not person_path:
        raise RuntimeError(f"Person {args.person} not found under {exercise_path}")

    set_path = find_matching_folder(person_path, args.set_id)
    if not set_path:
        raise RuntimeError(f"Set {args.set_id} not found under {person_path}")

    json_path = os.path.join(set_path, "joint_data.json")
    if not os.path.isfile(json_path):
        raise RuntimeError(f"joint_data.json not found in {set_path}")

    print(f"Playing back: {json_path}")

    # ----------------------------------------------------
    # Open3D playback
    # ----------------------------------------------------
    USE_TIMESTAMP = True
    PAUSE_TIME = 0.05

    with open(json_path) as f:
        data = json.load(f)

    joint_names = data["joint_names"]
    bone_list = data["bone_list"]
    frames = data["frames"]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    points = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()

    geometry_added = False
    prev_timestamp = None

    for frame in frames:
        bodies = frame.get("bodies", [])
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

        if USE_TIMESTAMP:
            timestamp = frame["timestamp_usec"] / 1_000_000
            if prev_timestamp is not None:
                delay = timestamp - prev_timestamp
                if delay > 0:
                    time.sleep(delay)
            prev_timestamp = timestamp
        else:
            time.sleep(PAUSE_TIME)

    vis.destroy_window()

if __name__ == "__main__":
    main()
