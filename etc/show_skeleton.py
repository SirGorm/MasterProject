import os
import json
import open3d as o3d
import numpy as np
import time

def recording_folder_name(rec_id):
    return f"recording_{int(rec_id):03d}"

# --- Playback settings ---
USE_TIMESTAMP = True
PAUSE_TIME = 0.05

# --- Dataset selection ---
# Update these paths to match your dataset location
DATASET_ROOT = r"C:\Users\skogl\Downloads\eirikgsk\MasterProject\data"
EXERCISE_NAME = "Squat"                              # None = all exercises
RECORDING_ID = "001"
RECORDING_NAME = recording_folder_name(RECORDING_ID) # None = all recordings

       

# Find chosen joint data
json_files = []

for exercise in os.listdir(DATASET_ROOT):
    if EXERCISE_NAME and exercise != EXERCISE_NAME:
        continue

    exercise_path = os.path.join(DATASET_ROOT, exercise)
    if not os.path.isdir(exercise_path):
        continue

    for recording in os.listdir(exercise_path):
        if RECORDING_NAME and recording != RECORDING_NAME:
            continue

        recording_path = os.path.join(exercise_path, recording)
        json_path = os.path.join(recording_path, "joint_data.json")

        if os.path.isfile(json_path):
            json_files.append(json_path)

if not json_files:
    raise RuntimeError("No joint_data.json for chosen participant or excersie")

print(f"Found {len(json_files)} files")

# --- Open3D visualizer ---
vis = o3d.visualization.Visualizer()
vis.create_window()

points = o3d.geometry.PointCloud()
lines = o3d.geometry.LineSet()

prev_timestamp = None
geometry_added = False

# --- Spill av alle funnede filer ---
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



