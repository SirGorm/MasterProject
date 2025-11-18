import json
import open3d as o3d
import numpy as np
import time

# --- Innstillinger ---
USE_TIMESTAMP = True  # True = playback i samme hastighet som opptaket
PAUSE_TIME = 0.05     # Hvis USE_TIMESTAMP=False, pause mellom frames (sekunder)

# --- Last JSON ---
with open("001.json") as f:
    data = json.load(f)

joint_names = data["joint_names"]
bone_list = data["bone_list"]
frames = data["frames"]

# --- Initialiser Open3D visualizer ---
vis = o3d.visualization.Visualizer()
vis.create_window()
points = o3d.geometry.PointCloud()
lines = o3d.geometry.LineSet()

prev_timestamp = None

for frame in frames:
    bodies = frame["bodies"]
    if len(bodies) == 0:
        continue

    joints = bodies[0]["joint_positions"]
    pts = np.array(joints)
    pts[:, 1] *= -1 # Speil Y for å vise skeleton riktig vei
    points.points = o3d.utility.Vector3dVector(pts)

    line_indices = []
    for a, b in bone_list:
        i = joint_names.index(a)
        j = joint_names.index(b)
        line_indices.append([i, j])

    lines.points = points.points
    lines.lines = o3d.utility.Vector2iVector(line_indices)

    vis.add_geometry(lines)
    vis.poll_events()
    vis.update_renderer()

    # --- Playback kontroll ---
    if USE_TIMESTAMP:
        timestamp = frame["timestamp_usec"] / 1_000_000  # microsek -> sekunder
        if prev_timestamp is not None:
            delay = timestamp - prev_timestamp
            if delay > 0:
                time.sleep(delay)
        prev_timestamp = timestamp
    else:
        time.sleep(PAUSE_TIME)

    vis.remove_geometry(lines)

vis.destroy_window()
