"""
Overlay skeleton joints on video playback from Azure Kinect recordings.

Interactive controls for adjusting skeleton scale, offset, and flip.

Usage:
    python show_overlay.py --exercise Squat --person Person01 --recording 001
    python show_overlay.py --dataset C:\\path\\to\\data --exercise Warmup
"""

import os
import cv2
import json
import argparse
import numpy as np


def parse_args():
    default_dataset = os.environ.get(
        "STRENGTH_DATASET_PATH",
        os.path.join(os.path.dirname(__file__), "..", "dataset")
    )
    parser = argparse.ArgumentParser(
        description="Overlay skeleton on video playback"
    )
    parser.add_argument("--dataset", "-d", type=str, default=default_dataset,
                        help="Root directory of dataset")
    parser.add_argument("--exercise", "-e", type=str, default="Warmup",
                        help="Exercise name")
    parser.add_argument("--person", "-p", type=str, default="person1",
                        help="Person name")
    parser.add_argument("--recording", "-r", type=str, default="001",
                        help="Recording ID")
    return parser.parse_args()


def recording_folder_name(rec_id):
    return f"recording_{int(rec_id):03d}"


def _transform_joints_2d(joints, scale_factor, offset_x, offset_y,
                          flip_x, flip_y, center_x, center_y):
    """Transform 2D joints with scale, offset and flip around a center point."""
    scaled_joints = []
    for x, y in joints:
        x_centered = x - center_x
        y_centered = y - center_y

        if flip_x:
            x_centered = -x_centered
        if flip_y:
            y_centered = -y_centered

        x_final = x_centered * scale_factor + center_x + offset_x
        y_final = y_centered * scale_factor + center_y + offset_y

        scaled_joints.append([int(x_final), int(y_final)])
    return scaled_joints


def get_2d_joints(body, scale_factor, offset_x, offset_y,
                  flip_x, flip_y, frame_w, frame_h):
    """Get 2D coordinates with adjustable scale, offset and flip."""

    # 1. Check color_positions (Azure Kinect color space)
    if "color_positions" in body:
        joints = np.array(body["color_positions"])
        cx, cy = np.mean(joints[:, 0]), np.mean(joints[:, 1])
        return _transform_joints_2d(
            joints, scale_factor, offset_x, offset_y,
            flip_x, flip_y, frame_w // 2, frame_h // 2
        )

    # 2. Check joint_positions_2d
    if "joint_positions_2d" in body:
        joints = np.array(body["joint_positions_2d"])
        return _transform_joints_2d(
            joints, scale_factor, offset_x, offset_y,
            flip_x, flip_y, frame_w // 2, frame_h // 2
        )

    # 3. Fallback to 3D projection
    if "joint_positions" in body:
        joints_3d = np.array(body["joint_positions"])
        all_x, all_y = joints_3d[:, 0], joints_3d[:, 1]
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        joints_2d = []
        margin = 100
        for x, y, z in joints_3d:
            x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0.5
            y_norm = (y - y_min) / (y_max - y_min) if y_max != y_min else 0.5

            if flip_x:
                x_norm = 1 - x_norm
            if flip_y:
                y_norm = 1 - y_norm

            u = int(x_norm * (frame_w - 2 * margin) * scale_factor + margin + offset_x)
            v = int(y_norm * (frame_h - 2 * margin) * scale_factor + margin + offset_y)
            joints_2d.append([u, v])

        return joints_2d

    return None


def main():
    args = parse_args()

    recording_folder = os.path.join(
        args.dataset, args.exercise, args.person,
        recording_folder_name(args.recording)
    )

    # Find the .mkv video file inside the recording folder
    video_path = None
    for f in os.listdir(recording_folder):
        if f.endswith(".mkv"):
            video_path = os.path.join(recording_folder, f)
            break

    json_path = os.path.join(recording_folder, "joint_data.json")

    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"No .mkv video found in {recording_folder}")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"joint_data.json not found in {recording_folder}")

    print(f"Video: {video_path}")
    print(f"JSON:  {json_path}")

    # --- Load JSON ---
    with open(json_path) as f:
        data = json.load(f)

    joint_names = data["joint_names"]
    bone_list = data["bone_list"]
    frames = data["frames"]

    # Check available data
    print("=" * 60)
    print("JSON Data Info:")
    print("=" * 60)
    if frames and frames[0]["bodies"]:
        sample = frames[0]["bodies"][0]
        print("Available data in bodies[0]:")
        for key in sample.keys():
            print(f"  - {key}")
    print("=" * 60)

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"Video: {w}x{h}, {fps} fps")
    print("=" * 60)

    # --- Adjustable parameters with persistence ---
    settings_file = os.path.join(recording_folder, "skeleton_settings.json")

    if os.path.exists(settings_file):
        with open(settings_file, 'r') as f:
            settings = json.load(f)
            scale = settings.get('scale', 0.5)
            offset_x = settings.get('offset_x', 0)
            offset_y = settings.get('offset_y', 0)
            flip_x = settings.get('flip_x', False)
            flip_y = settings.get('flip_y', False)
        print(f"Loaded saved settings:")
        print(f"   Scale: {scale:.2f}, Offset: ({offset_x}, {offset_y})")
        print(f"   Flip X: {flip_x}, Flip Y: {flip_y}")
    else:
        scale = 0.5
        offset_x = 0
        offset_y = 0
        flip_x = False
        flip_y = False
        print("Using default settings (no saved settings found)")

    def save_settings():
        nonlocal scale, offset_x, offset_y, flip_x, flip_y
        settings = {
            'scale': scale,
            'offset_x': offset_x,
            'offset_y': offset_y,
            'flip_x': flip_x,
            'flip_y': flip_y
        }
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Settings saved: Scale={scale:.2f}, Offset=({offset_x}, {offset_y}), Flip X={flip_x}, Flip Y={flip_y}")

    frame_idx = 0
    paused = False

    # --- Main loop ---
    print("\nCONTROLS:")
    print("  SPACE    - Pause/Play")
    print("  Q/E      - Previous/Next frame (when paused)")
    print("  +/-      - Increase/Decrease size")
    print("  W/A/S/D  - Move skeleton (Up/Left/Down/Right)")
    print("  X        - Flip X-axis (mirror horizontally)")
    print("  Y        - Flip Y-axis (mirror vertically)")
    print("  R        - Reset (scale=0.5, offset=0)")
    print("  L        - Save settings")
    print("  P        - Screenshot")
    print("  ESC      - Quit")
    print("=" * 60 + "\n")

    current_frame = None
    needs_redraw = True

    while cap.isOpened() and frame_idx < len(frames):
        if not paused or needs_redraw:
            if not paused:
                ret, current_frame = cap.read()
                if not ret:
                    break
            elif current_frame is None:
                ret, current_frame = cap.read()
                if not ret:
                    break

            frame = current_frame.copy()

            frame_data = frames[frame_idx]
            bodies = frame_data["bodies"]

            if bodies:
                body = bodies[0]
                joints_2d = get_2d_joints(body, scale, offset_x, offset_y,
                                          flip_x, flip_y, w, h)

                if joints_2d:
                    joints_2d = np.array(joints_2d)

                    for bone_a, bone_b in bone_list:
                        try:
                            i = joint_names.index(bone_a)
                            j = joint_names.index(bone_b)
                            pt1 = tuple(joints_2d[i])
                            pt2 = tuple(joints_2d[j])
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                        except (ValueError, IndexError):
                            pass

                    for idx, p in enumerate(joints_2d):
                        cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)
                        cv2.circle(frame, tuple(p), 7, (255, 255, 255), 2)

            # Info overlay
            info_y = 30
            cv2.putText(frame, f"Frame: {frame_idx}/{len(frames)}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"Scale: {scale:.2f}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"Offset: ({offset_x}, {offset_y})", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += 25
            cv2.putText(frame, f"Flip: X={'ON' if flip_x else 'OFF'} Y={'ON' if flip_y else 'OFF'}", (10, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if paused:
                cv2.putText(frame, "PAUSED", (10, info_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(f"{args.exercise} / {args.person} / recording_{args.recording}", frame)
            needs_redraw = False

            if not paused:
                frame_idx += 1

        key = cv2.waitKey(int(1000 / fps) if not paused else 10) & 0xFF

        if key == 27:  # ESC
            save_settings()
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('q'):
            if paused and frame_idx > 0:
                frame_idx -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_frame = None
                needs_redraw = True
        elif key == ord('e'):
            if paused and frame_idx < len(frames) - 1:
                frame_idx += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                current_frame = None
                needs_redraw = True
        elif key == ord('+') or key == ord('='):
            scale += 0.05
            print(f"Scale: {scale:.2f}")
            if paused:
                needs_redraw = True
        elif key == ord('-') or key == ord('_'):
            scale = max(0.1, scale - 0.05)
            print(f"Scale: {scale:.2f}")
            if paused:
                needs_redraw = True
        elif key == ord('w'):
            offset_y -= 10
            print(f"Offset: ({offset_x}, {offset_y})")
            if paused:
                needs_redraw = True
        elif key == ord('s'):
            offset_y += 10
            print(f"Offset: ({offset_x}, {offset_y})")
            if paused:
                needs_redraw = True
        elif key == ord('a'):
            offset_x -= 10
            print(f"Offset: ({offset_x}, {offset_y})")
            if paused:
                needs_redraw = True
        elif key == ord('d'):
            offset_x += 10
            print(f"Offset: ({offset_x}, {offset_y})")
            if paused:
                needs_redraw = True
        elif key == ord('r'):
            scale = 0.5
            offset_x = 0
            offset_y = 0
            flip_x = False
            flip_y = False
            print(f"Reset - Scale: {scale:.2f}, Offset: ({offset_x}, {offset_y})")
            if paused:
                needs_redraw = True
        elif key == ord('x'):
            flip_x = not flip_x
            print(f"Flip X: {'ON' if flip_x else 'OFF'}")
            if paused:
                needs_redraw = True
        elif key == ord('y'):
            flip_y = not flip_y
            print(f"Flip Y: {'ON' if flip_y else 'OFF'}")
            if paused:
                needs_redraw = True
        elif key == ord('l'):
            save_settings()
        elif key == ord('p'):
            screenshot_path = f"screenshot_frame_{frame_idx:04d}.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")

    cap.release()
    cv2.destroyAllWindows()

    save_settings()

    print(f"\nDone! Settings saved to {settings_file}")


if __name__ == "__main__":
    main()
