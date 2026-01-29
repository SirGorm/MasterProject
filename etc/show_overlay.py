import os
import cv2
import json
import numpy as np

# --- Brukervalg ---
# Update these paths to match your dataset location
DATASET_ROOT = r"C:\Users\skogl\Downloads\eirikgsk\MasterProject\data"
EXERCISE_NAME = "Pullups"
RECORDING_ID = "003"

def recording_folder_name(rec_id):
    return f"recording_{int(rec_id):03d}"

# Files are directly in exercise folder: data/Squat/recording_001.mkv
recording_folder = os.path.join(DATASET_ROOT, EXERCISE_NAME)
video_path = os.path.join(recording_folder, f"{recording_folder_name(RECORDING_ID)}.mkv")
json_path = os.path.join(recording_folder, "joint_data.json")

if not os.path.isfile(video_path) or not os.path.isfile(json_path):
    raise FileNotFoundError("Video eller JSON fil finnes ikke!")

# --- Last JSON ---
with open(json_path) as f:
    data = json.load(f)

joint_names = data["joint_names"]
bone_list = data["bone_list"]
frames = data["frames"]

# Sjekk hvilke data som finnes
print("=" * 60)
print("JSON Data Info:")
print("=" * 60)
if frames and frames[0]["bodies"]:
    sample = frames[0]["bodies"][0]
    print("Tilgjengelige data i bodies[0]:")
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

# --- JUSTERBARE PARAMETERE MED PERSISTENS ---
settings_file = os.path.join(recording_folder, "skeleton_settings.json")

# Last inn lagrede innstillinger hvis de finnes
if os.path.exists(settings_file):
    with open(settings_file, 'r') as f:
        settings = json.load(f)
        scale = settings.get('scale', 0.5)
        offset_x = settings.get('offset_x', 0)
        offset_y = settings.get('offset_y', 0)
        flip_x = settings.get('flip_x', False)
        flip_y = settings.get('flip_y', False)
    print(f"✅ Lastet lagrede innstillinger:")
    print(f"   Scale: {scale:.2f}, Offset: ({offset_x}, {offset_y})")
    print(f"   Flip X: {flip_x}, Flip Y: {flip_y}")
else:
    scale = 0.5  # Start med 50% skala
    offset_x = 0
    offset_y = 0
    flip_x = False
    flip_y = False
    print("📝 Bruker standard innstillinger (ingen lagrede funnet)")

def save_settings():
    """Lagre innstillinger til fil"""
    settings = {
        'scale': scale,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'flip_x': flip_x,
        'flip_y': flip_y
    }
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"💾 Innstillinger lagret: Scale={scale:.2f}, Offset=({offset_x}, {offset_y}), Flip X={flip_x}, Flip Y={flip_y}")

frame_idx = 0
paused = False

# Prøv å finne beste 2D koordinater
def get_2d_joints(body, scale_factor=1.0, offset_x=0, offset_y=0, flip_x=False, flip_y=False):
    """Henter 2D koordinater med justerbar skala, offset og flip"""
    
    # 1. Sjekk color_positions (Azure Kinect color space)
    if "color_positions" in body:
        joints = np.array(body["color_positions"])
        
        # Finn senter av skjelettet (ikke bildet!)
        skeleton_center_x = np.mean(joints[:, 0])
        skeleton_center_y = np.mean(joints[:, 1])
        
        scaled_joints = []
        for x, y in joints:
            # Skaler rundt skjelett-senter
            x_centered = x - skeleton_center_x
            y_centered = y - skeleton_center_y
            
            # Flip hvis nødvendig
            if flip_x:
                x_centered = -x_centered
            if flip_y:
                y_centered = -y_centered
            
            # Skaler
            x_scaled = x_centered * scale_factor
            y_scaled = y_centered * scale_factor
            
            # Flytt til bilde-senter + offset
            x_final = x_scaled + w // 2 + offset_x
            y_final = y_scaled + h // 2 + offset_y
            
            scaled_joints.append([int(x_final), int(y_final)])
        return scaled_joints
    
    # 2. Sjekk joint_positions_2d
    if "joint_positions_2d" in body:
        joints = np.array(body["joint_positions_2d"])
        
        skeleton_center_x = np.mean(joints[:, 0])
        skeleton_center_y = np.mean(joints[:, 1])
        
        scaled_joints = []
        for x, y in joints:
            x_centered = x - skeleton_center_x
            y_centered = y - skeleton_center_y
            
            if flip_x:
                x_centered = -x_centered
            if flip_y:
                y_centered = -y_centered
            
            x_scaled = x_centered * scale_factor
            y_scaled = y_centered * scale_factor
            
            x_final = x_scaled + w // 2 + offset_x
            y_final = y_scaled + h // 2 + offset_y
            
            scaled_joints.append([int(x_final), int(y_final)])
        return scaled_joints
    
    # 3. Fallback til 3D projeksjon
    if "joint_positions" in body:
        joints_3d = np.array(body["joint_positions"])
        
        all_x = joints_3d[:, 0]
        all_y = joints_3d[:, 1]
        
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
            
            u = int(x_norm * (w - 2*margin) * scale_factor + margin + offset_x)
            v = int(y_norm * (h - 2*margin) * scale_factor + margin + offset_y)
            joints_2d.append([u, v])
        
        return joints_2d
    
    return None

# --- Hovedløkke ---
print("\n🎮 KONTROLLER:")
print("  SPACE    - Pause/Play")
print("  Q/E      - Forrige/Neste frame (når pauset)")
print("  +/-      - Øk/Reduser størrelse")
print("  W/A/S/D  - Flytt skjelett (Opp/Venstre/Ned/Høyre)")
print("  X        - Flip X-akse (speil horisontalt)")
print("  Y        - Flip Y-akse (speil vertikalt)")
print("  R        - Reset (skala=0.5, offset=0)")
print("  L        - Lagre innstillinger")
print("  P        - Screenshot")
print("  ESC      - Avslutt")
print("=" * 60 + "\n")

current_frame = None
needs_redraw = True

while cap.isOpened() and frame_idx < len(frames):
    # Les ny frame hvis ikke pauset eller hvis vi trenger redraw
    if not paused or needs_redraw:
        if not paused:
            ret, current_frame = cap.read()
            if not ret:
                break
        elif current_frame is None:
            # Hvis pauset men ingen frame er lastet, last den
            ret, current_frame = cap.read()
            if not ret:
                break
        
        # Lag en kopi for å tegne på
        frame = current_frame.copy()
        
        frame_data = frames[frame_idx]
        bodies = frame_data["bodies"]
        
        if bodies:
            body = bodies[0]
            joints_2d = get_2d_joints(body, scale, offset_x, offset_y, flip_x, flip_y)
            
            if joints_2d:
                joints_2d = np.array(joints_2d)
                
                # Tegn bones
                for bone_a, bone_b in bone_list:
                    try:
                        i = joint_names.index(bone_a)
                        j = joint_names.index(bone_b)
                        pt1 = tuple(joints_2d[i])
                        pt2 = tuple(joints_2d[j])
                        
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                    except (ValueError, IndexError):
                        pass

                # Tegn joints
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
        
        cv2.imshow(f"{EXERCISE_NAME} - Recording {RECORDING_ID}", frame)
        needs_redraw = False
        
        if not paused:
            frame_idx += 1

    # Kontroller
    key = cv2.waitKey(int(1000 / fps) if not paused else 10) & 0xFF
    
    if key == 27:  # ESC
        save_settings()  # Auto-lagre ved avslutning
        break
    elif key == ord(' '):  # Space
        paused = not paused
    elif key == ord('q'):  # Q - tilbake
        if paused and frame_idx > 0:
            frame_idx -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame = None
            needs_redraw = True
    elif key == ord('e'):  # E - frem
        if paused and frame_idx < len(frames) - 1:
            frame_idx += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_frame = None
            needs_redraw = True
    elif key == ord('+') or key == ord('='):  # Øk skala
        scale += 0.05
        print(f"Scale: {scale:.2f}")
        if paused:
            needs_redraw = True
    elif key == ord('-') or key == ord('_'):  # Reduser skala
        scale = max(0.1, scale - 0.05)
        print(f"Scale: {scale:.2f}")
        if paused:
            needs_redraw = True
    elif key == ord('w'):  # W - Opp
        offset_y -= 10
        print(f"Offset: ({offset_x}, {offset_y})")
        if paused:
            needs_redraw = True
    elif key == ord('s'):  # S - Ned
        offset_y += 10
        print(f"Offset: ({offset_x}, {offset_y})")
        if paused:
            needs_redraw = True
    elif key == ord('a'):  # A - Venstre
        offset_x -= 10
        print(f"Offset: ({offset_x}, {offset_y})")
        if paused:
            needs_redraw = True
    elif key == ord('d'):  # D - Høyre
        offset_x += 10
        print(f"Offset: ({offset_x}, {offset_y})")
        if paused:
            needs_redraw = True
    elif key == ord('r'):  # Reset
        scale = 0.5
        offset_x = 0
        offset_y = 0
        flip_x = False
        flip_y = False
        print(f"Reset - Scale: {scale:.2f}, Offset: ({offset_x}, {offset_y}), Flip X: {flip_x}, Flip Y: {flip_y}")
        if paused:
            needs_redraw = True
    elif key == ord('x'):  # Flip X-akse
        flip_x = not flip_x
        print(f"Flip X: {'ON' if flip_x else 'OFF'}")
        if paused:
            needs_redraw = True
    elif key == ord('y'):  # Flip Y-akse
        flip_y = not flip_y
        print(f"Flip Y: {'ON' if flip_y else 'OFF'}")
        if paused:
            needs_redraw = True
    elif key == ord('l'):  # Lagre innstillinger
        save_settings()
    elif key == ord('p'):  # Screenshot
        screenshot_path = f"screenshot_frame_{frame_idx:04d}.png"
        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")

cap.release()
cv2.destroyAllWindows()

# Lagre innstillinger automatisk
save_settings()

print("\n✅ Ferdig! Innstillinger lagret:")
print(f"   Scale: {scale:.2f}")
print(f"   Offset: ({offset_x}, {offset_y})")
print(f"   Flip X: {flip_x}, Flip Y: {flip_y}")
print(f"   Fil: {settings_file}")