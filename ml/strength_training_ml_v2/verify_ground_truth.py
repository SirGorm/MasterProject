"""
Script for å verifisere at ground truth labels er korrekte.

Kjør: python verify_ground_truth.py
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from data.preprocessing import DataPreprocessor, JointProcessor
from data.validate_data import DataValidator


def verify_single_session(session_path: Path, exercise: str):
    """Verifiser ground truth for én sesjon."""
    print(f"\n{'='*60}")
    print(f"Verifiserer: {exercise} / {session_path.name}")
    print('='*60)

    preprocessor = DataPreprocessor()
    joint_processor = JointProcessor()

    # Last markers
    markers_path = session_path / 'markers.json'
    if markers_path.exists():
        with open(markers_path, 'r') as f:
            markers = json.load(f)

        marker_list = markers.get('markers', [])
        print(f"\n[MARKERS] Fant {len(marker_list)} markører:")
        for m in marker_list:
            print(f"  - t={m.get('time', 0):.2f}s: {m.get('label', 'unknown')}")

        # Tell reps fra markers
        rep_markers = [m for m in marker_list if m.get('label', '').lower() not in ['start', 'end']]
        print(f"\n  -> Antall rep-markører: {len(rep_markers)}")
    else:
        print("\n[MARKERS] Ingen markers.json funnet!")
        return

    # Last joint data
    joint_path = session_path / 'joint_data.json'
    if joint_path.exists():
        joint_data = joint_processor.load_joint_data(joint_path)
        frames = joint_data.get('frames', [])

        if frames:
            first_ts = frames[0].get('timestamp_usec', 0)
            last_ts = frames[-1].get('timestamp_usec', 0)
            duration = (last_ts - first_ts) / 1e6

            print(f"\n[JOINT DATA]")
            print(f"  Antall frames: {len(frames)}")
            print(f"  Varighet: {duration:.2f} sekunder")
            print(f"  FPS: {len(frames) / duration:.1f}")

            # Test fasedeteksjon på noen vinduer
            print(f"\n[FASEDETEKSJON TEST]")
            window_sec = 2.0
            test_times = [0, 2, 4, 6, 8, 10]

            for t in test_times:
                if t + window_sec > duration:
                    break

                phase = joint_processor.detect_phase(
                    joint_data, t, t + window_sec, exercise
                )
                velocity = joint_processor.calculate_movement_velocity(
                    joint_data, t, t + window_sec, exercise
                )
                print(f"  t={t:.0f}-{t+window_sec:.0f}s: fase={phase:12s}, velocity={velocity:.4f} m/s")

            # Test rep-telling
            start_marker = next(
                (m for m in marker_list if m.get('label', '').lower() == 'start'),
                None
            )
            if start_marker:
                start_time = start_marker.get('time', 0)

                print(f"\n[REP-TELLING TEST] (fra joint peaks)")
                test_end_times = [5, 10, 15, 20, duration]

                for end_t in test_end_times:
                    if end_t > duration:
                        end_t = duration

                    reps = joint_processor.count_reps_from_peaks(
                        joint_data, start_time, end_t, exercise
                    )
                    print(f"  Ved t={end_t:.0f}s: {reps} reps detektert")

                    if end_t >= duration:
                        break

            # Visualiser bevegelse
            print(f"\n[BEVEGELSESANALYSE]")
            visualize_movement(joint_data, exercise, session_path)

    else:
        print("\n[JOINT DATA] Ingen joint_data.json funnet!")


def visualize_movement(joint_data: dict, exercise: str, session_path: Path):
    """Visualiser nøkkelledd-bevegelse over tid."""
    frames = joint_data.get('frames', [])
    if not frames:
        return

    first_ts = frames[0].get('timestamp_usec', 0)

    # Velg nøkkelledd basert på øvelse
    if exercise.lower() == 'squat':
        joint_idx = 0  # PELVIS
        joint_name = 'PELVIS'
    elif exercise.lower() == 'benchpress':
        joint_idx = 7  # WRIST_LEFT
        joint_name = 'WRIST_LEFT'
    elif exercise.lower() == 'pullups':
        joint_idx = 2  # SPINE_CHEST
        joint_name = 'SPINE_CHEST'
    else:
        joint_idx = 0
        joint_name = 'PELVIS'

    times = []
    y_positions = []

    for frame in frames:
        rel_time = (frame.get('timestamp_usec', 0) - first_ts) / 1e6
        bodies = frame.get('bodies', [])

        if bodies:
            joint_positions = bodies[0].get('joint_positions', [])
            if len(joint_positions) > joint_idx:
                times.append(rel_time)
                y_positions.append(joint_positions[joint_idx][1])  # Y-posisjon

    if not times:
        print("  Ingen bevegelsesdata å visualisere")
        return

    # Lag plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Posisjon over tid
    axes[0].plot(times, y_positions, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Tid (sekunder)')
    axes[0].set_ylabel(f'{joint_name} Y-posisjon (m)')
    axes[0].set_title(f'{exercise} - Bevegelse over tid')
    axes[0].grid(True, alpha=0.3)

    # Velocity (derivert)
    if len(times) > 1:
        dt = np.diff(times)
        dy = np.diff(y_positions)
        velocity = dy / np.where(dt > 0, dt, 1e-6)

        # Smooth velocity
        from scipy.ndimage import uniform_filter1d
        velocity_smooth = uniform_filter1d(velocity, size=15)

        axes[1].plot(times[1:], velocity_smooth, 'r-', linewidth=0.5)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Tid (sekunder)')
        axes[1].set_ylabel('Hastighet (m/s)')
        axes[1].set_title('Hastighet (positiv=opp, negativ=ned)')
        axes[1].grid(True, alpha=0.3)

        # Marker faser
        # Positiv velocity = concentric (for de fleste øvelser)
        # Negativ velocity = eccentric
        axes[1].fill_between(
            times[1:], velocity_smooth, 0,
            where=velocity_smooth > 0,
            alpha=0.3, color='green', label='Concentric'
        )
        axes[1].fill_between(
            times[1:], velocity_smooth, 0,
            where=velocity_smooth < 0,
            alpha=0.3, color='red', label='Eccentric'
        )
        axes[1].legend()

    plt.tight_layout()

    # Lagre plot
    output_dir = CONFIG.output.output_dir / "ground_truth_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"{exercise}_{session_path.name}_movement.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  Lagret bevegelsesplot: {plot_path}")


def verify_phase_distribution(windows: list):
    """Vis distribusjon av faser."""
    phases = [w.get('phase', 'unknown') for w in windows]

    from collections import Counter
    phase_counts = Counter(phases)

    print(f"\n[FASE-DISTRIBUSJON]")
    total = len(phases)
    for phase, count in sorted(phase_counts.items()):
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {phase:12s}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    """Hovedfunksjon for verifisering."""
    print("="*60)
    print("GROUND TRUTH VERIFISERING")
    print("="*60)

    # Valider datasett først
    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    validator.validate_all()
    valid_sessions = validator.get_valid_sessions()

    if not valid_sessions:
        print("\nIngen gyldige sesjoner funnet!")
        return

    print(f"\nFant {len(valid_sessions)} gyldige sesjoner")

    # Verifiser første sesjon av hver øvelse
    verified_exercises = set()
    for exercise, session_id, session_path in valid_sessions:
        if exercise not in verified_exercises:
            verify_single_session(session_path, exercise)
            verified_exercises.add(exercise)

    # Preprosesser og sjekk distribusjon
    print("\n" + "="*60)
    print("PREPROSESSERER ALLE VINDUER...")
    print("="*60)

    preprocessor = DataPreprocessor()
    all_windows = []

    for exercise, session_id, session_path in valid_sessions[:5]:  # Begrens til 5 sesjoner
        session_data = preprocessor.preprocess_session(session_path, exercise)
        windows = preprocessor.create_windows(session_data, exercise)

        for w in windows:
            w['exercise'] = exercise

        all_windows.extend(windows)
        print(f"  {exercise}/{session_id}: {len(windows)} vinduer")

    print(f"\nTotalt: {len(all_windows)} vinduer")

    # Vis distribusjon
    verify_phase_distribution(all_windows)

    # Vis rep-telling statistikk
    rep_counts = [w.get('rep_count', 0) for w in all_windows]
    print(f"\n[REP-TELLING STATISTIKK]")
    print(f"  Min: {min(rep_counts)}")
    print(f"  Max: {max(rep_counts)}")
    print(f"  Gjennomsnitt: {np.mean(rep_counts):.1f}")

    # Vis fatigue statistikk
    fatigue_scores = [w.get('fatigue_score', 0) for w in all_windows]
    print(f"\n[FATIGUE STATISTIKK]")
    print(f"  Min: {min(fatigue_scores):.3f}")
    print(f"  Max: {max(fatigue_scores):.3f}")
    print(f"  Gjennomsnitt: {np.mean(fatigue_scores):.3f}")

    print("\n" + "="*60)
    print("VERIFISERING FULLFØRT")
    print(f"Se visualiseringer i: {CONFIG.output.output_dir / 'ground_truth_verification'}")
    print("="*60)


if __name__ == "__main__":
    main()
