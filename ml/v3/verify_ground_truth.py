"""
Script for å verifisere at ground truth labels er korrekte.

En rep = 1 eccentric (ned til valley) + 1 concentric (opp til peak).
Markers: valley = bunn av bevegelse, peak = topp av bevegelse.
Antall valleys ≈ antall reps.

Kjør: python verify_ground_truth.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy import signal as scipy_signal

from ml.v3.config import CONFIG
from ml.v3.data.preprocessing import DataPreprocessor, JointProcessor
from ml.v3.data.validate_data import DataValidator


def count_reps_from_markers(marker_list: list) -> dict:
    """Tell reps fra markers. 1 rep = 1 valley (eccentric) + 1 peak (concentric)."""
    valleys = [m for m in marker_list if m.get('label', '').lower() == 'valley']
    peaks = [m for m in marker_list if m.get('label', '').lower() == 'peak']
    start = next((m for m in marker_list if m.get('label', '').lower() == 'start'), None)
    stop = next((m for m in marker_list if m.get('label', '').lower() == 'stop'), None)

    return {
        'valleys': len(valleys),
        'peaks': len(peaks),
        'total_reps': len(valleys),  # Hver valley = 1 eccentric gjennomført = 1 rep
        'complete_reps': len(peaks),  # Hver peak = fullført eccentric + concentric
        'start_time': start.get('time', 0) if start else 0,
        'stop_time': stop.get('time', 0) if stop else 0,
        'valley_times': [m.get('time', 0) for m in valleys],
        'peak_times': [m.get('time', 0) for m in peaks],
    }


def verify_single_session(session_path: Path, exercise: str):
    """Verifiser ground truth for én sesjon."""
    print(f"\n{'='*60}")
    print(f"Verifiserer: {exercise} / {session_path.name}")
    print('='*60)

    joint_processor = JointProcessor()

    # Last markers
    markers_path = session_path / 'markers.json'
    if not markers_path.exists():
        print("\n[MARKERS] Ingen markers.json funnet!")
        return

    with open(markers_path, 'r') as f:
        markers = json.load(f)

    marker_list = markers.get('markers', [])
    rep_info = count_reps_from_markers(marker_list)

    print(f"\n[MARKERS]")
    print(f"  Start: t={rep_info['start_time']:.2f}s")
    print(f"  Stop:  t={rep_info['stop_time']:.2f}s")
    print(f"  Varighet: {rep_info['stop_time'] - rep_info['start_time']:.1f}s")
    print(f"  Valleys (eccentric bunn): {rep_info['valleys']}")
    print(f"  Peaks (concentric topp):  {rep_info['peaks']}")
    print(f"  -> Totalt reps (valleys):   {rep_info['total_reps']}")
    print(f"  -> Fullførte reps (peaks):  {rep_info['complete_reps']}")

    # Vis rep-tidspunkter
    print(f"\n  Rep-sekvens:")
    for i, (v_time, p_time) in enumerate(zip(rep_info['valley_times'], rep_info['peak_times'])):
        ecc_dur = v_time - (rep_info['peak_times'][i-1] if i > 0 else rep_info['start_time'])
        con_dur = p_time - v_time
        print(f"    Rep {i+1}: eccentric={ecc_dur:.2f}s, concentric={con_dur:.2f}s "
              f"(valley={v_time:.2f}s, peak={p_time:.2f}s)")
    # Siste valley uten peak
    if rep_info['valleys'] > rep_info['peaks']:
        last_v = rep_info['valley_times'][-1]
        ecc_dur = last_v - rep_info['peak_times'][-1]
        print(f"    Rep {rep_info['valleys']}: eccentric={ecc_dur:.2f}s, concentric=ufullført "
              f"(valley={last_v:.2f}s)")

    # Last joint data
    joint_path = session_path / 'joint_data.json'
    if not joint_path.exists():
        print("\n[JOINT DATA] Ingen joint_data.json funnet!")
        return

    joint_data = joint_processor.load_joint_data(joint_path)
    frames = joint_data.get('frames', [])

    if not frames:
        print("\n[JOINT DATA] Ingen frames!")
        return

    first_ts = frames[0].get('timestamp_usec', 0)
    last_ts = frames[-1].get('timestamp_usec', 0)
    duration = (last_ts - first_ts) / 1e6

    print(f"\n[JOINT DATA]")
    print(f"  Antall frames: {len(frames)}")
    print(f"  Varighet: {duration:.2f} sekunder")
    if duration > 0:
        print(f"  FPS: {len(frames) / duration:.1f}")
    else:
        print(f"  FPS: ukjent (timestamps syntetisert fra metadata)")

    # Test fasedeteksjon med overlapping vinduer
    print(f"\n[FASEDETEKSJON TEST]")
    window_sec = 2.0
    start_time = rep_info['start_time']
    stop_time = min(rep_info['stop_time'], duration)

    t = start_time
    phases_detected = {'eccentric': 0, 'concentric': 0, 'unknown': 0}
    while t + window_sec <= stop_time:
        phase = joint_processor.detect_phase(joint_data, t, t + window_sec, exercise)
        velocity = joint_processor.calculate_movement_velocity(joint_data, t, t + window_sec, exercise)
        phases_detected[phase] = phases_detected.get(phase, 0) + 1

        # Vis kun noen eksempler
        if t <= start_time + 12:
            print(f"  t={t:.1f}-{t+window_sec:.1f}s: fase={phase:12s}, velocity={velocity:.4f} m/s")
        t += window_sec

    print(f"\n  Fasefordeling i vinduer:")
    total_windows = sum(phases_detected.values())
    for phase, count in sorted(phases_detected.items()):
        pct = 100 * count / total_windows if total_windows > 0 else 0
        print(f"    {phase:12s}: {count:3d} ({pct:.0f}%)")

    # Sammenlign rep-telling: markers vs joint peaks
    print(f"\n[REP-TELLING: MARKERS vs JOINT PEAKS]")
    joint_reps = joint_processor.count_reps_from_peaks(
        joint_data, start_time, stop_time, exercise
    )
    print(f"  Markers (valleys):   {rep_info['total_reps']} reps")
    print(f"  Joint peak detect:   {joint_reps} reps")

    diff = abs(rep_info['total_reps'] - joint_reps)
    if diff <= 1:
        print(f"  -> OK: Forskjell = {diff}")
    else:
        print(f"  -> ADVARSEL: Forskjell = {diff} (bør sjekke bevegelsesplot)")

    # Visualiser bevegelse med markers
    print(f"\n[BEVEGELSESANALYSE]")
    visualize_movement(joint_data, exercise, session_path, rep_info)


def visualize_movement(joint_data: dict, exercise: str, session_path: Path, rep_info: dict):
    """Visualiser nøkkelledd-bevegelse med markers overlay."""
    frames = joint_data.get('frames', [])
    if not frames:
        return

    first_ts = frames[0].get('timestamp_usec', 0)

    # Velg nøkkelledd basert på øvelse (samme mapping som JointProcessor.EXERCISE_JOINT)
    joint_map = {
        'squat': (0, 'PELVIS'),
        'benchpress': (2, 'SPINE_CHEST'),
        'pullup': (2, 'SPINE_CHEST'),
        'pullups': (2, 'SPINE_CHEST'),
        'deadlift': (2, 'SPINE_CHEST'),
    }
    joint_idx, joint_name = joint_map.get(exercise.lower(), (0, 'PELVIS'))

    times = []
    y_positions = []

    for frame in frames:
        rel_time = (frame.get('timestamp_usec', 0) - first_ts) / 1e6
        bodies = frame.get('bodies', [])

        if bodies:
            joint_positions = bodies[0].get('joint_positions', [])
            if len(joint_positions) > joint_idx:
                times.append(rel_time)
                y_positions.append(joint_positions[joint_idx][1])

    if not times:
        print("  Ingen bevegelsesdata å visualisere")
        return

    times = np.array(times)
    y_positions = np.array(y_positions)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # --- Plot 1: Posisjon med markers ---
    axes[0].plot(times, y_positions, 'b-', linewidth=0.8, label=f'{joint_name} Y')

    # Overlay valley og peak markers
    for v_time in rep_info['valley_times']:
        axes[0].axvline(x=v_time, color='red', linestyle='--', alpha=0.6, linewidth=0.8)
    for p_time in rep_info['peak_times']:
        axes[0].axvline(x=p_time, color='green', linestyle='--', alpha=0.6, linewidth=0.8)

    # Start/stop
    axes[0].axvline(x=rep_info['start_time'], color='black', linestyle='-', alpha=0.8, linewidth=1.5, label='Start/Stop')
    axes[0].axvline(x=rep_info['stop_time'], color='black', linestyle='-', alpha=0.8, linewidth=1.5)

    # Dummy entries for legend
    axes[0].axvline(x=-1, color='red', linestyle='--', alpha=0.6, label='Valley (eccentric bunn)')
    axes[0].axvline(x=-1, color='green', linestyle='--', alpha=0.6, label='Peak (concentric topp)')

    axes[0].set_xlabel('Tid (sekunder)')
    axes[0].set_ylabel(f'{joint_name} Y-posisjon (m)')
    axes[0].set_title(f'{exercise} - {session_path.name} - Bevegelse med markers '
                      f'({rep_info["total_reps"]} reps)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(times[0], times[-1])

    # --- Plot 2: Hastighet med faser fargekodet ---
    if len(times) > 1:
        dt = np.diff(times)
        dy = np.diff(y_positions)
        velocity = dy / np.where(dt > 0, dt, 1e-6)
        velocity_smooth = uniform_filter1d(velocity, size=min(15, len(velocity)))

        axes[1].plot(times[1:], velocity_smooth, 'k-', linewidth=0.5, alpha=0.7)
        axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Fargelegg faser (Y synkende = concentric, Y stigende = eccentric)
        axes[1].fill_between(
            times[1:], velocity_smooth, 0,
            where=velocity_smooth < 0,
            alpha=0.4, color='green', label='Concentric (Y synker)'
        )
        axes[1].fill_between(
            times[1:], velocity_smooth, 0,
            where=velocity_smooth > 0,
            alpha=0.4, color='red', label='Eccentric (Y stiger)'
        )

        # Markers
        for v_time in rep_info['valley_times']:
            axes[1].axvline(x=v_time, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
        for p_time in rep_info['peak_times']:
            axes[1].axvline(x=p_time, color='green', linestyle='--', alpha=0.4, linewidth=0.8)

        axes[1].set_xlabel('Tid (sekunder)')
        axes[1].set_ylabel('Hastighet (m/s)')
        axes[1].set_title('Hastighet - grønn=concentric (Y synker), rød=eccentric (Y stiger)')
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(times[0], times[-1])

    plt.tight_layout()

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
        bar = '#' * int(pct / 2)
        print(f"  {phase:12s}: {count:4d} ({pct:5.1f}%) {bar}")


def main():
    """Hovedfunksjon for verifisering."""
    print("="*60)
    print("GROUND TRUTH VERIFISERING")
    print("="*60)
    print("\nDefinisjon: 1 rep = 1 eccentric (valley) + 1 concentric (peak)")

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

    for exercise, session_id, session_path in valid_sessions[:5]:
        session_data = preprocessor.preprocess_session(session_path, exercise)
        windows = preprocessor.create_windows(session_data, exercise)

        for w in windows:
            w['exercise'] = exercise

        all_windows.extend(windows)
        print(f"  {exercise}/{session_id}: {len(windows)} vinduer")

    print(f"\nTotalt: {len(all_windows)} vinduer")

    # Vis distribusjon
    verify_phase_distribution(all_windows)

    # Vis rep-telling statistikk per sesjon
    rep_counts = [w.get('rep_count', 0) for w in all_windows]
    print(f"\n[REP-TELLING STATISTIKK (over alle vinduer)]")
    print(f"  Min: {min(rep_counts)}")
    print(f"  Max: {max(rep_counts)}")
    print(f"  Gjennomsnitt: {np.mean(rep_counts):.1f}")

    # Vis siste vindu per sesjon (bør ha ~10 reps)
    print(f"\n[SISTE VINDU PER SESJON (bør vise ~10 reps)]")
    sessions_seen = set()
    for w in reversed(all_windows):
        key = (w.get('exercise'), w.get('session_id', ''))
        if key not in sessions_seen:
            sessions_seen.add(key)
            print(f"  {w.get('exercise')}/{w.get('session_id', '?')}: "
                  f"{w.get('rep_count', 0)} reps, fatigue={w.get('fatigue_score', 0):.3f}")

    # Vis fatigue statistikk
    fatigue_scores = [w.get('fatigue_score', 0) for w in all_windows]
    print(f"\n[FATIGUE STATISTIKK]")
    print(f"  Min: {min(fatigue_scores):.3f}")
    print(f"  Max: {max(fatigue_scores):.3f}")
    print(f"  Gjennomsnitt: {np.mean(fatigue_scores):.3f}")

    print("\n" + "="*60)
    print("VERIFISERING FULLFORT")
    print(f"Se visualiseringer i: {CONFIG.output.output_dir / 'ground_truth_verification'}")
    print("="*60)


if __name__ == "__main__":
    main()
