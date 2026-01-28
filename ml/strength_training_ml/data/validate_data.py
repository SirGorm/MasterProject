"""
Data Validation Script for Strength Training ML Pipeline.

Validates:
- All required data files are present
- Consistency between signals, markers, and joint data
- Temporal alignment across modalities
- Data quality (NaN, inf, outliers)

Sessions with errors are SKIPPED (not used for training).
Training continues with valid sessions only.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG, SIGNALS


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    severity: str = 'error'  # 'error', 'warning', 'info'
    details: Optional[Dict] = None


@dataclass
class SessionValidation:
    """Validation results for a single session."""
    exercise: str
    session_id: str
    path: Path
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(r.severity == 'error' and not r.passed for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.severity == 'warning' and not r.passed for r in self.results)


class DataValidator:
    """
    Validates dataset structure and data consistency.
    """

    def __init__(self, dataset_path: Path = None, config=None):
        """
        Initialize the data validator.

        Args:
            dataset_path: Path to dataset directory
            config: Configuration object
        """
        self.config = config or CONFIG
        self.dataset_path = Path(dataset_path or self.config.data.dataset_path)

        # Required files per session
        self.required_csv_files = [
            cfg.file_name for cfg in SIGNALS.values()
            if cfg.file_name.endswith('.csv')
        ]
        self.required_json_files = [
            self.config.data.markers_file,
            self.config.data.joints_file
        ]

        # Validation results
        self.sessions: List[SessionValidation] = []
        self.global_errors: List[str] = []
        self.global_warnings: List[str] = []

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if there are usable sessions (even if some have errors)
        """
        print("\n" + "="*70)
        print("DATA VALIDATION")
        print("="*70)

        # Check dataset path exists
        if not self.dataset_path.exists():
            self.global_errors.append(f"Dataset path does not exist: {self.dataset_path}")
            self._print_summary()
            return False

        # Discover and validate structure
        self._discover_structure()

        # Validate each session
        for session in self.sessions:
            self._validate_session(session)

        # Print summary
        self._print_summary()

        # Return True if we have at least some valid sessions
        valid_sessions = self.get_valid_sessions()
        return len(self.global_errors) == 0 and len(valid_sessions) > 0

    def _discover_structure(self):
        """Discover dataset structure (exercises and sessions)."""
        print(f"\nScanning dataset: {self.dataset_path}")

        expected_exercises = self.config.data.exercises

        for exercise_name in expected_exercises:
            exercise_path = self.dataset_path / exercise_name

            if not exercise_path.exists():
                self.global_warnings.append(f"Exercise folder not found: {exercise_name}")
                continue

            # Find session folders (numbered directories)
            session_dirs = [
                d for d in sorted(exercise_path.iterdir())
                if d.is_dir() and d.name.isdigit()
            ]

            if not session_dirs:
                self.global_warnings.append(f"No sessions found for exercise: {exercise_name}")
                continue

            print(f"  {exercise_name}: {len(session_dirs)} sessions")

            for session_dir in session_dirs:
                self.sessions.append(SessionValidation(
                    exercise=exercise_name,
                    session_id=session_dir.name,
                    path=session_dir
                ))

    def _validate_session(self, session: SessionValidation):
        """Validate a single session."""
        # Check required files exist
        self._check_required_files(session)

        # Validate markers.json
        markers_data = self._validate_markers(session)

        # Validate joint_data.json
        joint_data = self._validate_joints(session)

        # Validate CSV signal files
        signal_durations = self._validate_signals(session)

        # Cross-validate temporal alignment
        if markers_data and signal_durations:
            self._validate_temporal_alignment(session, markers_data, signal_durations, joint_data)

    def _check_required_files(self, session: SessionValidation):
        """Check that all required files are present."""
        # Check CSV files
        for csv_file in self.required_csv_files:
            file_path = session.path / csv_file
            if not file_path.exists():
                # Check for backup file
                backup_path = file_path.with_suffix('.csv.bak')
                if backup_path.exists():
                    session.results.append(ValidationResult(
                        passed=True,
                        message=f"CSV file has backup: {csv_file}",
                        severity='info'
                    ))
                else:
                    session.results.append(ValidationResult(
                        passed=False,
                        message=f"Missing CSV file: {csv_file}",
                        severity='error'
                    ))

        # Check JSON files
        for json_file in self.required_json_files:
            file_path = session.path / json_file
            if not file_path.exists():
                session.results.append(ValidationResult(
                    passed=False,
                    message=f"Missing JSON file: {json_file}",
                    severity='error'
                ))

    def _validate_markers(self, session: SessionValidation) -> Optional[Dict]:
        """Validate markers.json file."""
        markers_path = session.path / self.config.data.markers_file

        if not markers_path.exists():
            return None

        try:
            with open(markers_path, 'r') as f:
                markers_data = json.load(f)

            markers = markers_data.get('markers', [])

            # Check for start marker
            start_markers = [m for m in markers if m.get('label', '').lower() == 'start']
            if not start_markers:
                session.results.append(ValidationResult(
                    passed=False,
                    message="No 'start' marker found in markers.json",
                    severity='error'
                ))
                return None

            start_time = start_markers[0].get('time', 0)

            # Check for end marker or use last marker
            end_time = markers[-1].get('time', 0) if markers else 0

            # Count repetitions (markers after start, excluding start itself)
            rep_markers = [m for m in markers if m.get('time', 0) > start_time]
            n_reps = len(rep_markers)

            session.results.append(ValidationResult(
                passed=True,
                message=f"Markers valid: {n_reps} repetitions, duration {end_time - start_time:.1f}s",
                severity='info',
                details={
                    'start_time': start_time,
                    'end_time': end_time,
                    'n_reps': n_reps,
                    'duration': end_time - start_time
                }
            ))

            return {
                'start_time': start_time,
                'end_time': end_time,
                'n_reps': n_reps,
                'markers': markers
            }

        except json.JSONDecodeError as e:
            session.results.append(ValidationResult(
                passed=False,
                message=f"Invalid JSON in markers.json: {e}",
                severity='error'
            ))
            return None
        except Exception as e:
            session.results.append(ValidationResult(
                passed=False,
                message=f"Error reading markers.json: {e}",
                severity='error'
            ))
            return None

    def _validate_joints(self, session: SessionValidation) -> Optional[Dict]:
        """Validate joint_data.json file."""
        joints_path = session.path / self.config.data.joints_file

        if not joints_path.exists():
            return None

        try:
            with open(joints_path, 'r') as f:
                joint_data = json.load(f)

            frames = joint_data.get('frames', [])
            bone_list = joint_data.get('bone_list', [])

            if not frames:
                session.results.append(ValidationResult(
                    passed=False,
                    message="No frames in joint_data.json",
                    severity='error'
                ))
                return None

            # Get timestamps
            timestamps = [f.get('timestamp_usec', 0) / 1e6 for f in frames]  # Convert to seconds
            duration = timestamps[-1] - timestamps[0] if timestamps else 0

            # Check for bodies in frames
            n_bodies = sum(f.get('num_bodies', 0) for f in frames)
            frames_with_bodies = sum(1 for f in frames if f.get('num_bodies', 0) > 0)

            if frames_with_bodies < len(frames) * 0.5:
                session.results.append(ValidationResult(
                    passed=False,
                    message=f"Less than 50% of frames have body data ({frames_with_bodies}/{len(frames)})",
                    severity='warning'
                ))

            session.results.append(ValidationResult(
                passed=True,
                message=f"Joint data valid: {len(frames)} frames, {len(bone_list)} bones, {duration:.1f}s",
                severity='info',
                details={
                    'n_frames': len(frames),
                    'n_bones': len(bone_list),
                    'duration': duration,
                    'start_time': timestamps[0] if timestamps else 0,
                    'end_time': timestamps[-1] if timestamps else 0
                }
            ))

            return {
                'n_frames': len(frames),
                'start_time': timestamps[0] if timestamps else 0,
                'end_time': timestamps[-1] if timestamps else 0,
                'duration': duration
            }

        except json.JSONDecodeError as e:
            session.results.append(ValidationResult(
                passed=False,
                message=f"Invalid JSON in joint_data.json: {e}",
                severity='error'
            ))
            return None
        except Exception as e:
            session.results.append(ValidationResult(
                passed=False,
                message=f"Error reading joint_data.json: {e}",
                severity='error'
            ))
            return None

    def _validate_signals(self, session: SessionValidation) -> Dict[str, float]:
        """Validate signal CSV files."""
        signal_durations = {}

        for signal_name, signal_config in SIGNALS.items():
            if not signal_config.file_name.endswith('.csv'):
                continue

            file_path = session.path / signal_config.file_name

            if not file_path.exists():
                continue

            try:
                # Try to read CSV
                df = pd.read_csv(file_path)

                # Check for required columns
                if len(df.columns) < 2:
                    session.results.append(ValidationResult(
                        passed=False,
                        message=f"{signal_name}: Less than 2 columns in CSV",
                        severity='error'
                    ))
                    continue

                # Get time column (first column)
                time_col = df.iloc[:, 0]
                value_col = df.iloc[:, 1]

                # Check for NaN values
                n_nan = value_col.isna().sum()
                if n_nan > 0:
                    pct_nan = (n_nan / len(value_col)) * 100
                    session.results.append(ValidationResult(
                        passed=pct_nan < 5,  # Allow up to 5% NaN
                        message=f"{signal_name}: {n_nan} NaN values ({pct_nan:.1f}%)",
                        severity='warning' if pct_nan < 5 else 'error'
                    ))

                # Check for inf values
                n_inf = np.isinf(value_col.values).sum()
                if n_inf > 0:
                    session.results.append(ValidationResult(
                        passed=False,
                        message=f"{signal_name}: {n_inf} inf values",
                        severity='error'
                    ))

                # Calculate duration
                duration = time_col.max() - time_col.min()
                signal_durations[signal_name] = duration

                # Validate sampling rate
                expected_samples = duration * signal_config.sampling_rate
                actual_samples = len(df)
                sample_ratio = actual_samples / expected_samples if expected_samples > 0 else 0

                if sample_ratio < 0.9 or sample_ratio > 1.1:
                    session.results.append(ValidationResult(
                        passed=False,
                        message=f"{signal_name}: Sampling rate mismatch (expected ~{expected_samples:.0f}, got {actual_samples})",
                        severity='warning'
                    ))

            except Exception as e:
                session.results.append(ValidationResult(
                    passed=False,
                    message=f"{signal_name}: Error reading CSV - {e}",
                    severity='error'
                ))

        return signal_durations

    def _validate_temporal_alignment(
        self,
        session: SessionValidation,
        markers_data: Dict,
        signal_durations: Dict[str, float],
        joint_data: Optional[Dict]
    ):
        """Validate temporal alignment across modalities."""
        marker_duration = markers_data['end_time'] - markers_data['start_time']

        # Check signal durations cover marker period
        for signal_name, signal_duration in signal_durations.items():
            if signal_duration < marker_duration * 0.9:
                session.results.append(ValidationResult(
                    passed=False,
                    message=f"{signal_name} duration ({signal_duration:.1f}s) shorter than exercise ({marker_duration:.1f}s)",
                    severity='warning'
                ))

        # Check joint data covers marker period
        if joint_data:
            joint_duration = joint_data['duration']
            if joint_duration < marker_duration * 0.9:
                session.results.append(ValidationResult(
                    passed=False,
                    message=f"Joint data duration ({joint_duration:.1f}s) shorter than exercise ({marker_duration:.1f}s)",
                    severity='warning'
                ))

    def _print_summary(self):
        """Print validation summary."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        # Global errors and warnings
        if self.global_errors:
            print("\nGLOBAL ERRORS:")
            for error in self.global_errors:
                print(f"  [ERROR] {error}")

        if self.global_warnings:
            print("\nGLOBAL WARNINGS:")
            for warning in self.global_warnings:
                print(f"  [WARNING] {warning}")

        # Per-session summary
        sessions_with_errors = [s for s in self.sessions if s.has_errors]
        sessions_with_warnings = [s for s in self.sessions if s.has_warnings and not s.has_errors]
        sessions_ok = [s for s in self.sessions if not s.has_errors and not s.has_warnings]

        # Sessions that can be used (OK + warnings only)
        usable_sessions = len(sessions_ok) + len(sessions_with_warnings)

        print(f"\nSESSION RESULTS:")
        print(f"  OK:       {len(sessions_ok)}")
        print(f"  Warnings: {len(sessions_with_warnings)}")
        print(f"  Errors:   {len(sessions_with_errors)} (will be SKIPPED)")

        # Detail errors
        if sessions_with_errors:
            print("\nSESSIONS WITH ERRORS (will be skipped):")
            for session in sessions_with_errors:
                print(f"  - {session.exercise}/{session.session_id}:")
                for result in session.results:
                    if not result.passed and result.severity == 'error':
                        print(f"      {result.message}")

        print("\n" + "="*70)
        if usable_sessions > 0:
            print(f"VALIDATION COMPLETE: {usable_sessions} usable sessions")
            if sessions_with_errors:
                print(f"  ({len(sessions_with_errors)} sessions will be skipped due to errors)")
        else:
            print("VALIDATION FAILED: No usable sessions found!")
        print("="*70)

    def get_valid_sessions(self) -> List[Tuple[str, str, Path]]:
        """
        Get list of sessions that passed validation.

        Returns:
            List of (exercise, session_id, path) tuples
        """
        return [
            (s.exercise, s.session_id, s.path)
            for s in self.sessions
            if not s.has_errors
        ]


def validate_dataset(dataset_path: Path = None, stop_on_error: bool = True) -> bool:
    """
    Validate dataset and optionally stop if errors found.

    Args:
        dataset_path: Path to dataset (default from config)
        stop_on_error: If True, raise exception on validation failure

    Returns:
        True if validation passed
    """
    validator = DataValidator(dataset_path)
    passed = validator.validate_all()

    if not passed and stop_on_error:
        raise ValueError("Data validation failed. Please fix errors before training.")

    return passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset structure and consistency")
    parser.add_argument('--dataset', type=str, help='Dataset path (default from config)')
    parser.add_argument('--no-stop', action='store_true', help='Do not exit on error')

    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else None
    stop_on_error = not args.no_stop

    try:
        passed = validate_dataset(dataset_path, stop_on_error=stop_on_error)
        sys.exit(0 if passed else 1)
    except ValueError as e:
        print(f"\n{e}")
        sys.exit(1)
