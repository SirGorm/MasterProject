"""Utility modules for Strength Training ML Pipeline v3."""

from .logging_utils import (
    setup_logging,
    get_logger,
)

from .constants import (
    JOINT_NAMES,
    JOINT_TO_IDX,
    EXERCISE_JOINT_IDX,
    EXERCISE_PHASE_CONFIG,
    EXERCISE_JOINTS_DETAILED,
    BONE_CONNECTIONS,
    PHASES,
    REST_VELOCITY_THRESHOLD,
    get_exercise_joint_index,
    get_exercise_phase_config,
    get_exercise_joints,
)

from .signal_utils import (
    compute_emg_frequency_features,
    extract_emg_features,
    extract_acc_features,
    compute_acc_basic_features,
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    # Constants
    'JOINT_NAMES',
    'JOINT_TO_IDX',
    'EXERCISE_JOINT_IDX',
    'EXERCISE_PHASE_CONFIG',
    'EXERCISE_JOINTS_DETAILED',
    'BONE_CONNECTIONS',
    'PHASES',
    'REST_VELOCITY_THRESHOLD',
    'get_exercise_joint_index',
    'get_exercise_phase_config',
    'get_exercise_joints',
    # Signal utils
    'compute_emg_frequency_features',
    'extract_emg_features',
    'extract_acc_features',
    'compute_acc_basic_features',
]
