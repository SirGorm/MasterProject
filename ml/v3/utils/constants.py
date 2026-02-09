"""
Shared constants for Strength Training ML Pipeline v3.

This module contains constants used across multiple modules to avoid duplication.
"""

from typing import Dict, List

# Azure Kinect joint names (32 joints)
JOINT_NAMES: List[str] = [
    'PELVIS', 'SPINE_NAVEL', 'SPINE_CHEST', 'NECK',
    'CLAVICLE_LEFT', 'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',
    'HAND_LEFT', 'HANDTIP_LEFT', 'THUMB_LEFT',
    'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
    'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT',
    'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT',
    'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT',
    'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
]

# Joint name to index mapping
JOINT_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(JOINT_NAMES)}

# Exercise-specific joint indices for tracking (simple mapping)
# Maps exercise name to primary joint index for phase/rep detection
EXERCISE_JOINT_IDX: Dict[str, int] = {
    'squat': 0,       # PELVIS
    'deadlift': 2,    # SPINE_CHEST
    'benchpress': 7,  # WRIST_LEFT
    'pullup': 2,      # SPINE_CHEST
    'pullups': 2,     # SPINE_CHEST
}

# Exercise-specific phase configuration for rule-based detection
# y_decrease: phase when Y decreases, y_increase: phase when Y increases
EXERCISE_PHASE_CONFIG: Dict[str, Dict] = {
    'squat': {
        'joint_idx': 0,  # PELVIS
        'y_decrease': 'eccentric',   # Going down
        'y_increase': 'concentric',  # Standing up
    },
    'benchpress': {
        'joint_idx': 7,  # WRIST_LEFT
        'y_decrease': 'eccentric',   # Bar going down
        'y_increase': 'concentric',  # Pressing up
    },
    'deadlift': {
        'joint_idx': 2,  # SPINE_CHEST
        'y_decrease': 'concentric',  # Standing up with bar
        'y_increase': 'eccentric',   # Lowering bar
    },
    'pullup': {
        'joint_idx': 2,  # SPINE_CHEST
        'y_decrease': 'concentric',  # Pulling up
        'y_increase': 'eccentric',   # Going down
    },
    'pullups': {
        'joint_idx': 2,  # SPINE_CHEST
        'y_decrease': 'concentric',  # Pulling up
        'y_increase': 'eccentric',   # Going down
    },
}

# Exercise joints for clustering-based phase detection (detailed)
# Primary joints: main joints for movement detection
# Secondary joints: supporting joints for context
EXERCISE_JOINTS_DETAILED: Dict[str, Dict[str, List[str]]] = {
    'squat': {
        'primary': ['KNEE_RIGHT', 'KNEE_LEFT', 'HIP_RIGHT', 'HIP_LEFT'],
        'secondary': ['ANKLE_RIGHT', 'ANKLE_LEFT', 'PELVIS']
    },
    'deadlift': {
        'primary': ['HIP_RIGHT', 'HIP_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT'],
        'secondary': ['PELVIS', 'SPINE_CHEST']
    },
    'benchpress': {
        'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
        'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_CHEST']
    },
    'pullup': {
        'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
        'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_NAVEL', 'SPINE_CHEST']
    },
    'pullups': {
        'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
        'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_NAVEL', 'SPINE_CHEST']
    },
}

# Bone connections for skeleton visualization (Azure Kinect)
BONE_CONNECTIONS: List[tuple] = [
    # Spine
    ('PELVIS', 'SPINE_NAVEL'),
    ('SPINE_NAVEL', 'SPINE_CHEST'),
    ('SPINE_CHEST', 'NECK'),
    ('NECK', 'HEAD'),
    # Left arm
    ('SPINE_CHEST', 'CLAVICLE_LEFT'),
    ('CLAVICLE_LEFT', 'SHOULDER_LEFT'),
    ('SHOULDER_LEFT', 'ELBOW_LEFT'),
    ('ELBOW_LEFT', 'WRIST_LEFT'),
    ('WRIST_LEFT', 'HAND_LEFT'),
    # Right arm
    ('SPINE_CHEST', 'CLAVICLE_RIGHT'),
    ('CLAVICLE_RIGHT', 'SHOULDER_RIGHT'),
    ('SHOULDER_RIGHT', 'ELBOW_RIGHT'),
    ('ELBOW_RIGHT', 'WRIST_RIGHT'),
    ('WRIST_RIGHT', 'HAND_RIGHT'),
    # Left leg
    ('PELVIS', 'HIP_LEFT'),
    ('HIP_LEFT', 'KNEE_LEFT'),
    ('KNEE_LEFT', 'ANKLE_LEFT'),
    ('ANKLE_LEFT', 'FOOT_LEFT'),
    # Right leg
    ('PELVIS', 'HIP_RIGHT'),
    ('HIP_RIGHT', 'KNEE_RIGHT'),
    ('KNEE_RIGHT', 'ANKLE_RIGHT'),
    ('ANKLE_RIGHT', 'FOOT_RIGHT'),
]

# Phase names
PHASES: List[str] = ['rest', 'concentric', 'eccentric']

# Velocity threshold for rest detection (m/s)
REST_VELOCITY_THRESHOLD: float = 0.02


def get_exercise_joint_index(exercise_type: str) -> int:
    """Get primary joint index for an exercise (case-insensitive)."""
    return EXERCISE_JOINT_IDX.get(exercise_type.lower(), 0)


def get_exercise_phase_config(exercise_type: str) -> Dict:
    """Get phase configuration for an exercise (case-insensitive)."""
    config = EXERCISE_PHASE_CONFIG.get(exercise_type.lower())
    if config is None:
        # Default to squat-like behavior
        config = EXERCISE_PHASE_CONFIG['squat']
    return config


def get_exercise_joints(exercise_type: str) -> Dict[str, List[str]]:
    """Get detailed joint configuration for an exercise (case-insensitive)."""
    joints = EXERCISE_JOINTS_DETAILED.get(exercise_type.lower())
    if joints is None:
        # Default to squat joints
        joints = EXERCISE_JOINTS_DETAILED['squat']
    return joints
