"""
Shared constants for Strength Training ML Pipeline v3.

This module contains constants used across multiple modules to avoid duplication.
"""

from typing import Dict, List
from ml.v3.utils.logging_utils import get_logger, setup_logging
# Set up logging
logger = get_logger('constants')


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

EXERCISE_CONFIG: Dict[str, Dict] = {
    "squat": {
        "primary_joint": "PELVIS",
        "assist_joints": ["HIP_LEFT", "SPINE_CHEST"],
        "validation_joints": ["HIP_RIGHT", "SPINE_NAVEL"],
        "direction": {
            "y_decrease": "eccentric",   # ned = excentrisk
            "y_increase": "concentric",  # opp = concentrisk
        },
        "min_agreement": 0.6,
    },
    "benchpress": {
        "primary_joint": "WRIST_RIGHT",  
        "assist_joints": ["SHOULDER_RIGHT", "ELBOW_RIGHT"],
        "validation_joints": ["WRIST_LEFT", "ELBOW_LEFT", "SHOULDER_LEFT"],
        "direction": {
            "y_decrease": "eccentric",
            "y_increase": "concentric",
        },
        "min_agreement": 0.65,
    },
    "deadlift": {
        "primary_joint": "SPINE_CHEST",
        "assist_joints": ["PELVIS", "SPINE_NAVEL", "HIP_LEFT", "SHOULDER_LEFT"],
        "validation_joints": ["HIP_RIGHT", "HIP_LEFT", "SHOULDER_RIGHT"],
        "direction": {
            "y_decrease": "concentric",   # opp = concentric
            "y_increase": "eccentric",    # ned = eccentric
        },
        "min_agreement": 0.7,
    },
    "pullup": {
        "primary_joint": "SPINE_CHEST",
        "assist_joints": ["PELVIS", "SPINE_NAVEL", "SHOULDER_LEFT", "SHOULDER_RIGHT"],
        "validation_joints": ["HIP_RIGHT", "HIP_LEFT", "NECK"],
        "direction": {
            "y_decrease": "concentric",   # opp = concentric
            "y_increase": "eccentric",    # ned = eccentric
        },
        "min_agreement": 0.7,
    },
}


for cfg in EXERCISE_CONFIG.values():
    cfg["primary_idx"] = JOINT_TO_IDX[cfg["primary_joint"]]
    
    # all_idx = primary + assist 
    cfg["all_idx"] = [
        JOINT_TO_IDX[cfg["primary_joint"]],
        *[JOINT_TO_IDX[j] for j in cfg.get("assist_joints", [])]
    ]
    
    #
    cfg["validation_idx"] = [
        JOINT_TO_IDX[j] for j in cfg.get("validation_joints", [])
    ]
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
    """
    Returnerer PRIMARY joint index for one excercies
    """
    if not exercise_type:
        return 0  # fallback / feilhåndtering
    
    key = exercise_type.lower().strip()
    
    if key not in EXERCISE_CONFIG:
        raise ValueError(f"Ukjent øvelse: {exercise_type}. Kjente: {list(EXERCISE_CONFIG.keys())}")
    
    return EXERCISE_CONFIG[key]["primary_idx"]


def get_exercise_all_joint_index(exercise_type: str) -> List[int]:
    """Returnerer list with primary + assist joint indices."""
    key = exercise_type.lower().strip()
    if key not in EXERCISE_CONFIG:
        raise ValueError(f"Unknown Excercie: {exercise_type}")
    
    return EXERCISE_CONFIG[key]["all_idx"]


def get_exercise_phase_config(exercise_type: str) -> Dict:
    """Get phase configuration for an exercise (case-insensitive)."""
    config = EXERCISE_CONFIG.get(exercise_type.lower())
    if config is None:
        logger.warning(f"Exercise type '{exercise_type}' not found in phase config. Defaulting to squat config.")
    return config


def get_exercise_joints(exercise_type: str) -> Dict[str, List[str]]:
    """Get detailed joint configuration for an exercise (case-insensitive)."""
    joints = EXERCISE_CONFIG.get(exercise_type.lower())
    if joints is None:
        logger.warning(f"Exercise type '{exercise_type}' not found in detailed joint config. Defaulting to squat config.")
    return joints

# Bare for manuell testing 
if __name__ == "__main__":
    print("Testing constants module directly")
    print("Primary joint index for squat:", get_exercise_all_joint_index("deadlift"))