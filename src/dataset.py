import numpy as np
import pandas as pd
import torch
import json

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import CONFIG 
class Dataset_Scanner:
    '''Scans the dataset direcotory and finds excercise and participats automatially.'''

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
    
    def discover_structure(self):
        '''
        Auto find datastructure.

        Returns:
            directory with exercises as keys and list of participant IDs as values.
        '''
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        structure = {}

        # Goes thorough each folder in dataset path
        for exercise_dir in sorted(self.dataset_path.iterdir()):
            if not exercise_dir.is_dir():
                continue

            exercise_name = exercise_dir.name
            person_ids = []

            # Find all participant folders in each exercise folder
            for person_dir in sorted(exercise_dir.iterdir()):
                if person_dir.is_dir():
                    person_ids.append(person_dir.name)
            
            if person_ids:
                structure[exercise_name] = person_ids
    
        return structure   

    def print_structure(self, structure: dict):
        '''Prints the discovered dataset structure.'''
        print("\n" + "="*60)
        print("DATASETSTRUKTUR")
        print("="*60)

        total_persons = 0
        for exercise, persons in structure.items():
            print(f"\n {exercise}:")
            print(f"   Number of Persons: {len(persons)}")
            print(f"   Person-IDer: {', '.join(persons)}")
            total_persons += len(persons)
        
        print(f"\n{'='*60}")
        print(f"Totalt {len(structure)} excercises, {total_persons} persons")
        print("="*60)

class ExcerciseDataPreprocessor:
    '''Handels loading and preprocessing of biosignal data and joint-data.'''

    def __init__(self, config: dict):
        self.config = config
        self.target_length = config['data']['target_length']
        self.biosignals = config['data']['biosignals']
        self.use_joint_data = config['data']['use_joint_data']
        self.min_signal_length = config['data']['min_signal_length']
        self.scaler = StandardScaler()
        self.augmentation_enabled = config['data']['augmentation']['enabled']

        # Statistics 
        self.stats = {
            'total_samples': 0,
            'failed_samples': 0,
            'warnings': []
        }
    
    def load_biosignals(self, person_path: Path) -> Optional[List[np.ndarray]]:
        '''
        Loads biosignal data for one person from a CSV file.
        
        return:
            List of numpy arrays, or None of data shortage
        '''
        
        signals = []
        missing_signals = []

        for signal_name in self.biosignals:
            csv_path = person_path / f"{signal_name}.csv"

            if not csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if df.empty:
                        missing_signals.append(signal_name)
                        signals.append(np.zeros(self.target_length))
                        continue

                    # Assuming first column contains the signal data
                    signal_data = df.iloc[:, 0].values

                    # Check min length
                    if len(signal_data) < self.min_signal_length:
                        self.stats['warnings'].append(
                            f'Signal {signal_name} i {person_path} is to short ({len(signal_data)} samples)'
                        )
                        return None
                    
                    signals.append(signal_data)

                except Exception as e:
                    self.stats['warnings'].append(
                        f'Error loading {signal_name} in {person_path}: {str(e)}'
                    )
                    return None
            else:
                missing_signals.append(signal_name)
                signals.append(np.zeros(self.target_length))
        
        if missing_signals:
            self.stats['warnings'].append(
                f'Missing signals in {person_path}: {", ".join(missing_signals)}'
            )
        return signals if signals else None

    def load_joint_data(self, json_path: Path) -> Optional[np.ndarray]:
        '''
        Loads joint data from a JSON file.

        return:
            Numpy array of joint data, or None if loading fails
        '''
        if not self.use_joint_data or not json_path.exists():
            return None
        
        try: 
            with open(json_path, 'r', encoding='utf-8') as f:
                joint_data = json.load(f)

            frames = joint_data.get('frames', [])
            joint_sequences = []

            for frame in frames:
                bodies = frame.get('bodies', [])
                if bodies:
                    body = bodies[0]
                    joints = body.get('joints', [])
                    joint_coords = []
                    for joint in joints:
                        joint_coords.extend([
                            joint.get('x', 0.0),
                            joint.get('y', 0.0),
                            joint.get('z', 0.0)
                        ])
                    joint_sequences.append(joint_coords)
                
            if not joint_sequences:
                return None
            
            return np.array(joint_sequences)
        
        except Exception as e:
            self.stats['warnings'].append(
                f'Error loading joint data from {json_path}: {str(e)}'
            )
            return None
        

    def pad_or_trim(self, signal: np.ndarray) -> np.ndarray:
        '''
        Pads or trims a signal to the target length.

        return:
            Numpy array of length target_length
        '''
        if signal.ndim == 1:
            current_length = len(signal)
            if current_length < self.target_length:
                padding = self.target_length - current_length
                return np.pad(signal, (0, padding), 'constant')
            elif current_length > self.target_length:
                return signal[:self.target_length]
            return signal
        else:
            # Muilti-dimensional signal
            current_length = signal.shape[0]
            if current_length < self.target_length:
                padding = self.target_length - current_length
                pad_width = [(0, padding)] + [(0, 0)] * (signal.ndim - 1)
                return np.pad(signal, pad_width, mode='constant')
            elif current_length > self.target_length:
                return signal[:self.target_length]
            return signal
    
    def normalize_signals(self, signals: List[np.ndarray], fit: bool = False) -> np.ndarray:
        '''Normalize biosignals.'''
        signals_array = np.array(signals)
        signals_transposed = signals_array.T
        
        if fit:
            normalized = self.scaler.fit_transform(signals_transposed)
        else:
            normalized = self.scaler.transform(signals_transposed)
        
        return normalized.T
    
    def augment_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Data augmentation for bio time signals.
        - Gaussian noise
        - Time shifting
        - Amplitude scaling
        """
        if not self.augmentation_enabled:
            return signal
        
        aug_config = self.config['data']['augmentation']
        augmented = signal.copy()
        
        # Gaussian noise
        noise = np.random.normal(0, aug_config['noise_factor'], signal.shape)
        augmented += noise
        
        # Time shifting
        shift = np.random.randint(-aug_config['time_shift_max'], aug_config['time_shift_max'])
        augmented = np.roll(augmented, shift, axis=-1)
        
        # Amplitude scaling
        scale = 1 + np.random.uniform(-aug_config['scaling_factor'], aug_config['scaling_factor'])
        augmented *= scale
        
        return augmented
    
    def process_sample(self, person_path: Path, fit_scaler: bool = False, augment: bool = False) -> Optional[np.ndarray]:
        '''
        Preocess a single person's data sample.

        Args: 
              person_path: Path to the person's data directory
              fit_scaler: Whether to fit the scaler on this data
              augment: Whether to apply data augmentation

        Returns:
            Combined singnal array or None if loading fails
        '''

        # Load biosignals
        biosignals = self.load_biosignals(person_path)
        if biosignals is None:
            self.stats['failed_samples'] += 1
            return None
       
        # Pad/trim biosignals
        biosignals = [self.pad_or_trim(sig) for sig in biosignals]
        # Normalize biosignals
        biosignals = self.normalize_signals(biosignals, fit=fit_scaler)
       
        # Load joint-data if enabled
        if self.use_joint_data:
            json_path = person_path / "joint_data.json"
            joint_data = self.load_joint_data(json_path)

            if joint_data is not None:
                joint_data = self.pad_or_trim(joint_data)
                if joint_data.ndim == 2:
                    joint_data = joint_data.T
            else:
                # If joint data loading fails, use zeros
                joint_data = np.zeros((75, self.target_length))  # Assuming 25 joints * 3 coords
            
            combined = np.vstack([biosignals, joint_data])
        else:
            combined = np.array(biosignals)

        # Data augmentation if enabled
        if augment:
            combined = self.augment_signal(combined)
        
        self.stats['total_samples'] += 1
        return combined.astype(np.float32)
       
    def print_statistics(self):
        ''' Prints preprocessing statistics. '''
        print("\n" + "="*60)
        print("DATAPROSESSERING - STATISTIKK")
        print("="*60)
        print(f"Totalt sampels processed: {self.stats['total_samples']}")
        print(f"Failed sampels: {self.stats['failed_samples']}")
        
        if self.stats['warnings']:
            print(f"\nWarnings ({len(self.stats['warnings'])}):")
            for warning in self.stats['warnings'][:10]:  # SHow only first 10 warnings
                print(f"{warning}")
            if len(self.stats['warnings']) > 10:
                print(f"  ... and {len(self.stats['warnings']) - 10} more warnings.")
        print("="*60)

class ExerciseDataset(Dataset):
    '''
    Transorms data and labels into a PyTorch Dataset.
    '''
    def __init__(self, data: List[np.ndarray], labels: List[int], exercise_names: List[str], person_ids: List[str]):
        self.data           = data
        self.labels         = labels
        self.exercise_names = exercise_names
        self.person_ids     = person_ids
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
    
    def get_metadata(self, idx: int) -> Dict:
        '''Get metadata for a given sample.'''
        return {
            'exercise': self.exercise_names[idx],
            'person_id': self.person_ids[idx],
            'label': self.labels[idx]
        }

def load_dataset(config: Dict) -> Tuple[DataLoader, DataLoader, ExcerciseDataPreprocessor, Dict]:
    '''
    Load and process the dataset with multiple persons
    Returns:
        train_loader, val_loader, preprocessor, metadata
    '''

    dataset_path = Path(config['data']['dataset_path'])

    # Atomatically discover dataset structure 
    scanner = Dataset_Scanner(dataset_path)
    structure = scanner.discover_structure()
    scanner.print_structure(structure)

    if not structure:
        raise ValueError("No exercises found in dataset.")
    
    # Initialize preprocessor
    preprocessor = ExcerciseDataPreprocessor(CONFIG)

    all_data = []
    all_labels = []
    all_exercise_names = []
    all_person_ids = []

    # Build label mapping
    exercises = sorted(structure.keys())
    label_map = {exercise: idx for idx, exercise in enumerate(exercises)}

    print("\nLoading dataset...")

    # Search through all exercises and persons
    first_sample = True
    for exercise_name, person_ids in structure.items():
        label_idx = label_map[exercise_name]
        print(f"\nProcessing exercise: {exercise_name} ({len(person_ids)} persons)")

        for person_id in tqdm(person_ids, desc=f'Person',leave=False):
            person_path = dataset_path / exercise_name / person_id # Path to person's data directory for this exercise

            # process sample
            sample = preprocessor.process_sample(
                person_path, 
                fit_scaler=first_sample,
                augment=False       
            )

            if sample is not None:
                all_data.append(sample)
                all_labels.append(label_idx)
                all_exercise_names.append(exercise_name)
                all_person_ids.append(person_id)
                first_sample = False

    if not all_data:
        raise ValueError("No valid data samples were loaded.")    
        
    #print statistics
    preprocessor.print_statistics()

    print(f'\n Total valid samples: {len(all_data)}')
    print(f' samples per exercise:')
    for exercise in exercises:
        count = sum(1 for e in all_exercise_names if e == exercise)
        print(f'  {exercise}: {count} samples')
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val, ex_train, ex_val, pid_train, pid_val = train_test_split(
        all_data, all_labels, all_exercise_names, all_person_ids,
        test_size=config['data']['train_val_split'],
        random_state=config['data']['random_seed'],
        stratify=all_labels if len(set(all_labels)) > 1 else None
    )
    
    # Create PyTorch datasets
    train_dataset = ExerciseDataset(X_train, y_train, ex_train, pid_train)
    val_dataset = ExerciseDataset(X_val, y_val, ex_val, pid_val)        

    # Create DataLoaders    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0  # Sett til 0 for Windows kompatibilitet
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Metadata
    metadata = {
        'exercises': exercises,
        'label_map': label_map,
        'n_channels': all_data[0].shape[0],
        'target_length': config['data']['target_length'],
        'train_samples': len(X_train),
        'val_samples': len(X_val)
    }
    
    return train_loader, val_loader, preprocessor, metadata