from pathlib import Path
import logging
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset directory structure")  
    parser.add_argument("--num_people", "-p", type=int, default=10, help="Number of participants")
    parser.add_argument("--sets", "-s", type=int, default=4, help="Number of sets per person")
    parser.add_argument("--dataset_root", "-d", type=str, default=None, help="Root directory for dataset")
    return parser.parse_args()

def main():
    args = parse_args()

    # ---------- DATASET ROOT ----------
    DATASET_ROOT = Path(args.dataset_root) if args.dataset_root else Path(__file__).parent.parent / "dataset"

    exercises = ["Squat", "Benchpress", "Pullup", "Deadlift"]

    # ---------- LOGGING ONLY TO FILE ----------
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "dataset_creation.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file)]  # kun fil, ingen StreamHandler
    )

    logging.info("Starting dataset structure creation")
    logging.info(f"Participants: {args.num_people}, Sets: {args.sets}, Dataset root: {DATASET_ROOT}")

    # ---------- CHECK ROOT ----------
    if not DATASET_ROOT.exists():
        choice = input(f"Dataset root does not exist: {DATASET_ROOT}\nCreate it? (Y/N): ").strip().lower()
        if choice == "y":
            DATASET_ROOT.mkdir(parents=True)
            logging.info(f"Dataset root created: {DATASET_ROOT}")
        else:
            logging.info("User chose to exit without creating dataset root")
            print("Exiting program.")
            sys.exit(0)
    elif not DATASET_ROOT.is_dir():
        logging.error(f"Dataset root is not a directory: {DATASET_ROOT}")
        raise RuntimeError(f"Dataset root is not a directory: {DATASET_ROOT}")

    # ---------- COUNTERS ----------
    total_dirs_created = 0
    new_persons_created = 0

    # ---------- CREATE DIRECTORY STRUCTURE ----------
    for exercise in exercises:
        exercise_dir = DATASET_ROOT / exercise
        if not exercise_dir.exists():
            exercise_dir.mkdir(parents=True)
            total_dirs_created += 1
            logging.info(f"Created folder: {exercise_dir}")

        for person_id in range(1, args.num_people + 1):
            person_dir = exercise_dir / f"Person{person_id:02d}"
            if not person_dir.exists():
                person_dir.mkdir(parents=True)
                total_dirs_created += 1
                new_persons_created += 1
                logging.info(f"Added new person: {person_dir}")

            for set_id in range(1, args.sets + 1):
                recording_dir = person_dir / f"Set{set_id:02d}"
                if not recording_dir.exists():
                    recording_dir.mkdir(parents=True)
                    total_dirs_created += 1
                    logging.info(f"Created folder: {recording_dir}")

    # ---------- PRINT SUMMARY TO CONSOLE ----------
    print("\nSummary:")
    print(f"Total folders created: {total_dirs_created}")
    print(f"New persons added: {new_persons_created/len(exercises)}")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Log file: {log_file}\n")

if __name__ == "__main__":
    main()
