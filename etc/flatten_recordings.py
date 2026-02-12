"""
Flatten dataset structure by moving files from recording_NNN/ subfolders
up one level into the SetNN/ folder.

Before:  dataset / Exercise / PersonNN / SetNN / recording_NNN / [files]
After:   dataset / Exercise / PersonNN / SetNN / [files]

The .mkv file already at the SetNN level is left untouched.
Empty recording_NNN folders are removed after moving.

Usage:
    python flatten_recordings.py
    python flatten_recordings.py --dataset_root "C:\\path\\to\\dataset"
    python flatten_recordings.py --dry-run
"""

import os
import shutil
import argparse

DEFAULT_DATASET_ROOT = os.environ.get(
    "STRENGTH_DATASET_PATH",
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flatten recording subfolders into set folders"
    )
    parser.add_argument(
        "--dataset_root", "-d", type=str, default=DEFAULT_DATASET_ROOT,
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only print what would be done, without moving files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root

    if not os.path.isdir(dataset_root):
        print(f"ERROR: Dataset root not found: {dataset_root}")
        return

    print(f"Dataset root: {dataset_root}")
    if args.dry_run:
        print("MODE: Dry run (no files will be moved)")
    print("=" * 60)

    moved_count = 0
    removed_dirs = 0
    skipped_sets = 0
    conflicts = 0

    for exercise in sorted(os.listdir(dataset_root)):
        exercise_path = os.path.join(dataset_root, exercise)
        if not os.path.isdir(exercise_path):
            continue

        for person in sorted(os.listdir(exercise_path)):
            person_path = os.path.join(exercise_path, person)
            if not os.path.isdir(person_path):
                continue

            for set_name in sorted(os.listdir(person_path)):
                set_path = os.path.join(person_path, set_name)
                if not os.path.isdir(set_path):
                    continue

                # Find recording_NNN subfolders
                recording_dirs = [
                    d for d in os.listdir(set_path)
                    if os.path.isdir(os.path.join(set_path, d))
                    and d.startswith("recording_")
                ]

                if not recording_dirs:
                    skipped_sets += 1
                    continue

                label = f"{exercise}/{person}/{set_name}"

                for rec_dir in recording_dirs:
                    rec_path = os.path.join(set_path, rec_dir)
                    files = os.listdir(rec_path)

                    if not files:
                        print(f"  Empty dir, removing: {label}/{rec_dir}")
                        if not args.dry_run:
                            os.rmdir(rec_path)
                        removed_dirs += 1
                        continue

                    # Move each file up one level
                    for filename in files:
                        src = os.path.join(rec_path, filename)
                        dst = os.path.join(set_path, filename)

                        if os.path.exists(dst):
                            print(f"  CONFLICT: {label}/{filename} already exists, skipping")
                            conflicts += 1
                            continue

                        if args.dry_run:
                            print(f"  [DRY RUN] {label}/{rec_dir}/{filename} -> {label}/{filename}")
                        else:
                            shutil.move(src, dst)

                        moved_count += 1

                    # Remove empty recording folder
                    remaining = os.listdir(rec_path)
                    if not remaining:
                        if not args.dry_run:
                            os.rmdir(rec_path)
                        print(f"  Flattened: {label}/{rec_dir} ({len(files)} files)")
                        removed_dirs += 1
                    else:
                        print(f"  WARNING: {label}/{rec_dir} still has {len(remaining)} "
                              f"files after move (conflicts)")

    print("=" * 60)
    print("Summary:")
    print(f"  Files moved:        {moved_count}")
    print(f"  Folders removed:    {removed_dirs}")
    print(f"  Sets already flat:  {skipped_sets}")
    print(f"  Conflicts:          {conflicts}")
    print("Done.")


if __name__ == "__main__":
    main()
