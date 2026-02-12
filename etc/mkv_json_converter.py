"""
Convert .mkv recordings to joint_data.json using offline_processor.exe.

Walks the dataset directory tree, finds .mkv files, and runs the offline
processor to extract skeleton joint data into JSON format.

Usage:
    python mkv_json_converter.py --exe path/to/offline_processor.exe
    python mkv_json_converter.py --exe path/to/offline_processor.exe --dataset path/to/dataset
"""

import os
import subprocess
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert .mkv recordings to joint_data.json"
    )
    parser.add_argument(
        "--exe", "-e", type=str,
        default=os.environ.get("OFFLINE_PROCESSOR_EXE", ""),
        help="Path to offline_processor.exe"
    )
    parser.add_argument(
        "--dataset", "-d", type=str,
        default=os.environ.get(
            "STRENGTH_DATASET_PATH",
            os.path.join(os.path.dirname(__file__), "..", "dataset")
        ),
        help="Root directory of dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exe_path = args.exe
    root_path = args.dataset

    if not exe_path or not os.path.isfile(exe_path):
        print(f"ERROR: offline_processor.exe not found: '{exe_path}'")
        print("Provide path via --exe or set OFFLINE_PROCESSOR_EXE environment variable.")
        return

    if not os.path.isdir(root_path):
        print(f"ERROR: Dataset root not found: '{root_path}'")
        return

    # Counters
    total_mkv_files = 0
    processed_files = 0
    skipped_files = 0

    # Total process timer
    total_start_time = time.time()

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(".mkv"):
                total_mkv_files += 1

                input_file = file
                output_file = "joint_data.json"
                output_path = os.path.join(root, output_file)

                # Skip if already processed
                if os.path.exists(output_path):
                    skipped_files += 1
                    print(f"[SKIP] Already exists: {output_path}")
                    continue

                print(f"[START] {os.path.join(root, input_file)}")

                # Timer for processing each file
                file_start_time = time.time()

                cmd = [
                    exe_path,
                    input_file,
                    output_file
                ]

                result = subprocess.run(cmd, cwd=root)

                # Stop file timer
                file_elapsed = time.time() - file_start_time

                if result.returncode == 0:
                    processed_files += 1
                    print(f"Time used {file_elapsed:.2f} sec")
                else:
                    print(f"[ERROR] Problem at {input_file} (return code {result.returncode})")

    # Stop total timer
    total_elapsed = time.time() - total_start_time

    # Summary
    print("\nSummary")
    print(f"Total .mkv files found : {total_mkv_files}")
    print(f"Processed              : {processed_files}")
    print(f"Skipped                : {skipped_files}")
    print(f"Total time             : {total_elapsed:.2f} sec")
    print("=================================")
    print("[ok] All finished!")


if __name__ == "__main__":
    main()
