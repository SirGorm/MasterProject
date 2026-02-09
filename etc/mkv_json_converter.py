import os
import subprocess
import time
"""
Batch process .mkv files in dataset to generate joint_data.json files
    - Run script, it will walk through dataset folders
    - For each .mkv file found, it will check if joint_data.json exists
    - If not, it will call offline_processor.exe to process the .mkv file

"""
# Path to offline_processor.exe
exe_path = r"C:\Users\skogl\Downloads\eirikgsk\MasterProject\Offline_processor\build\bin\Debug\offline_processor.exe"

# Path to dataset - UPDATE THIS to match your actual dataset location
root_path = r"c:\Users\skogl\Downloads\eirikgsk\Master_git\dataset"   # Dataset

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
                print(f"[SKIP] Finnes allerede: {output_path}")
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
                print(f"[ERROR] Problem at {input_file} (returkode {result.returncode})")

# Stop total timer
total_elapsed = time.time() - total_start_time

#Summary
print("\nSummary")
print(f"Total .mkv-filer found : {total_mkv_files}")
print(f"Processed                     : {processed_files}")
print(f"Skipped                       : {skipped_files}")
print(f"Total time                    : {total_elapsed:.2f} sec")
print("=================================")
print("[ok] All finished!")
