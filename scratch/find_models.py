import os
import glob

downloads_dir = "/Users/vducc3110/Downloads"
print(f"Searching for .zip files in {downloads_dir} and its subdirectories...")

zip_files = glob.glob(os.path.join(downloads_dir, "**/*.zip"), recursive=True)
for f in zip_files:
    print(f"Found ZIP: {f} ({os.path.getsize(f)} bytes)")

print("\nSearching for any files with 'model' in their name in Downloads...")
model_files = glob.glob(os.path.join(downloads_dir, "**/*model*"), recursive=True)
for f in model_files:
    if os.path.isfile(f):
        print(f"Found model file: {f} ({os.path.getsize(f)} bytes)")
