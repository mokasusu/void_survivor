import os
import shutil
import glob

artifact_dir = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57"
dest_dir = "/Users/vducc3110/Desktop/void_survivor/media_assets"

os.makedirs(dest_dir, exist_ok=True)
print(f"Creating directory: {dest_dir}")

media_files = glob.glob(os.path.join(artifact_dir, "media__*"))
print(f"Copying {len(media_files)} files...")

for f in media_files:
    filename = os.path.basename(f)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(f, dest_path)
    print(f"Copied: {filename} to {dest_path}")
