import os
import glob
from PIL import Image

artifact_dir = "/Users/vducc3110/.gemini/antigravity-ide/brain/83e38a68-08ef-4c2f-b33c-207cc0581e57"
print(f"Artifact directory: {artifact_dir}")

media_files = glob.glob(os.path.join(artifact_dir, "media__*"))
print(f"Found {len(media_files)} media files.")

for f in sorted(media_files):
    size_bytes = os.path.getsize(f)
    try:
        with Image.open(f) as img:
            print(f"File: {os.path.basename(f)} | Format: {img.format} | Size: {img.size} | Bytes: {size_bytes}")
    except Exception as e:
        print(f"File: {os.path.basename(f)} | Error: {e} | Bytes: {size_bytes}")
