import os
import glob

search_paths = [
    "/Users/vducc3110/Downloads",
    "/Users/vducc3110/Downloads/1",
    "/Users/vducc3110/Desktop/void_survivor",
    "/Users/vducc3110/Desktop"
]

print("Searching for any .zip files...")
found_any = False
for path in search_paths:
    if os.path.exists(path):
        # We can search recursively up to depth 3 to avoid infinite loops or massive app bundles
        zips = glob.glob(os.path.join(path, "*.zip")) + \
               glob.glob(os.path.join(path, "*/*.zip")) + \
               glob.glob(os.path.join(path, "*/*/*.zip"))
        
        for z in set(zips):
            print(f"Found ZIP in {path}: {z} ({os.path.getsize(z)} bytes)")
            found_any = True

if not found_any:
    print("No .zip files found in typical directories.")
