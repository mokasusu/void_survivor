import os
import glob

search_paths = [
    "/Users/vducc3110/Downloads/*",
    "/Users/vducc3110/Downloads/1/*",
    "/Users/vducc3110/Desktop/void_survivor/*",
    "/Users/vducc3110/Desktop/void_survivor/rl/*"
]

for path in search_paths:
    print(f"--- Glob: {path} ---")
    files = glob.glob(path)
    for f in sorted(files):
        if os.path.isfile(f) and f.endswith(('.json', '.csv', '.txt', '.log')):
            print(f"  {f} ({os.path.getsize(f)} bytes)")
