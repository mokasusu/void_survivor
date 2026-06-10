import os
import fnmatch

home_dir = "/Users/vducc3110"
print(f"Searching home directory {home_dir} for any model zip files...")

found = []
# Skip system directories to search efficiently
skip_dirs = {
    'Library', 'Applications', 'Pictures', 'Music', 'Movies', 'Public',
    '.Trash', '.git', '.gemini', 'node_modules', 'venv', '.cache'
}

for root, dirs, files in os.walk(home_dir):
    # Prune directory search
    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
    
    for f in files:
        if f.endswith('.zip') and ('model' in f.lower() or 'ppo' in f.lower() or 'run' in f.lower()):
            full_path = os.path.join(root, f)
            print(f"Found match: {full_path} ({os.path.getsize(full_path)} bytes)")
            found.append(full_path)

if not found:
    print("No PPO or model zip files found in the user's home directories.")
