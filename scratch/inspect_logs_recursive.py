import os
import glob

log_dir = "/Users/vducc3110/Downloads/1"
all_files = glob.glob(os.path.join(log_dir, "**/*"), recursive=True)
non_event_files = []
for f in all_files:
    if os.path.isfile(f) and not ("tfevents" in f):
        non_event_files.append(f)

print(f"Found {len(non_event_files)} non-event files:")
for f in sorted(non_event_files):
    print(f"  {f} ({os.path.getsize(f)} bytes)")
