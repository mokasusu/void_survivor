import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "/Users/vducc3110/Downloads/1"
print(f"Checking directory: {log_dir}")
event_files = sorted(glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True))
print(f"Found {len(event_files)} event files.")

if event_files:
    # Read the last file or first file to see what tags are present
    ea = EventAccumulator(event_files[-1])
    ea.Reload()
    print("Tags in the last event file:")
    print("Scalars:", ea.Tags().get("scalars", []))
    
    # Also check a few other files if needed
    # Let's see if we can accumulate tags across files
    all_tags = set()
    for f in event_files[:10]:
        try:
            e = EventAccumulator(f)
            e.Reload()
            all_tags.update(e.Tags().get("scalars", []))
        except Exception as ex:
            pass
    print("Accumulated tags from first 10 files:", list(all_tags))
else:
    print("No event files found in /Users/vducc3110/Downloads/1")
