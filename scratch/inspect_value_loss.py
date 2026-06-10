import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "/Users/vducc3110/Downloads/1"
event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))

# Filter out files without numeric suffix
def get_suffix(filepath):
    try:
        return int(filepath.split('.')[-1])
    except ValueError:
        return -1
        
event_files = [f for f in event_files if get_suffix(f) != -1]
event_files.sort(key=get_suffix)

value_losses = []
for idx, f in enumerate(event_files):
    try:
        ea = EventAccumulator(f)
        ea.Reload()
        if 'train/value_loss' in ea.Tags().get('scalars', []):
            for e in ea.Scalars('train/value_loss'):
                value_losses.append({'step': e.step, 'value': e.value})
    except Exception as e:
        pass

if value_losses:
    df = pd.DataFrame(value_losses)
    df = df.drop_duplicates(subset=['step']).sort_values('step')
    print(f"Value Loss count: {len(df)}")
    print(f"Min: {df['value'].min()}")
    print(f"Max: {df['value'].max()}")
    print(f"Mean: {df['value'].mean()}")
    print(f"Last value: {df['value'].iloc[-1] if not df.empty else None}")
else:
    print("No value_loss data found.")
