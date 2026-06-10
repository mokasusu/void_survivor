import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    log_dir = "/Users/vducc3110/Downloads/1"
    output_dir = "/Users/vducc3110/Desktop/void_survivor"
    
    print(f"Reading TensorBoard events from: {log_dir}")
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    # Sort files numerically by suffix
    def get_suffix(filepath):
        try:
            return int(filepath.split('.')[-1])
        except ValueError:
            return -1
            
    event_files.sort(key=get_suffix)
    event_files = [f for f in event_files if get_suffix(f) != -1]
    
    print(f"Found {len(event_files)} event files.")
    
    metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'time/fps',
        'train/approx_kl',
        'train/entropy_loss',
        'train/explained_variance'
    ]
    
    tag_data = {m: [] for m in metrics}
    
    for idx, f in enumerate(event_files):
        if idx % 50 == 0:
            print(f"  Processing file {idx}/{len(event_files)}: {os.path.basename(f)}")
        try:
            ea = EventAccumulator(f)
            ea.Reload()
            available_scalars = ea.Tags().get('scalars', [])
            for m in metrics:
                if m in available_scalars:
                    for e in ea.Scalars(m):
                        tag_data[m].append({
                            'step': e.step,
                            'value': e.value,
                            'iteration': get_suffix(f)
                        })
        except Exception as e:
            print(f"  Warning: failed to read {os.path.basename(f)}: {e}")
            
    print("Converting data to DataFrames and plotting...")
    
    # Custom color palette (clean and professional like TensorBoard / Tailwind)
    color_map = {
        'rollout/ep_rew_mean': '#3B82F6',        # Blue
        'rollout/ep_len_mean': '#10B981',        # Green
        'time/fps': '#8B5CF6',                   # Purple
        'train/approx_kl': '#F59E0B',            # Orange/Amber
        'train/entropy_loss': '#EC4899',         # Pink
        'train/explained_variance': '#06B6D4'    # Teal
    }
    
    title_map = {
        'rollout/ep_rew_mean': 'Episode Mean Reward (ep_rew_mean)',
        'rollout/ep_len_mean': 'Episode Mean Length (ep_len_mean)',
        'time/fps': 'Training FPS (time/fps)',
        'train/approx_kl': 'Approximate KL Divergence (approx_kl)',
        'train/entropy_loss': 'Policy Entropy Loss (entropy_loss)',
        'train/explained_variance': 'Value Explained Variance (explained_variance)'
    }
    
    ylabel_map = {
        'rollout/ep_rew_mean': 'Reward Value',
        'rollout/ep_len_mean': 'Length (Frames)',
        'time/fps': 'Frames Per Second',
        'train/approx_kl': 'KL Divergence',
        'train/entropy_loss': 'Entropy Loss',
        'train/explained_variance': 'Explained Variance Value'
    }
    
    for tag in metrics:
        points = tag_data[tag]
        if not points:
            print(f"No data points found for tag: {tag}")
            continue
            
        df = pd.DataFrame(points)
        # Sort and deduplicate
        df = df.drop_duplicates(subset=['step']).sort_values('step').reset_index(drop=True)
        
        # Plotting
        plt.figure(figsize=(10, 5), dpi=300)
        
        # Enable grid
        plt.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')
        
        # Raw line (faded color)
        plt.plot(df['step'], df['value'], color=color_map[tag], alpha=0.3, label='Raw')
        
        # Smooth line using EMA
        # Tensorboard uses exponential moving average smoothing
        # smooth_val = raw_val * (1 - weight) + prev_smooth_val * weight
        # weight is typically 0.6 to 0.9. Let's use EMA with span=15
        df['smoothed'] = df['value'].ewm(span=15, adjust=False).mean()
        plt.plot(df['step'], df['smoothed'], color=color_map[tag], linewidth=2, label='Smoothed')
        
        # Add labels and styling
        plt.title(title_map[tag], fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Timesteps', fontsize=11, labelpad=8)
        plt.ylabel(ylabel_map[tag], fontsize=11, labelpad=8)
        
        # Formatting X axis labels (e.g. 10M, 20M)
        def format_steps(x, pos):
            if x >= 1e6:
                return f'{x*1e-6:.1f}M'
            elif x >= 1e3:
                return f'{x*1e-3:.0f}K'
            return str(int(x))
            
        from matplotlib.ticker import FuncFormatter
        plt.gca().xaxis.set_major_formatter(FuncFormatter(format_steps))
        
        # Styling ticks and spine
        plt.tick_params(colors='#333333', which='both', labelsize=10)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_color('#888888')
        plt.gca().spines['bottom'].set_color('#888888')
        
        # Legend
        plt.legend(loc='best', frameon=True, facecolor='#FFFFFF', edgecolor='none', shadow=True)
        
        plt.tight_layout()
        
        # Construct filename
        safe_filename = tag.replace('/', '_') + '.png'
        output_path = os.path.join(output_dir, safe_filename)
        
        plt.savefig(output_path, dpi=300, facecolor='#FFFFFF')
        plt.close()
        print(f"  Saved plot: {output_path}")
        
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
