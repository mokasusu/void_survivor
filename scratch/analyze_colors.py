import os
import glob
import numpy as np
from PIL import Image

media_dir = "/Users/vducc3110/Desktop/void_survivor/media_assets"
media_files = sorted(glob.glob(os.path.join(media_dir, "media__*")))

for f in media_files:
    filename = os.path.basename(f)
    try:
        with Image.open(f) as img:
            # Convert to numpy array to analyze colors
            arr = np.array(img.convert('RGB'))
            mean_color = arr.mean(axis=(0,1))
            # Check how dark it is (game is usually dark background)
            darkness = (arr < 30).all(axis=2).mean() * 100
            
            # Print analysis
            print(f"File: {filename}")
            print(f"  Dimensions: {img.size} | Mode: {img.mode}")
            print(f"  Mean RGB: {mean_color.round(1)}")
            print(f"  Dark pixels (<30,30,30): {darkness:.1f}%")
            
            # Check if there is a dominant color channel
            r, g, b = mean_color
            if darkness > 40:
                print("  => Possible gameplay screenshot (dark background)")
            elif max(r, g, b) - min(r, g, b) < 10 and mean_color.mean() > 240:
                print("  => Possible document/screenshot of text (near white background)")
            elif abs(r - g) < 20 and b < r - 30:
                print("  => Possible warm/yellowish document or graph")
            else:
                print("  => General image/plot")
            print("-" * 40)
    except Exception as e:
        print(f"File: {filename} | Error: {e}")
