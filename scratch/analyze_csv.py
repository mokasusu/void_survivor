import pandas as pd
import glob

csv_files = glob.glob("/Users/vducc3110/Desktop/void_survivor/*.csv")
for f in sorted(csv_files):
    df = pd.read_csv(f)
    print(f"File: {f}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Shape: {df.shape}")
    print(f"  Step range: {df['Step'].min()} to {df['Step'].max()}")
    print(f"  Value range: {df['Value'].min()} to {df['Value'].max()}")
    print(f"  Value mean: {df['Value'].mean()}")
    print(f"  First few rows:\n{df.head(2)}")
