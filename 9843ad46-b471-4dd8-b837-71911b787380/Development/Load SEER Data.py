
import pandas as pd
import numpy as np

# Load SEER esophageal cancer dataset
seer_df = pd.read_csv("export - Sheet2.csv", low_memory=False)

print(f"Shape: {seer_df.shape[0]:,} rows × {seer_df.shape[1]} columns")
print(f"\nColumn names and dtypes:")
for col in seer_df.columns:
    n_missing = seer_df[col].isna().sum()
    pct_missing = 100 * n_missing / len(seer_df)
    print(f"  {col:<45} {str(seer_df[col].dtype):<12} missing: {pct_missing:.1f}%")

# Year of diagnosis range
yr_col = [c for c in seer_df.columns if 'year' in c.lower() and 'diag' in c.lower()]
if yr_col:
    col = yr_col[0]
    print(f"\nYear of Diagnosis range: {seer_df[col].min()} – {seer_df[col].max()}")
else:
    print(f"\nYear columns found: {[c for c in seer_df.columns if 'year' in c.lower()]}")
