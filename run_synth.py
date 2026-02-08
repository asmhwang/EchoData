import pandas as pd
import numpy as np
import sys
from shattered_synth import ShatteredSynth
from clean_data import DataCleaner

# Load data
df = pd.read_csv('Ecommerce_Consumer_Behavior_Analysis_Data.csv')
print(f"Original data shape: {df.shape}")

# Clean the data first
cleaner = DataCleaner()
df = cleaner.clean_data(df)

print(f"After cleaning: {df.shape}")

# Check if we have data left
if len(df) == 0:
    print("ERROR: All rows were removed during cleaning!")
    print("This likely means your data has issues. Check the original CSV file.")
    sys.exit(1)

if len(df) < 100:
    print(f"WARNING: Only {len(df)} rows remaining. This might not be enough for synthesis.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

print("=" * 70)
print("DATA SYNTHESIZER")
print("=" * 70)
print()

# ── Run the pipeline ──
synth = ShatteredSynth(epsilon=1.0, seed=123)
synth.shatter(df)
synthetic_df = synth.generate(n=min(500, len(df)))  # Don't generate more than we have

# Save synthetic
synthetic_df.to_csv("Synthetic_Ecommerce_Consumer_Behavior_Analysis_Data.csv", index=False)

# ── Comparison Report ──
print("=" * 70)
print("  COMPARISON: Original vs Synthetic")
print("=" * 70)
print()

# Numeric columns comparison
numeric_cols = [
    'Age',
    'Purchase_Amount',
    'Frequency_of_Purchase',
    'Product_Rating',
    'Return_Rate',
    'Customer_Satisfaction'
]

print(f"  {'Column':<45} {'Metric':<10} {'Original':>12} {'Synthetic':>12} {'Δ%':>8}")
print(f"  {'─'*45} {'─'*10} {'─'*12} {'─'*12} {'─'*8}")

for col in numeric_cols:
    if col in df.columns:
        orig_mean = df[col].mean()
        syn_mean = synthetic_df[col].mean()
        delta = abs(syn_mean - orig_mean) / (abs(orig_mean) + 1e-10) * 100
        print(f"  {col:<45} {'mean':<10} {orig_mean:>12.2f} {syn_mean:>12.2f} {delta:>7.1f}%")
        
        orig_std = df[col].std()
        syn_std = synthetic_df[col].std()
        delta = abs(syn_std - orig_std) / (abs(orig_std) + 1e-10) * 100
        print(f"  {'':<45} {'std':<10} {orig_std:>12.2f} {syn_std:>12.2f} {delta:>7.1f}%")

print()

# Categorical columns comparison
cat_cols = [
    'Purchase_Category',
]

print(f"  Categorical Distributions:")
print(f"  {'─'*60}")

for col in cat_cols:
    if col in df.columns:
        print(f"\n  {col}:")
        orig_dist = df[col].value_counts(normalize=True).sort_index()
        syn_dist = synthetic_df[col].value_counts(normalize=True).sort_index()
        
        all_vals = sorted(set(list(orig_dist.index) + list(syn_dist.index)))
        for val in all_vals:
            orig_p = orig_dist.get(val, 0)
            syn_p = syn_dist.get(val, 0)
            print(f"    {val:<15} Original: {orig_p:.1%}   Synthetic: {syn_p:.1%}")

print()

# ── Irreversibility check ──
print("=" * 70)
print("  IRREVERSIBILITY VERIFICATION")
print("=" * 70)
print()

# Check: Can we find any original row in the synthetic data?
matches = 0
check_rows = min(50, len(df))
for _, orig_row in df.head(check_rows).iterrows():
    for _, syn_row in synthetic_df.iterrows():
        if all(
            abs(float(orig_row[c]) - float(syn_row[c])) < 0.01
            for c in numeric_cols
            if c in df.columns and pd.notna(orig_row[c]) and pd.notna(syn_row[c])
        ):
            matches += 1
            break

print(f"  Exact row match test (checked {check_rows} original rows):")
print(f"    Matches found: {matches}")
print(f"    → {'PASS — No original rows recoverable' if matches == 0 else '⚠️  Some near-matches found'}")
print()

# Correlation preservation check
numeric_cols_present = [c for c in numeric_cols if c in df.columns]
if len(numeric_cols_present) >= 2:
    orig_corr = df[numeric_cols_present].corr()
    syn_corr = synthetic_df[numeric_cols_present].corr()
    corr_diff = (orig_corr - syn_corr).abs().mean().mean()
    
    print(f"  Correlation structure preservation:")
    print(f"    Mean absolute correlation difference: {corr_diff:.4f}")
    print(f"    → {'Good' if corr_diff < 0.15 else 'Some correlation loss'} (expected with ε={synth.epsilon})")
    print()

# Privacy report
report = synth.get_privacy_report()
print(f"  Privacy Report:")
for g in report["guarantees"]:
    print(f"    • {g}")
print(f"    Destruction certificate: {report['destruction_certificate']}")
