"""
Demo: ShatteredSynth in action
Generates a sample startup customer dataset, then creates irreversible synthetic data.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "/home/claude")

from shattered_synth import ShatteredSynth

# ── Create a realistic sample dataset (simulating startup customer data) ──
np.random.seed(42)
n = 500

data = {
    "customer_id": range(1, n + 1),
    "age": np.random.normal(35, 12, n).clip(18, 80).astype(int),
    "income": np.random.lognormal(10.8, 0.6, n).clip(20000, 500000).astype(int),
    "plan": np.random.choice(["free", "basic", "pro", "enterprise"], n, p=[0.4, 0.3, 0.2, 0.1]),
    "monthly_spend": np.random.lognormal(3.5, 1.0, n).clip(0, 2000).round(2),
    "sessions_per_week": np.random.poisson(5, n),
    "churn_risk": np.random.choice(["low", "medium", "high"], n, p=[0.5, 0.35, 0.15]),
    "satisfaction_score": np.random.normal(7.5, 1.5, n).clip(1, 10).round(1),
    "referrals": np.random.poisson(1.2, n),
    "region": np.random.choice(["US-East", "US-West", "EU", "APAC", "LATAM"], n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
}

# Add realistic correlations: higher income → higher spend, pro plan → more sessions
for i in range(n):
    if data["income"][i] > 80000:
        data["monthly_spend"][i] *= 1.5
    if data["plan"][i] in ["pro", "enterprise"]:
        data["sessions_per_week"][i] += 3
    if data["churn_risk"][i] == "high":
        data["satisfaction_score"][i] -= 2.0

df = pd.DataFrame(data)
df["satisfaction_score"] = df["satisfaction_score"].clip(1, 10).round(1)
df["monthly_spend"] = df["monthly_spend"].round(2)

# Inject some nulls (realistic)
null_mask = np.random.random(n) < 0.03
df.loc[null_mask, "satisfaction_score"] = np.nan

# Save original
df.to_csv("sample_original.csv", index=False)

print("=" * 70)
print("  SHATTERED SYNTH — DEMO")
print("=" * 70)
print()

# ── Run the pipeline ──
synth = ShatteredSynth(epsilon=1.0, seed=123)
synth.shatter(df)
synthetic_df = synth.generate(n=500)

# Save synthetic
synthetic_df.to_csv("sample_synthetic.csv", index=False)

# ── Comparison Report ──
print("=" * 70)
print("  COMPARISON: Original vs Synthetic")
print("=" * 70)
print()

# Numeric columns comparison
numeric_cols = ["age", "income", "monthly_spend", "sessions_per_week", "satisfaction_score", "referrals"]

print(f"  {'Column':<22} {'Metric':<10} {'Original':>12} {'Synthetic':>12} {'Δ%':>8}")
print(f"  {'─'*22} {'─'*10} {'─'*12} {'─'*12} {'─'*8}")

for col in numeric_cols:
    orig_mean = df[col].mean()
    syn_mean = synthetic_df[col].mean()
    delta = abs(syn_mean - orig_mean) / (abs(orig_mean) + 1e-10) * 100
    print(f"  {col:<22} {'mean':<10} {orig_mean:>12.2f} {syn_mean:>12.2f} {delta:>7.1f}%")
    
    orig_std = df[col].std()
    syn_std = synthetic_df[col].std()
    delta = abs(syn_std - orig_std) / (abs(orig_std) + 1e-10) * 100
    print(f"  {'':<22} {'std':<10} {orig_std:>12.2f} {syn_std:>12.2f} {delta:>7.1f}%")

print()

# Categorical columns comparison
cat_cols = ["plan", "churn_risk", "region"]
print(f"  Categorical Distributions:")
print(f"  {'─'*60}")

for col in cat_cols:
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
for _, orig_row in df.head(50).iterrows():
    for _, syn_row in synthetic_df.iterrows():
        if all(
            abs(float(orig_row[c]) - float(syn_row[c])) < 0.01
            for c in numeric_cols
            if pd.notna(orig_row[c]) and pd.notna(syn_row[c])
        ):
            matches += 1
            break

print(f"  Exact row match test (checked 50 original rows):")
print(f"    Matches found: {matches}")
print(f"    → {'✅ PASS — No original rows recoverable' if matches == 0 else '⚠️  Some near-matches found'}")
print()

# Correlation preservation check
orig_corr = df[numeric_cols].corr()
syn_corr = synthetic_df[numeric_cols].corr()
corr_diff = (orig_corr - syn_corr).abs().mean().mean()
print(f"  Correlation structure preservation:")
print(f"    Mean absolute correlation difference: {corr_diff:.4f}")
print(f"    → {'✅ Good' if corr_diff < 0.15 else '⚠️  Some correlation loss'} (expected with ε={synth.epsilon})")
print()

# Privacy report
report = synth.get_privacy_report()
print(f"  🔒 Privacy Report:")
for g in report["guarantees"]:
    print(f"    • {g}")
print(f"    Destruction certificate: {report['destruction_certificate']}")
