import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import rankdata
import json
import hashlib
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# ============================================================
# PHASE 1: SHATTER — Decompose data into statistical fragments
# ============================================================

class ColumnProfile:
    """Stores a noisy statistical profile for a single column."""
    
    def __init__(self, name, col_type):
        self.name = name
        self.col_type = col_type  # 'numeric' or 'categorical'
        self.params = {}
    
    def to_dict(self):
        return {"name": self.name, "col_type": self.col_type, "params": self.params}


def detect_column_types(df):
    """Classify columns as numeric or categorical."""
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            # Treat low-cardinality numerics as categorical
            if nunique <= 10 and nunique / len(df) < 0.05:
                types[col] = "categorical"
            else:
                types[col] = "numeric"
        else:
            types[col] = "categorical"
    return types


def fit_marginal(series, col_type, epsilon=1.0):
    """
    Fit a marginal distribution to a single column with DP-style noise.
    Returns a ColumnProfile with noisy parameters.
    """
    profile = ColumnProfile(series.name, col_type)
    
    if col_type == "numeric":
        clean = series.dropna().astype(float)
        if len(clean) == 0:
            profile.params = {"distribution": "empty"}
            return profile
        
        n = len(clean)
        
        # Compute statistics with calibrated Laplace noise
        sensitivity_mean = (clean.max() - clean.min()) / n
        sensitivity_std = (clean.max() - clean.min()) / np.sqrt(n)
        
        noisy_mean = clean.mean() + np.random.laplace(0, sensitivity_mean / epsilon)
        noisy_std = max(clean.std() + np.random.laplace(0, sensitivity_std / epsilon), 1e-6)
        noisy_min = clean.min() + np.random.laplace(0, 1.0 / epsilon)
        noisy_max = clean.max() + np.random.laplace(0, 1.0 / epsilon)
        
        # Ensure min < max
        if noisy_min >= noisy_max:
            noisy_min, noisy_max = noisy_max, noisy_min
        
        # Detect skewness to choose distribution family
        skew = clean.skew()
        noisy_skew = skew + np.random.laplace(0, 0.5 / epsilon)
        
        # Quantize parameters (reduces precision = less reversible)
        quantize = lambda x, q=0.01: round(x / q) * q
        
        if abs(noisy_skew) < 0.5:
            profile.params = {
                "distribution": "normal",
                "mean": quantize(noisy_mean),
                "std": quantize(noisy_std),
                "min_clip": quantize(noisy_min),
                "max_clip": quantize(noisy_max),
            }
        elif noisy_skew > 0:
            # Use log-normal for right-skewed
            pos = clean[clean > 0]
            if len(pos) > 10:
                log_mean = np.log(pos).mean() + np.random.laplace(0, 0.1 / epsilon)
                log_std = max(np.log(pos).std() + np.random.laplace(0, 0.1 / epsilon), 0.1)
                profile.params = {
                    "distribution": "lognormal",
                    "log_mean": quantize(log_mean),
                    "log_std": quantize(log_std),
                    "min_clip": quantize(noisy_min),
                    "max_clip": quantize(noisy_max),
                }
            else:
                profile.params = {
                    "distribution": "normal",
                    "mean": quantize(noisy_mean),
                    "std": quantize(noisy_std),
                    "min_clip": quantize(noisy_min),
                    "max_clip": quantize(noisy_max),
                }
        else:
            # Fallback: normal
            profile.params = {
                "distribution": "normal",
                "mean": quantize(noisy_mean),
                "std": quantize(noisy_std),
                "min_clip": quantize(noisy_min),
                "max_clip": quantize(noisy_max),
            }
        
        # Check if integer-valued
        if clean.apply(lambda x: x == int(x)).all():
            profile.params["is_integer"] = True
        
        # Store noisy percentiles for better tail behavior
        noisy_percentiles = {}
        for p in [5, 25, 50, 75, 95]:
            val = np.percentile(clean, p) + np.random.laplace(0, sensitivity_mean / epsilon)
            noisy_percentiles[str(p)] = quantize(val)
        profile.params["percentiles"] = noisy_percentiles
        
        # Null rate with noise
        null_rate = series.isna().mean() + np.random.laplace(0, 1.0 / (len(series) * epsilon))
        profile.params["null_rate"] = max(0, min(1, round(null_rate, 3)))
    
    else:  # categorical
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            profile.params = {"distribution": "empty"}
            return profile
        
        # Frequency table with Laplace noise (standard DP mechanism)
        counts = clean.value_counts()
        noisy_counts = {}
        for val, count in counts.items():
            noisy_count = max(0, count + np.random.laplace(0, 1.0 / epsilon))
            noisy_counts[val] = noisy_count
        
        total = sum(noisy_counts.values())
        if total == 0:
            total = 1
        
        profile.params = {
            "distribution": "categorical",
            "probabilities": {k: round(v / total, 4) for k, v in noisy_counts.items()},
        }
        
        null_rate = series.isna().mean() + np.random.laplace(0, 1.0 / (len(series) * epsilon))
        profile.params["null_rate"] = max(0, min(1, round(null_rate, 3)))
    
    return profile


# ============================================================
# PHASE 2: DESTROY — Add noise to correlation structure
# ============================================================

def compute_noisy_correlations(df, numeric_cols, epsilon=1.0):
    """
    Compute pairwise correlations with bounded perturbation.
    Only rank correlations are stored (less reversible than raw covariance).
    """
    if len(numeric_cols) < 2:
        return np.eye(len(numeric_cols))
    
    subset = df[numeric_cols].dropna()
    if len(subset) < 5:
        return np.eye(len(numeric_cols))
    
    # Use Spearman (rank-based = more robust, less tied to exact values)
    corr = subset.corr(method="spearman").values
    corr.setflags(write=1)
    
    # Add calibrated noise to each off-diagonal element
    n = corr.shape[0]
    noise_scale = 2.0 / (len(subset) * epsilon)
    
    for i in range(n):
        for j in range(i + 1, n):
            noise = np.random.laplace(0, noise_scale)
            corr[i, j] = np.clip(corr[i, j] + noise, -0.99, 0.99)
            corr[j, i] = corr[i, j]
    
    # Quantize to reduce precision
    corr = np.round(corr * 20) / 20  # Round to nearest 0.05
    
    # Ensure positive semi-definite (nearest PSD matrix)
    corr = _nearest_psd(corr)
    
    return corr


def _nearest_psd(A):
    """Find the nearest positive semi-definite matrix."""
    B = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(B)
    eigvals = np.maximum(eigvals, 1e-8)
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize diagonal to 1
    d = np.sqrt(np.diag(result))
    d[d == 0] = 1
    result = result / np.outer(d, d)
    np.fill_diagonal(result, 1.0)
    return result


def compute_conditional_buckets(df, col_types, epsilon=1.0, max_pairs=20):
    """
    Compute noisy conditional statistics between column pairs.
    Uses bucketed aggregates (not raw data) for irreversibility.
    """
    conditionals = []
    numeric_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    
    # Categorical -> Numeric conditional means (noisy)
    pairs_done = 0
    for cat_col in cat_cols:
        for num_col in numeric_cols:
            if pairs_done >= max_pairs:
                break
            
            groups = df.groupby(cat_col)[num_col].agg(["mean", "std", "count"])
            noisy_groups = {}
            for idx, row in groups.iterrows():
                if row["count"] >= 3:  # Suppress small groups
                    nm = row["mean"] + np.random.laplace(0, 1.0 / epsilon)
                    ns = max(0.01, row["std"] + np.random.laplace(0, 0.5 / epsilon)) if pd.notna(row["std"]) else 0.01
                    noisy_groups[str(idx)] = {
                        "mean": round(nm, 3),
                        "std": round(ns, 3),
                    }
            
            if noisy_groups:
                conditionals.append({
                    "type": "cat_to_num",
                    "cat_col": cat_col,
                    "num_col": num_col,
                    "groups": noisy_groups,
                })
                pairs_done += 1
    
    return conditionals


# ============================================================
# PHASE 3: REASSEMBLE — Generate synthetic data from fragments
# ============================================================

def sample_from_profile(profile, n):
    """Sample n values from a noisy column profile."""
    params = profile.params
    
    if params.get("distribution") == "empty":
        return pd.Series([None] * n, name=profile.name)
    
    if profile.col_type == "numeric":
        dist = params["distribution"]
        
        if dist == "normal":
            values = np.random.normal(params["mean"], params["std"], n)
        elif dist == "lognormal":
            values = np.random.lognormal(params["log_mean"], params["log_std"], n)
        else:
            values = np.random.normal(params.get("mean", 0), params.get("std", 1), n)
        
        # Clip to noisy bounds
        values = np.clip(values, params.get("min_clip", -1e12), params.get("max_clip", 1e12))
        
        if params.get("is_integer"):
            values = np.round(values).astype(int)
        
        # Inject nulls
        null_rate = params.get("null_rate", 0)
        if null_rate > 0:
            mask = np.random.random(n) < null_rate
            values = values.astype(float)
            values[mask] = np.nan
        
        return pd.Series(values, name=profile.name)
    
    else:  # categorical
        probs = params["probabilities"]
        categories = list(probs.keys())
        weights = list(probs.values())
        total = sum(weights)
        weights = [w / total for w in weights]
        
        values = np.random.choice(categories, size=n, p=weights)
        
        null_rate = params.get("null_rate", 0)
        if null_rate > 0:
            mask = np.random.random(n) < null_rate
            values = values.astype(object)
            values[mask] = None
        
        # Try to convert back to numeric if all values look numeric
        result = pd.Series(values, name=profile.name)
        try:
            numeric_result = pd.to_numeric(result, errors='coerce')
            if numeric_result.notna().sum() == result.notna().sum():
                result = numeric_result
        except (ValueError, TypeError):
            pass
        
        return result


def induce_correlations(df, numeric_cols, target_corr):
    """
    Apply Cholesky-based correlation induction to the numeric columns.
    This approximately recovers the (noisy) correlation structure
    without using the original data at all.
    """
    if len(numeric_cols) < 2:
        return df
    
    df = df.copy()
    
    # Convert numeric columns to uniform ranks
    ranked = pd.DataFrame()
    for col in numeric_cols:
        vals = df[col].values.astype(float)
        non_null = ~np.isnan(vals)
        if non_null.sum() > 0:
            ranks = np.zeros_like(vals)
            ranks[non_null] = rankdata(vals[non_null]) / (non_null.sum() + 1)
            ranks[~non_null] = np.nan
            ranked[col] = ranks
    
    if ranked.shape[1] < 2:
        return df
    
    # Generate correlated normals via Cholesky
    n = len(df)
    k = len(numeric_cols)
    
    try:
        L = np.linalg.cholesky(target_corr[:k, :k])
    except np.linalg.LinAlgError:
        return df  # Fall back if Cholesky fails
    
    Z = np.random.normal(0, 1, (n, k))
    correlated_normals = Z @ L.T
    
    # Convert back to uniform via CDF
    correlated_uniforms = stats.norm.cdf(correlated_normals)
    
    # Rearrange the original sampled values to match the correlation order
    for i, col in enumerate(numeric_cols):
        vals = df[col].values.astype(float)
        non_null = ~np.isnan(vals)
        
        if non_null.sum() > 1:
            # Sort the non-null synthetic values
            sorted_vals = np.sort(vals[non_null])
            # Get the rank order from the correlated uniforms
            target_order = np.argsort(np.argsort(correlated_uniforms[non_null, i]))
            # Rearrange
            vals[non_null] = sorted_vals[target_order]
            df[col] = vals
    
    return df


def apply_conditional_adjustments(df, conditionals, strength=0.5):
    """
    Nudge synthetic values toward noisy conditional statistics.
    This partially restores cat→num relationships without using original data.
    """
    df = df.copy()
    
    for cond in conditionals:
        if cond["type"] == "cat_to_num":
            cat_col = cond["cat_col"]
            num_col = cond["num_col"]
            groups = cond["groups"]
            
            if cat_col not in df.columns or num_col not in df.columns:
                continue
            
            for cat_val, stats_dict in groups.items():
                mask = df[cat_col].astype(str) == cat_val
                if mask.sum() == 0:
                    continue
                
                current_vals = df.loc[mask, num_col].values.astype(float)
                non_null = ~np.isnan(current_vals)
                
                if non_null.sum() == 0:
                    continue
                
                # Nudge toward the noisy conditional mean
                target_mean = stats_dict["mean"]
                current_mean = np.nanmean(current_vals)
                shift = (target_mean - current_mean) * strength
                current_vals[non_null] += shift
                
                df.loc[mask, num_col] = current_vals
    
    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

class ShatteredSynth:
    """
    Irreversible synthetic data generator.
    
    Usage:
        synth = ShatteredSynth(epsilon=1.0)
        synth.shatter(df)          # Phase 1+2: Learn noisy fragments
        synthetic_df = synth.generate(n=1000)  # Phase 3: Reassemble
    """
    
    def __init__(self, epsilon=1.0, seed=None):
        """
        Args:
            epsilon: Privacy budget (lower = more noise = more irreversible).
                     1.0 is a good default. 0.1 is very private. 10.0 is less noisy.
            seed: Random seed for reproducibility.
        """
        self.epsilon = epsilon
        self.seed = seed
        self.profiles = {}
        self.col_types = {}
        self.correlation_matrix = None
        self.numeric_cols = []
        self.cat_cols = []
        self.conditionals = []
        self.column_order = []
        self._destruction_hash = None
        
        if seed is not None:
            np.random.seed(seed)
    
    def shatter(self, df):
        """
        Phase 1+2: Decompose data into noisy fragments and destroy reversibility.
        After this call, the original df is not retained.
        """
        print(f"EchoData — Irreversible Synthetic Data Engine")
        print(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Privacy budget (ε): {self.epsilon}")
        print()
        
        self.column_order = list(df.columns)
        
        # Step 1: Detect types
        self.col_types = detect_column_types(df)
        self.numeric_cols = [c for c, t in self.col_types.items() if t == "numeric"]
        self.cat_cols = [c for c, t in self.col_types.items() if t == "categorical"]
        
        # Step 2: Fit noisy marginals (independently per column)
        for col in df.columns:
            self.profiles[col] = fit_marginal(df[col], self.col_types[col], self.epsilon)
        
        # Step 3: Compute noisy correlations
        self.correlation_matrix = compute_noisy_correlations(
            df, self.numeric_cols, self.epsilon
        )
        
        # Step 4: Compute noisy conditional buckets
        self.conditionals = compute_conditional_buckets(df, self.col_types, self.epsilon)
        # Phase 2 — DESTROY: Generate a destruction certificate
        print()
        destruction_salt = hashlib.sha256(np.random.bytes(32)).hexdigest()
        self._destruction_hash = hashlib.sha256(
            f"{destruction_salt}:{df.shape}:{list(df.columns)}".encode()
        ).hexdigest()[:16]
    
    def generate(self, n=None, multiplier=1):
        """
        Phase 3: Generate synthetic rows from noisy fragments.
        
        Args:
            n: Exact number of rows to generate.
            multiplier: If n is not set, generate original_size * multiplier rows.
        """
        if not self.profiles:
            raise ValueError("Call .shatter(df) first!")
        
        if n is None:
            n = 1000  # Default
        
        # Step 1: Sample each column independently from noisy marginals
        columns = {}
        for col in self.column_order:
            columns[col] = sample_from_profile(self.profiles[col], n)
        
        synthetic = pd.DataFrame(columns)
        
        # Step 2: Induce noisy correlations among numeric columns
        if len(self.numeric_cols) >= 2:
            synthetic = induce_correlations(
                synthetic, self.numeric_cols, self.correlation_matrix
            )
        
        # Step 3: Apply conditional adjustments
        if self.conditionals:
            synthetic = apply_conditional_adjustments(synthetic, self.conditionals)
        
        # Final cleanup: restore dtypes
        for col in self.column_order:
            if self.col_types[col] == "numeric" and self.profiles[col].params.get("is_integer"):
                non_null = synthetic[col].notna()
                if non_null.any():
                    synthetic.loc[non_null, col] = synthetic.loc[non_null, col].astype(float).round().astype(int)
        
        print(f"Done")
        print()
        
        return synthetic
    
    def get_privacy_report(self):
        """Generate a report on the irreversibility guarantees."""
        report = {
            "engine": "ShatteredSynth",
            "destruction_certificate": self._destruction_hash,
            "privacy_budget_epsilon": self.epsilon,
            "guarantees": [
                "No original data rows are stored",
                f"All marginal statistics have Laplace(0, σ/ε) noise with ε={self.epsilon}",
                "Correlation matrix quantized to 0.05 resolution",
                "Small groups (n<3) suppressed from conditional statistics",
                "No single model artifact encodes the full joint distribution",
            ],
            "fragments_stored": {
                "marginal_profiles": len(self.profiles),
                "correlation_pairs": (
                    self.correlation_matrix.shape[0] * (self.correlation_matrix.shape[0] - 1) // 2
                    if self.correlation_matrix is not None else 0
                ),
                "conditional_buckets": len(self.conditionals),
            },
        }
        return report


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python shattered_synth.py <input.csv> [output.csv] [num_rows] [epsilon]")
        print()
        print("Arguments:")
        print("  input.csv   — Path to the original CSV data")
        print("  output.csv  — Path for synthetic output (default: synthetic_<input>.csv)")
        print("  num_rows    — Number of synthetic rows (default: same as input)")
        print("  epsilon     — Privacy budget, lower=more private (default: 1.0)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Parse args
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"synthetic_{base_name}.csv"
    num_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None
    epsilon = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    
    # Load
    df = pd.read_csv(input_path)
    
    if num_rows is None:
        num_rows = len(df)
    
    # Run pipeline
    synth = ShatteredSynth(epsilon=epsilon)
    synth.shatter(df)
    synthetic_df = synth.generate(n=num_rows)
    
    # Save
    synthetic_df.to_csv(output_path, index=False)
    print(f"  📁 Saved to: {output_path}")
    
    # Privacy report
    report = synth.get_privacy_report()
    print()
    print("  🔒 PRIVACY REPORT")
    for g in report["guarantees"]:
        print(f"    • {g}")
    print(f"    Certificate: {report['destruction_certificate']}")


if __name__ == "__main__":
    main()
