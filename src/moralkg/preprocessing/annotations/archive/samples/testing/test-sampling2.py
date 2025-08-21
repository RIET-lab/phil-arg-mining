# sampling2.py

# Hybrid stratified/cluster sampling of the phil-papers dataset:

# Pipeline:
  # 1) Coarse stratification on year, category, and author buckets
  # 2) Semantic clustering on title embeddings (TF-IDF) within each stratum
  # 3) Neyman allocation of draws to each stratum & cluster
  # 4) Random sampling of unique paper IDs
  # 5) Validation based on:
    # - KS test for publication years
    # - Chi-square goodness-of-fit for category & author distributions
    # - Energy distance for title semantics

# See below for args

import argparse
import logging
import os
import re
from typing import Tuple, Optional, Any, cast

import numpy as np
import pandas as pd
import rootutils
from scipy.stats import chisquare, energy_distance, ks_2samp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.sparse import csr_matrix
from flexidate import parse as fl_parse


# Setup project root
ROOT = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Hybrid stratified + clustered sampling of phil-papers dataset")
    p.add_argument("-i", "--input",
                   default="data/metadata/2025-07-09-en-combined-metadata.csv",
                   help="Input CSV of dataset")
    p.add_argument("-o", "--output",
                   default="data/samples/n100.csv",
                   help="Output CSV for sampled subset")
    p.add_argument("-n", "--sample-size", type=int, default=100,
                   help="How many papers to sample")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--max-clusters", type=int, default=3,
                   help="Max semantic clusters per stratum")
    return p.parse_args()


def parse_year(year) -> Optional[int]:
    """Attempt to extract the year with flexidate first then with regex fallback."""
    s = str(year)
    if not s:
        return None
    
    # Try flexidate first
    try:
        parsed = fl_parse(s)
        if parsed and hasattr(parsed, 'year') and parsed.year:
            year = parsed.year
            return -int(year[1:]) if year.startswith('-') else int(year)
    except:
        pass
    
    # Simple regex search fallback for 1-4 digit numbers
    match = re.search(r'\b(\d{1,4})\b', s)
    if match:
        year = int(match.group(1))
        # Only accept years in reasonable range
        if 1 <= year <= 2026:
            return year
    
    return None


def preprocess(df: pd.DataFrame):
    """
    1) Clean & bin publication year into quartiles
    2) Parse primary subject category (first of any semicolon list)
    3) Parse authors list & bucket into top-K vs "OTHER" (with balanced buckets)
    """
    # 4.1) Parse and clean year
    df["year_clean"] = df["year"].apply(parse_year).astype("Int64")
    # We use quartiles so each temporal stratum has ~equal N
    year_bins = pd.qcut(
        df["year_clean"].dropna(),
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )
    # Map back to full dataframe, keeping NaN for missing years
    df["year_bin"] = df["year_clean"].map(dict(zip(df["year_clean"].dropna(), year_bins)))

    # 4.2) Primary category: explode semicolon lists, keep first as main axis
    #    We avoid full explosion & dedup by choosing primary category,
    #    trading off multi‐membership for simplicity.
    df["category_names"] = (
        df["category_names"]
        .fillna("")
        .astype(str)
        .str.split(r"\s*;\s*")
    )
    df["primary_category"] = df["category_names"].apply(
        lambda lst: lst[0] if lst and lst[0] else "UNKNOWN"
    )

    # 4.3) Authors list & balanced top-K bucketing
    df["authors"] = (
        df["authors"]
        .fillna("")
        .astype(str)
        .str.split(r"\s*;\s*")
    )
    # Count all appearances
    all_authors = pd.Series(
        [a for sub in df["authors"] for a in sub if a],
        name="author"
    )
    freq = all_authors.value_counts()
    
    # Use statistically principled bucketing for author productivity distribution
    
    # Convert to numpy array for analysis
    author_counts = np.array(freq.values)
    
    # Distribution analysis
    mean_papers = float(author_counts.mean())
    median_papers = float(np.median(author_counts))
    max_papers = int(author_counts.max())
    q75 = float(np.percentile(author_counts, 75))
    q90 = float(np.percentile(author_counts, 90))
    
    logger.info(f"Author productivity stats: mean={mean_papers:.1f}, median={median_papers:.1f}, "
               f"75th percentile={q75:.1f}, 90th percentile={q90:.1f}, max={max_papers}")
    
    # Check if distribution exhibits power-law characteristics
    # Power laws: most values are small, few are very large
    gini_coeff = 2 * np.sum(np.arange(1, len(author_counts) + 1) * np.sort(author_counts)) / (len(author_counts) * np.sum(author_counts)) - (len(author_counts) + 1) / len(author_counts)
    logger.info(f"Gini coefficient: {gini_coeff:.3f}")
    
    # Use data-driven breakpoints based on distribution characteristics
    # If high inequality (Gini > 0.7), use more conservative breakpoints
    if gini_coeff > 0.7:
        # High inequality - use natural breaks
        breakpoint_1 = max(2, int(median_papers))
        breakpoint_2 = max(breakpoint_1 + 1, int(q90))
        logger.info("High inequality detected - using conservative breakpoints")
    else:
        # Lower inequality - use more balanced approach
        breakpoint_1 = max(2, int(q75))
        breakpoint_2 = max(breakpoint_1 + 1, int(q90))
        logger.info("Moderate inequality - using balanced breakpoints")
    
    logger.info(f"Final breakpoints: OCCASIONAL (1-{breakpoint_1}), REGULAR ({breakpoint_1+1}-{breakpoint_2}), PROLIFIC ({breakpoint_2+1}+)")
    
    # Create author groups based on data-driven productivity thresholds
    def get_author_group(author_name):
        if author_name not in freq.index:
            return "UNKNOWN"
        count = freq[author_name]
        if count > breakpoint_2:
            return "PROLIFIC"
        elif count > breakpoint_1:
            return "REGULAR"
        else:
            return "OCCASIONAL"

    # Assign each paper to author productivity group
    def bucket_author(author_list):
        # Find the most prolific author for this paper
        best_group = "OCCASIONAL"  # Default
        for author in author_list:
            if author:  # Non-empty author
                group = get_author_group(author)
                # Rank groups: PROLIFIC > REGULAR > OCCASIONAL > UNKNOWN
                if group == "PROLIFIC":
                    return "PROLIFIC"
                elif group == "REGULAR" and best_group in ["OCCASIONAL", "UNKNOWN"]:
                    best_group = "REGULAR"
                elif group == "UNKNOWN" and best_group == "OCCASIONAL":
                    best_group = "UNKNOWN"
        return best_group

    df["author_bucket"] = df["authors"].apply(bucket_author)
    
    # Log bucket sizes for verification
    bucket_counts = df["author_bucket"].value_counts()
    total_papers = len(df)
    
    logger.info("Author productivity bucket distribution:")
    for group in ["PROLIFIC", "REGULAR", "OCCASIONAL", "UNKNOWN"]:
        count = int(bucket_counts.get(group, 0))
        pct = 100.0 * count / total_papers if total_papers > 0 else 0.0
        logger.info(f"  {group}: {count:,} papers ({pct:.1f}%)")

    # 4.4) Ensure every paper has a unique ID for sampling
    if "identifier" in df.columns:
        df["paper_id"] = df["identifier"]
    else:
        df["paper_id"] = df.index.astype(str)

    return df


def build_strata(df: pd.DataFrame):
    """
    Combine the three coarse axes into a single 'stratum' label:
      year_bin | primary_category | author_bucket
    """
    df["stratum"] = (
        df["year_bin"].astype(str) + "|"
        + df["primary_category"].astype(str) + "|"
        + df["author_bucket"].astype(str)
    )
    return df


def cluster_within_strata(df: pd.DataFrame, max_clusters: int):
    """
    Within each stratum, cluster titles by TF-IDF semantics.
    Returns:
      - df with a 'cluster' column
      - cluster_info DataFrame with columns:
          stratum, cluster, N_si (size), sigma_si (semantic variance)
    """
    vectorizer = TfidfVectorizer(max_features=1000)  # Reduced for performance
    records = []  # to collect per-cluster info
    df["cluster"] = np.nan

    strata_groups = list(df.groupby("stratum", sort=False))
    logger.info(f"Processing {len(strata_groups)} strata for clustering...")

    # Process each coarse stratum separately
    for i, (stratum, sub) in enumerate(strata_groups):
        if i % 50 == 0:  # Progress logging
            logger.info(f"Processing stratum {i+1}/{len(strata_groups)}")
            
        titles = sub["title"].fillna("").tolist()
        if not titles:
            continue

        # 6.1) Fit TF-IDF on these titles
        try:
            X_raw = vectorizer.fit_transform(titles)
            # Type guard for sklearn sparse matrix
            if X_raw is None or not hasattr(X_raw, 'shape') or X_raw.shape[0] == 0:
                continue
            # Cast to proper type after validation
            X = cast(csr_matrix, X_raw)
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to vectorize titles in stratum {stratum}: {e}")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error in vectorization for stratum {stratum}: {e}")
            continue

        # After type guard and cast, X is a valid sparse matrix
        n_samples = X.shape[0]
        
        # 6.2) Choose number of clusters: at most max_clusters, but no larger than N
        k = min(max_clusters, n_samples)
        if k <= 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            try:
                km = KMeans(n_clusters=k, random_state=0, n_init='auto')  # Use auto for efficiency
                labels = km.fit_predict(X)
            except Exception as e:
                logger.warning(f"Clustering failed for stratum {stratum}: {e}")
                labels = np.zeros(n_samples, dtype=int)

        # 6.3) Compute semantic variance of this stratum
        #      (sum of feature variances—trace of covariance)
        try:
            sigma_s = float(np.var(X.toarray(), axis=0).sum())
        except Exception:
            sigma_s = 1.0  # Default fallback

        # 6.4) Record cluster assignment & info
        df.loc[sub.index, "cluster"] = labels
        for cluster_id in range(max(1, k)):
            idxs = np.where(labels == cluster_id)[0]
            N_si = len(idxs)
            records.append({
                "stratum": stratum,
                "cluster": cluster_id,
                "N_si": N_si,
                "sigma_si": sigma_s
            })

    cluster_info = pd.DataFrame(records)
    logger.info(f"Generated {len(cluster_info)} clusters across all strata")
    return df, cluster_info


def allocate_samples(cluster_info: pd.DataFrame, n: int):
    """
    Compute weight = N_si * sigma_si for each cluster, then
    allocate n_si = round(n * weight / sum(weights)). Adjust
    for rounding error so sum(n_si) == n.
    """
    info = cluster_info.copy()
    info["weight"] = info["N_si"] * info["sigma_si"]
    total_w = info["weight"].sum()

    if total_w > 0:
        info["n_si"] = (n * info["weight"] / total_w).round().astype(int)
    else:
        # If zero variance everywhere, fall back to proportional by N_si
        info["n_si"] = (n * info["N_si"] / info["N_si"].sum()).round().astype(int)

    # Fix rounding drift
    diff = n - int(info["n_si"].sum())
    for _ in range(abs(diff)):
        if diff > 0:
            idx = info["weight"].idxmax()
            info.at[idx, "n_si"] += 1
        else:
            idx = info["weight"].idxmin()
            info.at[idx, "n_si"] -= 1

    return info


def sample_from_clusters(df: pd.DataFrame, alloc: pd.DataFrame, seed: int):
    """
    For each (stratum, cluster), sample n_si unique paper_ids,
    excluding any already-picked papers, to avoid duplicates.
    """
    np.random.seed(seed)
    chosen_ids = []
    for _, row in alloc.iterrows():
        s, c, draw = row["stratum"], row["cluster"], int(row["n_si"])
        pool = df[(df["stratum"] == s) & (df["cluster"] == c)]
        pool_ids = pool["paper_id"].unique().tolist()
        # Exclude already chosen
        pool_ids = [pid for pid in pool_ids if pid not in chosen_ids]
        if draw > len(pool_ids):
            logger.warning(
                f"Stratum={s!r}, cluster={c}: requested {draw} but only {len(pool_ids)} available."
            )
            draw = len(pool_ids)
        if draw <= 0:
            continue
        picks = np.random.choice(pool_ids, size=draw, replace=False)
        chosen_ids.extend(picks.tolist())

    # Final sample: unique papers
    sample_df = df[df["paper_id"].isin(chosen_ids)].drop_duplicates(subset="paper_id")
    return sample_df


def validate_sample(full: pd.DataFrame, sample: pd.DataFrame):
    """
    Run:
      - KS test on 'year_clean'
      - Chi-square on 'primary_category' and 'author_bucket'
      - Energy distance on title-to-centroid cosine distances
    Logs results so you can confirm representativeness.
    """
    # 9.1) KS test for publication year
    pop_years = full["year_clean"].dropna().astype(float)
    samp_years = sample["year_clean"].dropna().astype(float)
    if len(pop_years) > 0 and len(samp_years) > 0:
        D, p_ks = ks_2samp(pop_years, samp_years)
        logger.info(f"KS test (year): D={D:.3f}, p={p_ks:.3f}")

    # 9.2) Chi-square for primary_category
    pop_ct = full["primary_category"].value_counts()
    samp_ct = sample["primary_category"].value_counts()
    common = pop_ct.index.intersection(samp_ct.index)
    if common is not None and len(common) > 0:
        exp = (pop_ct[common] / pop_ct.sum()) * len(sample)
        obs = samp_ct.reindex(common).fillna(0)
        try:
            chi2, p_chi = chisquare(f_obs=obs, f_exp=exp)
            logger.info(f"Chi-square (category): χ²={chi2:.3f}, p={p_chi:.3f}")
        except Exception as e:
            logger.warning(f"Chi-square test for category failed: {e}")

    # 9.3) Chi-square for author_bucket
    pop_ab = full["author_bucket"].value_counts()
    samp_ab = sample["author_bucket"].value_counts()
    common = pop_ab.index.intersection(samp_ab.index)
    if common is not None and len(common) > 0:
        exp = (pop_ab[common] / pop_ab.sum()) * len(sample)
        obs = samp_ab.reindex(common).fillna(0)
        try:
            chi2, p_ab = chisquare(f_obs=obs, f_exp=exp)
            logger.info(f"Chi-square (author bucket): χ²={chi2:.3f}, p={p_ab:.3f}")
        except Exception as e:
            logger.warning(f"Chi-square test for author bucket failed: {e}")

    # 9.4) Energy distance on title semantics
    #    Compute distance‐to‐centroid for full vs. sample
    try:
        vec = TfidfVectorizer(max_features=1000)
        full_X_raw = vec.fit_transform(full["title"].fillna(""))
        if full_X_raw is not None and hasattr(full_X_raw, 'shape') and full_X_raw.shape[0] > 0:
            full_X = cast(csr_matrix, full_X_raw)
            centroid = full_X.mean(axis=0)
            full_d = cosine_distances(full_X, centroid).ravel()

            samp_X_raw = vec.transform(sample["title"].fillna(""))
            samp_X = cast(csr_matrix, samp_X_raw)
            samp_d = cosine_distances(samp_X, centroid).ravel()

            ed = energy_distance(full_d, samp_d)
            logger.info(f"Energy distance (title semantics): {ed:.5f}")
        else:
            logger.warning("Cannot compute title semantic validation - no valid titles")
    except Exception as e:
        logger.warning(f"Title semantic validation failed: {e}")


def main():
    # Parse args, get seed
    args = parse_args()
    np.random.seed(args.seed)
    logger.info(f"Random seed is {args.seed}")

    # Load dataset
    df = pd.read_csv(ROOT / args.input)
    logger.info(f"Loaded {len(df)} papers from dataset")

    # Preprocess: clean year, extract axes, bucket authors
    df = preprocess(df)
    logger.info(f"Dataset preprocessed")

    # Build coarse strata key
    df = build_strata(df)
    logger.info(f"Dataset stratified across 
")
    
    strata_counts = df.groupby("stratum").size()
    logger.info(f"Created {len(strata_counts)} strata, sizes: min={strata_counts.min()}, "
               f"max={strata_counts.max()}, mean={strata_counts.mean():.1f}")

    # Semantic clustering within strata
    df, cluster_info = cluster_within_strata(df, args.max_clusters)

    # Neyman allocation of draws
    alloc = allocate_samples(cluster_info, args.sample_size)

    # Draw the sample
    sample_df = sample_from_clusters(df, alloc, args.seed)
    
    if sample_df is None or len(sample_df) == 0:
        logger.error("Failed to generate sample or sample is empty")
        return

    logger.info(f"Successfully sampled {len(sample_df)} papers")

    # Save sampled subset
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sample_df.to_csv(args.output, index=False)
    logger.info(f"Sample saved to {args.output}")

    # Validate representativeness
    validate_sample(df, sample_df)

if __name__ == "__main__":
    main()
