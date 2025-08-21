#!/usr/bin/env python3
"""
Comparison script that runs all clustering/sampling methods and generates a summary.
Following KISS principle - simple comparison of all methods.
"""

import pandas as pd
import rootutils
from pathlib import Path
import subprocess
import sys
import logging
from datetime import datetime

def setup_comparison_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for the comparison runner."""
    log_file = output_dir / "comparison_runner.log"
    
    logger = logging.getLogger("comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_script(script_path: Path, logger: logging.Logger) -> bool:
    """Run a clustering script and return success status."""
    try:
        logger.info(f"Running {script_path.name}...")
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            logger.info(f"Success: {script_path.name} completed successfully")
            return True
        else:
            logger.error(f"Error: {script_path.name} failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"Error: {script_path.name} timed out after 30 minutes")
        return False
    except Exception as e:
        logger.error(f"Error: {script_path.name} failed with exception: {e}")
        return False

def parse_evaluation_summary(summary_file: Path) -> dict:
    """Parse evaluation summary file and extract key metrics."""
    metrics = {}
    
    if not summary_file.exists():
        return metrics
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
            
        # Extract clustering quality metrics
        for line in content.split('\n'):
            if ':' in line and not line.startswith('=') and not line.startswith('-'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    # Try to convert to float
                    metrics[key] = float(value)
                except ValueError:
                    # Keep as string if can't convert
                    metrics[key] = value
                    
    except Exception as e:
        print(f"Error parsing {summary_file}: {e}")
    
    return metrics

def create_comparison_table(output_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    """Create a comparison table of all methods."""
    
    methods = [
        ('TF-IDF Title', 'tfidf_title_evaluation_summary.txt'),
        ('TF-IDF Subset', 'tfidf_subset_evaluation_summary.txt'),
        ('TF-IDF All', 'tfidf_all_evaluation_summary.txt'),
        # Note: Gower methods removed - O(n²) complexity unsuitable for 71K dataset
    ]
    
    comparison_data = []
    
    for method_name, summary_file in methods:
        summary_path = output_dir / summary_file
        metrics = parse_evaluation_summary(summary_path)
        
        # Create row for this method
        row = {'Method': method_name}
        row.update(metrics)
        comparison_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    comparison_file = output_dir / 'method_comparison.csv'
    df.to_csv(comparison_file, index=False)
    logger.info(f"Comparison table saved to {comparison_file}")
    
    return df

def generate_summary_report(df: pd.DataFrame, output_dir: Path, logger: logging.Logger):
    """Generate a human-readable summary report."""
    
    report_file = output_dir / 'comparison_summary.txt'
    
    with open(report_file, 'w') as f:
        f.write("CLUSTERING METHOD COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("CLUSTERING QUALITY METRICS:\n")
        f.write("-" * 30 + "\n")
        
        # Davies-Bouldin (lower is better)
        if 'davies_bouldin' in df.columns:
            best_db = df.loc[df['davies_bouldin'].idxmin()]
            f.write(f"Best Davies-Bouldin Index: {best_db['Method']} ({best_db['davies_bouldin']:.4f})\n")
        
        # Calinski-Harabasz (higher is better)
        if 'calinski_harabasz' in df.columns:
            best_ch = df.loc[df['calinski_harabasz'].idxmax()]
            f.write(f"Best Calinski-Harabasz Index: {best_ch['Method']} ({best_ch['calinski_harabasz']:.4f})\n")
        
        # Silhouette (higher is better)
        if 'silhouette' in df.columns:
            best_sil = df.loc[df['silhouette'].idxmax()]
            f.write(f"Best Silhouette Score: {best_sil['Method']} ({best_sil['silhouette']:.4f})\n")
        
        f.write("\nREPRESENTATIVENESS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        # Count successful KS tests (p-value > 0.05 means distributions are similar)
        ks_columns = [col for col in df.columns if col.endswith('_ks_pval')]
        if ks_columns:
            for col in ks_columns:
                feature = col.replace('_ks_pval', '')
                good_methods = df[df[col] > 0.05]['Method'].tolist()
                f.write(f"{feature}: {len(good_methods)} methods passed KS test\n")
        
        f.write("\nRECOMMENDATION:\n")
        f.write("-" * 30 + "\n")
        f.write("Based on the evaluation metrics above:\n")
        f.write("1. Check clustering quality metrics (Davies-Bouldin, Calinski-Harabasz, Silhouette)\n")
        f.write("2. Verify representativeness through KS tests and Chi-squared tests\n")
        f.write("3. Consider computational efficiency for your use case\n")
        f.write("4. Review the enhanced MDS plots to visualize sample coverage\n")
    
    logger.info(f"Summary report saved to {report_file}")

def main():
    # Set up paths
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    script_dir = root / 'data' / 'scripts' / 'annotations' / 'sampling'
    output_dir = root / 'data' / 'annotations' / 'samples'
    
    # Set up logging
    logger = setup_comparison_logging(output_dir)
    
    logger.info("Starting comprehensive clustering method comparison...")
    
    # List of scripts to run
    scripts = [
        # script_dir / 'test-tf-idf-title.py',
        # script_dir / 'test-tf-idf-subset.py', 
        # script_dir / 'test-tf-idf-all.py',
        script_dir / 'test-gower-subset.py',
        # script_dir / 'test-gower-all.py',
    ]
    
    # Run all scripts
    success_count = 0
    for script in scripts:
        if script.exists():
            if run_script(script, logger):
                success_count += 1
        else:
            logger.warning(f"Script not found: {script}")
    
    logger.info(f"Completed {success_count}/{len(scripts)} scripts successfully")
    
    # Create comparison table
    logger.info("Creating comparison table...")
    df = create_comparison_table(output_dir, logger)
    
    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(df, output_dir, logger)
    
    logger.info("Comparison complete! Check the following files:")
    logger.info(f"  - method_comparison.csv: Raw comparison data")
    logger.info(f"  - comparison_summary.txt: Human-readable summary")
    logger.info(f"  - *_evaluation.log: Detailed logs for each method")
    logger.info(f"  - *_mds_with_samples.png: Enhanced visualizations")

if __name__ == "__main__":
    main() 