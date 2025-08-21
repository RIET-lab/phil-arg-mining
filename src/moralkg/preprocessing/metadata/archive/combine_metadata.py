#!/usr/bin/env python3
"""
Combine PhilPapers metadata from multiple sources into a single comprehensive CSV.

This script merges:
1. Paper metadata from OAI-PMH XML parsing (title, authors, year, etc.)
2. Category assignments from paper_categories_parsed.csv
3. Category names and hierarchy from categories_parsed.csv

Output: A CSV with complete paper metadata including category IDs and names.
Only includes papers that exist in the OAI-PMH parsed CSV.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set
import sys


class SimplifiedMetadataCombiner:
    def __init__(self):
        self.categories = {}  # category_id -> category_info
        self.paper_categories = {}  # paper_id -> [category_ids]
        self.paper_metadata = {}  # paper_id -> metadata_dict
        
    def load_categories_csv(self, categories_csv: str) -> bool:
        """Load category information from categories_parsed.csv"""
        print(f"Loading categories from: {categories_csv}")
        
        try:
            with open(categories_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    cat_id = int(row['category_id'])
                    self.categories[cat_id] = {
                        'id': cat_id,
                        'name': row['name'],
                        'parent_ids': [int(x.strip()) for x in row['parent_ids'].split(';') if x.strip()] if row['parent_ids'] else [],
                        'primary_parent': int(row['primary_parent']) if row['primary_parent'] else None,
                        'parent_names': row['parent_names'].split(';') if row['parent_names'] else [],
                        'primary_parent_name': row['primary_parent_name'] if row['primary_parent_name'] else None
                    }
            
            print(f"Loaded {len(self.categories)} categories")
            return True
            
        except Exception as e:
            print(f"Error loading categories CSV: {e}")
            return False
    
    def load_paper_categories_csv(self, paper_categories_csv: str) -> bool:
        """Load paper-to-category mappings from paper_categories_parsed.csv"""
        print(f"Loading paper categories from: {paper_categories_csv}")
        
        try:
            with open(paper_categories_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    paper_id = row['paper_id']
                    category_ids = [int(x.strip()) for x in row['category_ids'].split(',') if x.strip()] if row['category_ids'] else []
                    self.paper_categories[paper_id] = category_ids
            
            print(f"Loaded {len(self.paper_categories)} paper-category mappings")
            return True
            
        except Exception as e:
            print(f"Error loading paper categories CSV: {e}")
            return False
    
    def load_paper_metadata_csv(self, metadata_csv: str) -> bool:
        """Load paper metadata from the OAI-PMH parsed CSV"""
        print(f"Loading paper metadata from: {metadata_csv}")
        
        try:
            with open(metadata_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    paper_id = row.get('identifier', '').strip()
                    if paper_id:
                        self.paper_metadata[paper_id] = dict(row)
            
            print(f"Loaded metadata for {len(self.paper_metadata)} papers")
            return True
            
        except Exception as e:
            print(f"Error loading paper metadata: {e}")
            return False
    
    def get_category_names(self, category_ids: List[int]) -> List[str]:
        """Convert category IDs to category names"""
        names = []
        for cat_id in category_ids:
            if cat_id in self.categories:
                names.append(self.categories[cat_id]['name'])
            else:
                pass  # If category ID not found, skip it. Private categories/lists were accidentally included in the original data.
                # Optionally, you could append a placeholder name:
                #names.append(f"Unknown_Category_{cat_id}")
        return names
    
    def get_category_hierarchy_path(self, category_id: int, max_depth: int = 10) -> List[str]:
        """Get the full hierarchy path for a category (from root to category)"""
        if category_id not in self.categories:
            # If the category ID is not found, return an empty list or a placeholder
            return []
            # Or return a placeholder name if you prefer:
            #return [f"Unknown_Category_{category_id}"]
        
        path = []
        current_id = category_id
        visited = set()  # Prevent infinite loops
        depth = 0
        
        while current_id and current_id not in visited and depth < max_depth:
            visited.add(current_id)
            if current_id in self.categories:
                category = self.categories[current_id]
                path.insert(0, category['name'])  # Insert at beginning for root-to-leaf path
                current_id = category['primary_parent']
                depth += 1
            else:
                break
        
        return path
    
    def combine_metadata(self, output_file: str, include_hierarchy: bool = False, 
                        include_parent_info: bool = False) -> bool:
        """Combine all metadata sources into a single CSV"""
        print(f"Combining metadata into: {output_file}")
        
        # Only process papers that exist in the OAI-PMH metadata
        papers_to_process = set(self.paper_metadata.keys())
        
        if not papers_to_process:
            print("No papers found in metadata to process")
            return False
        
        # Count how many papers have categories vs don't
        papers_with_categories = len(papers_to_process & set(self.paper_categories.keys()))
        papers_without_categories = len(papers_to_process) - papers_with_categories
        
        print(f"Processing {len(papers_to_process)} papers from metadata CSV:")
        print(f"  - {papers_with_categories} papers have category assignments")
        print(f"  - {papers_without_categories} papers have no category assignments")
        
        # Determine fieldnames from the original metadata plus category fields
        sample_metadata = next(iter(self.paper_metadata.values()))
        base_fields = list(sample_metadata.keys())
        
        # Add category fields
        category_fields = ['category_ids', 'category_names', 'num_categories']
        
        if include_hierarchy:
            category_fields.extend(['primary_category_hierarchy', 'all_category_hierarchies'])
        
        if include_parent_info:
            category_fields.extend(['primary_category_parents', 'primary_category_primary_parent'])
        
        fieldnames = base_fields + category_fields
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)

                writer.writeheader()
                
                processed = 0
                for paper_id in sorted(papers_to_process):
                    row = {}
                    
                    # Start with base metadata (we know this exists)
                    row.update(self.paper_metadata[paper_id])
                    
                    # Add category information
                    category_ids = self.paper_categories.get(paper_id, [])
                    category_names = self.get_category_names(category_ids)
                    
                    row['category_ids'] = ';'.join(map(str, category_ids)) if category_ids else ''
                    row['category_names'] = ';'.join(category_names) if category_names else ''
                    row['num_categories'] = len(category_ids)
                    
                    # Add hierarchy information if requested
                    if include_hierarchy and category_ids:
                        # Primary category hierarchy (first category)
                        primary_hierarchy = self.get_category_hierarchy_path(category_ids[0])
                        row['primary_category_hierarchy'] = ' > '.join(primary_hierarchy)
                        
                        # All category hierarchies
                        all_hierarchies = []
                        for cat_id in category_ids:
                            hierarchy = self.get_category_hierarchy_path(cat_id)
                            all_hierarchies.append(' > '.join(hierarchy))
                        row['all_category_hierarchies'] = ' | '.join(all_hierarchies)
                    elif include_hierarchy:
                        row['primary_category_hierarchy'] = ''
                        row['all_category_hierarchies'] = ''
                    
                    # Add parent information if requested (using pre-computed data)
                    if include_parent_info and category_ids:
                        primary_cat = self.categories.get(category_ids[0], {})
                        row['primary_category_parents'] = ';'.join(primary_cat.get('parent_names', []))
                        row['primary_category_primary_parent'] = primary_cat.get('primary_parent_name', '')
                    elif include_parent_info:
                        row['primary_category_parents'] = ''
                        row['primary_category_primary_parent'] = ''
                    
                    writer.writerow(row)
                    processed += 1
                    
                    if processed % 1000 == 0:
                        print(f"Processed {processed} papers...")
                
                print(f"Successfully combined metadata for {processed} papers")
                return True
                
        except Exception as e:
            print(f"Error writing combined metadata: {e}")
            return False
    
    def generate_summary_report(self, output_file: str) -> bool:
        """Generate a summary report of the combined data"""
        print(f"Generating summary report: {output_file}")
        
        try:
            # Calculate statistics - now only for papers in metadata
            papers_to_process = set(self.paper_metadata.keys())
            total_papers = len(papers_to_process)
            papers_with_metadata = len(self.paper_metadata)  # Same as total_papers now
            papers_with_categories = len(papers_to_process & set(self.paper_categories.keys()))
            papers_with_both = papers_with_categories  # Same as papers_with_categories now
            
            # Category usage statistics (only for papers we're processing)
            category_usage = {}
            for paper_id, categories in self.paper_categories.items():
                if paper_id in papers_to_process:  # Only count categories for papers we're processing
                    for cat_id in categories:
                        category_usage[cat_id] = category_usage.get(cat_id, 0) + 1
            
            # Papers per category distribution
            papers_per_category = list(category_usage.values())

            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("PhilPapers Metadata Combination Summary Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("OVERVIEW\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total papers processed: {total_papers}\n")
                f.write(f"Papers with metadata: {papers_with_metadata}\n")
                f.write(f"Papers with categories: {papers_with_categories}\n")
                f.write(f"Papers with both: {papers_with_both}\n")
                f.write(f"Total categories defined: {len(self.categories)}\n")
                f.write(f"Categories used by processed papers: {len(category_usage)}\n\n")
                
                # Additional info about excluded papers
                total_category_mappings = len(self.paper_categories)
                excluded_papers = total_category_mappings - len(set(self.paper_categories.keys()) & papers_to_process)
                if excluded_papers > 0:
                    f.write(f"Papers with categories but excluded (no metadata): {excluded_papers}\n\n")
                
                f.write("CATEGORY USAGE\n")
                f.write("-" * 20 + "\n")
                if papers_per_category:
                    most_used = max(category_usage.items(), key=lambda x: x[1])
                    most_used_name = self.categories.get(most_used[0], {}).get('name', f'Unknown_{most_used[0]}')
                    f.write(f"Most used category: {most_used[0]} - {most_used_name} ({most_used[1]} papers)\n")
                    f.write(f"Average papers per category: {sum(papers_per_category) / len(papers_per_category):.1f}\n")
                    f.write(f"Max papers in a category: {max(papers_per_category)}\n")
                    f.write(f"Min papers in a category: {min(papers_per_category)}\n")
                
                # Top 10 most used categories
                f.write("\nTOP 10 MOST USED CATEGORIES:\n")
                top_categories = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)[:10]
                for cat_id, count in top_categories:
                    cat_name = self.categories.get(cat_id, {}).get('name', f'Unknown_{cat_id}')
                    f.write(f"  {cat_id}: {cat_name} ({count} papers)\n")
            
            return True
            
        except Exception as e:
            print(f"Error generating summary report: {e}")
            return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine PhilPapers metadata from multiple sources",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('-c', '--categories-csv', 
                       default='/opt/extra/avijit/projects/moralkg/data/metadata/categories_parsed.csv',
                       help='Path to categories_parsed.csv file')
    
    parser.add_argument('-p', '--paper-categories-csv',
                       default='/opt/extra/avijit/projects/moralkg/data/metadata/paper_categories_parsed.csv', 
                       help='Path to paper_categories_parsed.csv file')
    
    parser.add_argument('-m', '--metadata-csv',
                        default='/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/2025-07-09-en.csv',
                       help='Path to the paper metadata CSV from OAI-PMH parsing')
    
    parser.add_argument('-o', '--output',
                          default='/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/combined_metadata.csv',
                       help='Output path for combined CSV file')
    
    parser.add_argument('--include-hierarchy', action='store_true',
                       help='Include category hierarchy paths in output (computed from parent relationships)')
    
    parser.add_argument('--include-parent-info', action='store_true',
                       help='Include pre-computed parent category information in output')
    
    parser.add_argument('--summary-report',
                          default='/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/summary_report.txt',
                       help='Generate a summary report file (optional)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files exist
    for file_path, name in [(args.categories_csv, 'categories CSV'), 
                           (args.paper_categories_csv, 'paper categories CSV'),
                           (args.metadata_csv, 'metadata CSV')]:
        if not Path(file_path).exists():
            print(f"Error: {name} file not found: {file_path}")
            sys.exit(1)
    
    # Initialize combiner and load data
    combiner = SimplifiedMetadataCombiner()
    
    success = True
    success &= combiner.load_categories_csv(args.categories_csv)
    success &= combiner.load_paper_categories_csv(args.paper_categories_csv)
    success &= combiner.load_paper_metadata_csv(args.metadata_csv)
    
    if not success:
        print("Failed to load required data files")
        sys.exit(1)
    
    # Combine metadata
    if not combiner.combine_metadata(args.output, args.include_hierarchy, args.include_parent_info):
        print("Failed to combine metadata")
        sys.exit(1)
    
    # Generate summary report if requested
    if args.summary_report:
        combiner.generate_summary_report(args.summary_report)
    
    print(f"\nMetadata combination completed successfully!")
    print(f"Output file: {args.output}")
    if args.summary_report:
        print(f"Summary report: {args.summary_report}")


if __name__ == '__main__':
    main()