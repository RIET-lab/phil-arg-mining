# Parse the metadata from the philpapers json API files


# The file containing the list of philpapers categories is here: /opt/extra/avijit/projects/moralkg/data/metadata/categories.json
# The file containing the list of paper IDs and their respective categories is here: /opt/extra/avijit/projects/moralkg/data/metadata/archive_categories.json

# The categories.json file has the following documentation:
#  Each entry in this list represents a category. The included fields are:
#    Name of the category
#    ID of the category
#    Comma-separated list of IDs of parents
#    IDs of the primary parent
# An example entry looks like this:
# ["Philosophy of Mind",16,"10",10]
# The file is a list of such entries, with no linebreaks. The entries are in a [] list, separated by commas.
# Some entries have commas in the name and/or multiple parent categories.





# The archive_categories.json file does not have public documentation, but the entries are structured as follows:
# {"categories":"5282,5312","id":"AABNCA"}
# or
# {"id":"AARACS-2","categories":"4,36"}
# id is the paper ID, and categories is a comma-separated list of category IDs.
# The file is a list of such entries, one or two per line. The entries are in a [] list, separated by commas.


"""
Parse metadata from PhilPapers JSON API files.

This script parses two files:
1. categories.json - Contains category information with names, IDs, and parent relationships
2. archive_categories.json - Contains paper IDs mapped to their category IDs
"""

import json
import csv
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class PhilPapersParser:
    def __init__(self):
        self.categories = {}
        self.paper_categories = {}
        
    def parse_categories(self, file_path: str) -> Dict[int, Dict]:
        """
        Parse the categories.json file.
        
        Each entry format: ["Category Name", id, "parent_ids", primary_parent_id]
        Example: ["Philosophy of Mind", 16, "10", 10]
        
        Returns:
            Dict mapping category ID to category info
        """
        print(f"Parsing categories from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            categories = {}
            
            for entry in data:
                if len(entry) >= 4:
                    name = entry[0]
                    cat_id = int(entry[1])
                    parent_ids_str = str(entry[2]) if entry[2] else ""
                    primary_parent = int(entry[3]) if entry[3] else None
                    
                    # Parse parent IDs (comma-separated string to list of ints)
                    parent_ids = []
                    if parent_ids_str:
                        try:
                            parent_ids = [int(pid.strip()) for pid in parent_ids_str.split(',') if pid.strip()]
                        except ValueError:
                            print(f"Warning: Could not parse parent IDs for category {cat_id}: {parent_ids_str}")
                    
                    categories[cat_id] = {
                        'name': name,
                        'id': cat_id,
                        'parent_ids': parent_ids,
                        'primary_parent': primary_parent
                    }
                else:
                    print(f"Warning: Malformed entry with {len(entry)} fields: {entry}")
            
            print(f"Parsed {len(categories)} categories")
            self.categories = categories
            return categories
            
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
    
    def parse_archive_categories(self, file_path: str) -> Dict[str, List[int]]:
        """
        Parse the archive_categories.json file.
        
        Each entry format: {"categories": "5282,5312", "id": "AABNCA"}
        or: {"id": "AARACS-2", "categories": "4,36"}
        
        Returns:
            Dict mapping paper ID to list of category IDs
        """
        print(f"Parsing archive categories from: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paper_categories = {}
            
            for entry in data:
                if isinstance(entry, dict) and 'id' in entry and 'categories' in entry:
                    paper_id = entry['id']
                    categories_str = str(entry['categories'])
                    
                    # Parse category IDs (comma-separated string to list of ints)
                    category_ids = []
                    if categories_str:
                        try:
                            category_ids = [int(cid.strip()) for cid in categories_str.split(',') if cid.strip()]
                        except ValueError:
                            print(f"Warning: Could not parse category IDs for paper {paper_id}: {categories_str}")
                    
                    paper_categories[paper_id] = category_ids
                else:
                    print(f"Warning: Malformed entry: {entry}")
            
            print(f"Parsed {len(paper_categories)} paper-category mappings")
            self.paper_categories = paper_categories
            return paper_categories
            
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}
    
    def get_category_hierarchy(self, category_id: int) -> List[Dict]:
        """
        Get the full hierarchy path for a category (from root to category).
        
        Returns:
            List of category dictionaries from root to the specified category
        """
        if category_id not in self.categories:
            return []
        
        hierarchy = []
        current_id = category_id
        visited = set()  # Prevent infinite loops in case of circular references
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            if current_id in self.categories:
                category = self.categories[current_id]
                hierarchy.insert(0, category)  # Insert at beginning to build path from root
                current_id = category['primary_parent']
            else:
                break
        
        return hierarchy
    
    def get_category_name(self, category_id: int) -> str:
        """
        Get the name of a category by its ID.
        
        Returns:
            Category name or empty string if not found
        """
        if category_id in self.categories:
            return self.categories[category_id]['name']
        return ""
    
    def export_categories_csv(self, output_path: str):
        """Export categories to CSV format with parent names."""
        print(f"Exporting categories to: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['category_id', 'name', 'parent_ids', 'primary_parent', 'parent_names', 'primary_parent_name'])
            
            for cat_id, category in sorted(self.categories.items()):
                parent_ids_str = ';'.join(map(str, category['parent_ids'])) if category['parent_ids'] else ''
                
                # Get parent names
                parent_names = []
                if category['parent_ids']:
                    for parent_id in category['parent_ids']:
                        parent_name = self.get_category_name(parent_id)
                        if parent_name:
                            parent_names.append(parent_name)
                        else:
                            parent_names.append(f"[Unknown ID: {parent_id}]")
                parent_names_str = ';'.join(parent_names)
                
                # Get primary parent name
                primary_parent_name = ""
                if category['primary_parent']:
                    primary_parent_name = self.get_category_name(category['primary_parent'])
                    if not primary_parent_name:
                        primary_parent_name = f"[Unknown ID: {category['primary_parent']}]"
                
                writer.writerow([
                    cat_id,
                    category['name'],
                    parent_ids_str,
                    category['primary_parent'] or '',
                    parent_names_str,
                    primary_parent_name
                ])
    
    def export_paper_categories_csv(self, output_path: str):
        """Export paper-category mappings to CSV format."""
        print(f"Exporting paper categories to: {output_path}")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['paper_id', 'category_ids'])
            
            for paper_id, category_ids in sorted(self.paper_categories.items()):
                category_ids_str = ','.join(map(str, category_ids)) if category_ids else ''
                writer.writerow([paper_id, category_ids_str])
    
    def get_papers_in_category(self, category_id: int) -> List[str]:
        """Get all paper IDs that belong to a specific category."""
        papers = []
        for paper_id, categories in self.paper_categories.items():
            if category_id in categories:
                papers.append(paper_id)
        return papers
    
    def get_category_statistics(self) -> Dict:
        """Generate statistics about the parsed data."""
        stats = {
            'total_categories': len(self.categories),
            'total_papers': len(self.paper_categories),
            'categories_with_parents': len([c for c in self.categories.values() if c['parent_ids']]),
            'root_categories': len([c for c in self.categories.values() if not c['parent_ids']]),
        }
        
        # Category usage statistics
        category_usage = {}
        for paper_id, categories in self.paper_categories.items():
            for cat_id in categories:
                category_usage[cat_id] = category_usage.get(cat_id, 0) + 1
        
        if category_usage:
            stats['most_used_category'] = max(category_usage.items(), key=lambda x: x[1])
            stats['categories_with_papers'] = len(category_usage)
            stats['avg_papers_per_category'] = sum(category_usage.values()) / len(category_usage)
        
        return stats


def main():
    """Main function to demonstrate usage."""
    parser = PhilPapersParser()
    
    # Define file paths
    categories_file = "/opt/extra/avijit/projects/moralkg/data/metadata/categories_2025_07_09.json"
    archive_file = "/opt/extra/avijit/projects/moralkg/data/metadata/archive_categories_2025_07_09.json"
    
    # Parse the files
    categories = parser.parse_categories(categories_file)
    paper_categories = parser.parse_archive_categories(archive_file)
    
    if not categories and not paper_categories:
        print("No data parsed. Please check file paths and formats.")
        return
    
    # Display statistics
    print("\n" + "="*50)
    print("PARSING STATISTICS")
    print("="*50)
    stats = parser.get_category_statistics()
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Example: Show some categories
    if categories:
        print(f"\nFirst 5 categories:")
        for i, (cat_id, category) in enumerate(sorted(categories.items())[:5]):
            print(f"  {cat_id}: {category['name']} (Parents: {category['parent_ids']})")
    
    # Example: Show some paper mappings
    if paper_categories:
        print(f"\nFirst 5 paper-category mappings:")
        for i, (paper_id, cats) in enumerate(sorted(paper_categories.items())[:5]):
            print(f"  {paper_id}: {cats}")
    
    # Export to CSV files (optional)
    try:
        export_folder = "/opt/extra/avijit/projects/moralkg/data/metadata/"
        if categories:
            parser.export_categories_csv(f"{export_folder}categories_parsed.csv")
        if paper_categories:
            parser.export_paper_categories_csv(f"{export_folder}paper_categories_parsed.csv")
        print("\nCSV files exported successfully!")
    except Exception as e:
        print(f"Error exporting CSV files: {e}")


if __name__ == "__main__":
    main()