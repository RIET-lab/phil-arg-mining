"""Parse PhilPapers category metadata and paper-category assignments.

This module mirrors and modernizes the legacy categories_api_parser script,
providing a reusable class-based API and CSV export helpers.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


class PhilPapersParser:
    def __init__(self) -> None:
        self.categories: Dict[int, Dict] = {}
        self.paper_categories: Dict[str, List[int]] = {}

    def parse_categories(self, file_path: str) -> Dict[int, Dict]:
        """Parse the categories.json file.

        Each entry format: ["Category Name", id, "parent_ids", primary_parent_id]
        Example: ["Philosophy of Mind", 16, "10", 10]
        """
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        categories: Dict[int, Dict] = {}
        for entry in data:
            if len(entry) < 4:
                continue
            name = entry[0]
            cat_id = int(entry[1])
            parent_ids_str = str(entry[2]) if entry[2] else ""
            primary_parent = int(entry[3]) if entry[3] else None

            parent_ids: List[int] = []
            if parent_ids_str:
                try:
                    parent_ids = [int(pid.strip()) for pid in parent_ids_str.split(',') if pid.strip()]
                except ValueError:
                    # Skip malformed parent IDs
                    parent_ids = []

            categories[cat_id] = {
                'name': name,
                'id': cat_id,
                'parent_ids': parent_ids,
                'primary_parent': primary_parent,
            }

        self.categories = categories
        return categories

    def parse_archive_categories(self, file_path: str) -> Dict[str, List[int]]:
        """Parse the archive_categories.json file mapping paper IDs to category IDs."""
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        paper_categories: Dict[str, List[int]] = {}
        for entry in data:
            if not isinstance(entry, dict) or 'id' not in entry or 'categories' not in entry:
                continue
            paper_id = str(entry['id'])
            categories_str = str(entry['categories'])
            category_ids: List[int] = []
            if categories_str:
                try:
                    category_ids = [int(cid.strip()) for cid in categories_str.split(',') if cid.strip()]
                except ValueError:
                    category_ids = []
            paper_categories[paper_id] = category_ids

        self.paper_categories = paper_categories
        return paper_categories

    def get_category_name(self, category_id: int) -> str:
        if category_id in self.categories:
            return self.categories[category_id]['name']
        return ""

    def get_category_hierarchy(self, category_id: int) -> List[Dict]:
        if category_id not in self.categories:
            return []

        hierarchy: List[Dict] = []
        current_id: Optional[int] = category_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            if current_id in self.categories:
                category = self.categories[current_id]
                hierarchy.insert(0, category)
                current_id = category['primary_parent']
            else:
                break
        return hierarchy

    def export_categories_csv(self, output_path: str) -> None:
        """Export categories to CSV including parent names."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(['category_id', 'name', 'parent_ids', 'primary_parent', 'parent_names', 'primary_parent_name'])

            for cat_id, category in sorted(self.categories.items()):
                parent_ids_str = ';'.join(map(str, category['parent_ids'])) if category['parent_ids'] else ''

                parent_names: List[str] = []
                if category['parent_ids']:
                    for parent_id in category['parent_ids']:
                        parent_name = self.get_category_name(parent_id)
                        parent_names.append(parent_name if parent_name else f"[Unknown ID: {parent_id}]")
                parent_names_str = ';'.join(parent_names)

                primary_parent_name = ""
                if category['primary_parent']:
                    primary_parent_name = self.get_category_name(category['primary_parent']) or f"[Unknown ID: {category['primary_parent']}]"

                writer.writerow([
                    cat_id,
                    category['name'],
                    parent_ids_str,
                    category['primary_parent'] or '',
                    parent_names_str,
                    primary_parent_name,
                ])

    def export_paper_categories_csv(self, output_path: str) -> None:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['paper_id', 'category_ids'])
            for paper_id, category_ids in sorted(self.paper_categories.items()):
                category_ids_str = ','.join(map(str, category_ids)) if category_ids else ''
                writer.writerow([paper_id, category_ids_str])

    def get_papers_in_category(self, category_id: int) -> List[str]:
        return [paper_id for paper_id, cats in self.paper_categories.items() if category_id in cats]

    def get_category_statistics(self) -> Dict:
        stats = {
            'total_categories': len(self.categories),
            'total_papers': len(self.paper_categories),
            'categories_with_parents': len([c for c in self.categories.values() if c['parent_ids']]),
            'root_categories': len([c for c in self.categories.values() if not c['parent_ids']]),
        }

        category_usage: Dict[int, int] = {}
        for _, categories in self.paper_categories.items():
            for cat_id in categories:
                category_usage[cat_id] = category_usage.get(cat_id, 0) + 1

        if category_usage:
            stats['most_used_category'] = max(category_usage.items(), key=lambda x: x[1])
            stats['categories_with_papers'] = len(category_usage)
            stats['avg_papers_per_category'] = sum(category_usage.values()) / len(category_usage)

        return stats


