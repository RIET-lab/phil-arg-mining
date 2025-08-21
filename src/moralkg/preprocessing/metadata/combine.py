import csv
from pathlib import Path
from typing import Dict, List

from moralkg.logging import get_logger


class MetadataCombiner:
    """Combine OAI-PMH parsed paper metadata with PhilPapers categories."""

    def __init__(self) -> None:
        self.categories: Dict[int, Dict] = {}
        self.paper_categories: Dict[str, List[int]] = {}
        self.paper_metadata: Dict[str, Dict] = {}
        self._logger = get_logger(__name__)

    def load_categories_csv(self, categories_csv: str) -> bool:
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
                        'primary_parent_name': row['primary_parent_name'] if row['primary_parent_name'] else None,
                    }
            return True
        except Exception as e:
            self._logger.error("Error loading categories CSV %s: %s", categories_csv, e)
            return False

    def load_paper_categories_csv(self, paper_categories_csv: str) -> bool:
        try:
            with open(paper_categories_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    paper_id = row['paper_id']
                    category_ids = [int(x.strip()) for x in row['category_ids'].split(',') if x.strip()] if row['category_ids'] else []
                    self.paper_categories[paper_id] = category_ids
            return True
        except Exception as e:
            self._logger.error("Error loading paper categories CSV %s: %s", paper_categories_csv, e)
            return False

    def load_paper_metadata_csv(self, metadata_csv: str) -> bool:
        try:
            with open(metadata_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    paper_id = row.get('identifier', '').strip()
                    if paper_id:
                        self.paper_metadata[paper_id] = dict(row)
            return True
        except Exception as e:
            self._logger.error("Error loading paper metadata CSV %s: %s", metadata_csv, e)
            return False

    def get_category_names(self, category_ids: List[int]) -> List[str]:
        names: List[str] = []
        for cat_id in category_ids:
            if cat_id in self.categories:
                names.append(self.categories[cat_id]['name'])
        return names

    def _hierarchy_path(self, category_id: int, max_depth: int = 10) -> List[str]:
        if category_id not in self.categories:
            return []
        path: List[str] = []
        current_id = category_id
        visited = set()
        depth = 0
        while current_id and current_id not in visited and depth < max_depth:
            visited.add(current_id)
            category = self.categories.get(current_id)
            if not category:
                break
            path.insert(0, category['name'])
            current_id = category['primary_parent']
            depth += 1
        return path

    def combine(self, output_file: str, include_hierarchy: bool = False, include_parent_info: bool = False) -> bool:
        papers_to_process = set(self.paper_metadata.keys())
        if not papers_to_process:
            return False

        sample_metadata = next(iter(self.paper_metadata.values()))
        base_fields = list(sample_metadata.keys())
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

                for paper_id in sorted(papers_to_process):
                    row: Dict[str, object] = {}
                    row.update(self.paper_metadata[paper_id])

                    category_ids = self.paper_categories.get(paper_id, [])
                    category_names = self.get_category_names(category_ids)
                    row['category_ids'] = ';'.join(map(str, category_ids)) if category_ids else ''
                    row['category_names'] = ';'.join(category_names) if category_names else ''
                    row['num_categories'] = len(category_ids)

                    if include_hierarchy and category_ids:
                        primary_h = self._hierarchy_path(category_ids[0])
                        row['primary_category_hierarchy'] = ' > '.join(primary_h)
                        all_h = [' > '.join(self._hierarchy_path(cid)) for cid in category_ids]
                        row['all_category_hierarchies'] = ' | '.join(all_h)
                    elif include_hierarchy:
                        row['primary_category_hierarchy'] = ''
                        row['all_category_hierarchies'] = ''

                    if include_parent_info and category_ids:
                        primary_cat = self.categories.get(category_ids[0], {})
                        row['primary_category_parents'] = ';'.join(primary_cat.get('parent_names', []))
                        row['primary_category_primary_parent'] = primary_cat.get('primary_parent_name', '')
                    elif include_parent_info:
                        row['primary_category_parents'] = ''
                        row['primary_category_primary_parent'] = ''

                    writer.writerow(row)

            return True
        except Exception as e:
            self._logger.error("Error writing combined metadata to %s: %s", output_file, e)
            return False


def combine_metadata(
    categories_csv: str,
    paper_categories_csv: str,
    metadata_csv: str,
    output_csv: str,
    include_hierarchy: bool = False,
    include_parent_info: bool = False,
) -> bool:
    combiner = MetadataCombiner()
    ok = combiner.load_categories_csv(categories_csv)
    ok = ok and combiner.load_paper_categories_csv(paper_categories_csv)
    ok = ok and combiner.load_paper_metadata_csv(metadata_csv)
    if not ok:
        return False
    return combiner.combine(output_csv, include_hierarchy, include_parent_info)


def generate_summary_report(
    combiner: MetadataCombiner,
    output_file: str,
) -> bool:
    """Generate a summary report."""
    try:
        papers_to_process = set(combiner.paper_metadata.keys())
        total_papers = len(papers_to_process)
        papers_with_categories = len(papers_to_process & set(combiner.paper_categories.keys()))

        category_usage: Dict[int, int] = {}
        for paper_id, categories in combiner.paper_categories.items():
            if paper_id in papers_to_process:
                for cat_id in categories:
                    category_usage[cat_id] = category_usage.get(cat_id, 0) + 1

        papers_per_category = list(category_usage.values())

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("PhilPapers Metadata Combination Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total papers processed: {total_papers}\n")
            f.write(f"Papers with categories: {papers_with_categories}\n")
            f.write(f"Total categories defined: {len(combiner.categories)}\n")
            f.write(f"Categories used by processed papers: {len(category_usage)}\n\n")

            total_category_mappings = len(combiner.paper_categories)
            excluded_papers = total_category_mappings - len(set(combiner.paper_categories.keys()) & papers_to_process)
            if excluded_papers > 0:
                f.write(f"Papers with categories but excluded (no metadata): {excluded_papers}\n\n")

            f.write("CATEGORY USAGE\n")
            f.write("-" * 20 + "\n")
            if papers_per_category:
                most_used = max(category_usage.items(), key=lambda x: x[1])
                most_used_name = combiner.categories.get(most_used[0], {}).get('name', f'Unknown_{most_used[0]}')
                f.write(f"Most used category: {most_used[0]} - {most_used_name} ({most_used[1]} papers)\n")
                f.write(f"Average papers per category: {sum(papers_per_category) / len(papers_per_category):.1f}\n")
                f.write(f"Max papers in a category: {max(papers_per_category)}\n")
                f.write(f"Min papers in a category: {min(papers_per_category)}\n")

                f.write("\nTOP 10 MOST USED CATEGORIES:\n")
                top_categories = sorted(category_usage.items(), key=lambda x: x[1], reverse=True)[:10]
                for cat_id, count in top_categories:
                    cat_name = combiner.categories.get(cat_id, {}).get('name', f'Unknown_{cat_id}')
                    f.write(f"  {cat_id}: {cat_name} ({count} papers)\n")

        return True
    except Exception as e:
        get_logger(__name__).error("Error generating summary report %s: %s", output_file, e)
        return False
