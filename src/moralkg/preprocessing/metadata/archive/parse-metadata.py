# metadata-parse.py
"""
Parse OAI-PMH XML from PhilPapers into CSV table linking identifiers to metadata.

Usage:
    python metadata-parse.py [options]
Options:
    -n, --limit <N>            Limit to N records (default: no limit)
    -s, --start-date <DATE>    Start date (ISO format, inclusive)
    -e, --end-date <DATE>      End date (ISO format, inclusive)
    -l, --languages <LANGS>    Include only records with these dc:language codes (e.g. en fr)
    --exclude-deleted          Exclude records with header status='deleted'
    -i, --input-glob <GLOB>    Glob pattern for input XML files
    -m, --schema <PATH>        Path to metadata schema (YAML or JSON)
    -o, --output <FILE>        Output CSV filepath (default: Y-m-d.csv)
    -r, --range <START> <END>  Record range to parse (1-based, inclusive)

"""

import argparse, glob, re, json, yaml, csv
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

DEFAULT_SCHEMA = {
    "mappings": {
        "dc:title": "title",
        "dc:type": "type",
        "dc:creator": "authors",
        "dc:subject": "subjects",
        "dc:date": "year",
        "dc:identifier": "identifier-url",
        "dc:language": "language",
    },
    "special": {
        "identifier": {
            "xpath": "header/identifier",
            "pattern": r"oai:philpapers.org/rec/(?P<id>.+)$", # Note: this is a relatievely recent change; in May, 2025, the URL was oai:philarchive.org/rec/<id>
            "field": "identifier"
        }
    }
}

def load_schema(path):
    if not path:
        return DEFAULT_SCHEMA
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) if path.lower().endswith(('.yml','.yaml')) else json.load(f)

def parse_args():
    p = argparse.ArgumentParser(
        description="Parse PhilPapers OAI-PMH XML into CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('-n','--limit', type=int, default=None,
                   help="Max number of records to parse")
    p.add_argument('-s','--start-date', type=lambda s: datetime.fromisoformat(s),
                   help="ISO start datestamp (inclusive)")
    p.add_argument('-e','--end-date',   type=lambda s: datetime.fromisoformat(s),
                   help="ISO end datestamp (inclusive)")
    p.add_argument('-l','--languages', nargs='+', default=None,
                   help="dc:language codes to include")
    p.add_argument('--exclude-deleted', dest='include_deleted',
                   action='store_false', default=True,
                   help="Exclude records with header status='deleted'")
    p.add_argument('-i','--input-glob', default=None,
                   help="Glob for XML files (default: /opt/extra/avijit/projects/moralkg/data/metadata/phil-papers/*.xml or *.xml)")
    p.add_argument('-m','--schema', default=None,
                   help="Path to metadata-schema.yaml or .json")
    p.add_argument('-o','--output', default=None,
                   help="Output CSV filepath (default: Y-m-d.csv)")
    p.add_argument('-r','--range', nargs=2, type=int, metavar=('START','END'),
                   dest='record_range',
                   help="Inclusive record-range indices (1-based) to parse, e.g. -r 50 100")
    return p.parse_args()

def find_input_files(glob_pattern):
    base = Path('/opt/extra/avijit/projects/moralkg/data/metadata/phil-papers')
    pat = glob_pattern or (str(base/'*.xml') if base.exists() else '*.xml')
    files = sorted(Path(p) for p in glob.glob(pat))
    if not files:
        print(f"WARNING: no XML files found for '{pat}'")
    # else:
       # print(f"DEBUG: Found {len(files)} XML files for '{pat}': {files}")  
    return files

def extract_texts(elem, tag):
    return [e.text.strip() for e in elem.findall(f'.//{{*}}{tag}') if e.text]

def main():
    args = parse_args()
    schema = load_schema(args.schema)
    in_files = find_input_files(args.input_glob)

    # prepare output path
    today = datetime.now().strftime("%Y-%m-%d")
    out = args.output or f"{today}.csv"

    # build ordered fieldnames: identifier first, then all mapped fields in 
    # schema order
    mm = schema['mappings']
    fieldnames = ['identifier'] + list(mm.values())
    
    with open(out, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        processed = 0
        filtered_idx = 0

        for xml_file in in_files:
            # print(f"DEBUG: Processing file: {xml_file}")  
            try:
                tree = ET.parse(xml_file)
            except Exception as e:
                print(f"WARNING: failed to parse {xml_file}: {e}")
                continue
            root = tree.getroot()

            for rec in root.findall('.//record'):
                hdr = rec.find('header')
                raw_id = hdr.findtext('identifier', 'UNKNOWN_IDENTIFIER').strip()
                # print(f"DEBUG: Record {raw_id} - Checking status...")  
                
                if hdr.get('status')=='deleted' and not args.include_deleted:
                    print(f"DEBUG: Record {raw_id} - SKIPPED (deleted)")  
                    continue
                
                # datestamp filter
                dt = hdr.findtext('datestamp')
                # print(f"DEBUG: Record {raw_id} - Checking datestamp...")  
                if not dt:
                    print(f"WARNING: missing <datestamp> in record {raw_id} in {xml_file}")
                    continue
                dts = datetime.fromisoformat(dt.replace('Z',''))
                # print(f"DEBUG: Record {raw_id} - Datestamp: {dts}")  
                if args.start_date and dts<args.start_date:
                    # print(f"DEBUG: Record {raw_id} - SKIPPED (before start_date {args.start_date})")  
                    continue
                if args.end_date   and dts>args.end_date:
                    # print(f"DEBUG: Record {raw_id} - SKIPPED (after end_date {args.end_date})")
                    continue

                meta = rec.find('metadata')
                if meta is None:
                    print(f"WARNING: <metadata> missing in record {raw_id} in {xml_file}")
                    continue

                # language filter
                # print(f"DEBUG: Record {raw_id} - Checking language(s): {args.languages}")  
                if args.languages:
                    langs = extract_texts(meta,'language')
                    # print(f"DEBUG: Record {raw_id} - Found languages: {langs}")  
                    if not any(l in args.languages for l in langs):
                        # print(f"DEBUG: Record {raw_id} - SKIPPED (language not in {args.languages})")  
                        continue

                # record-range filtering
                filtered_idx += 1
                # print(f"DEBUG: Record {raw_id} - Passed initial filters. filtered_idx is now {filtered_idx}")  
                if args.record_range:
                    start,end = args.record_range
                    if filtered_idx < start:
                        # print(f"DEBUG: Record {raw_id} - SKIPPED (filtered_idx {filtered_idx} < start_range {start})")  
                        continue
                    if filtered_idx > end:
                        # print(f"DEBUG: Record {raw_id} - BREAK from file (filtered_idx {filtered_idx} > end_range {end})")  
                        break

                # limit check
                if args.limit is not None and processed>=args.limit:
                    # print(f"DEBUG: Global limit {args.limit} reached. Processed: {processed}. BREAK from file.")  
                    break 

                # extract fields
                row = {}
                
                # special identifier parsing
                raw_id = hdr.findtext('identifier','').strip()
                pat = schema.get('special',{}).get('identifier',{}).get('pattern','')
                m = re.match(pat, raw_id)
                row['identifier'] = m.group('id') if m else raw_id

                # regular mappings
                for xml_tag, col in mm.items():
                    pre,tag = xml_tag.split(':',1)
                    vals = extract_texts(meta, tag)
                    if vals:
                        row[col] = ';'.join(vals)

                writer.writerow(row)
                processed += 1
                # print(f"DEBUG: Record {raw_id} - PROCESSED. Total processed: {processed}")  

            # stop outer if limit reached
            if args.limit is not None and processed>=args.limit:
                # print(f"DEBUG: Global limit {args.limit} reached. Processed: {processed}. BREAK from all files.")  
                break

    print(f"Done: wrote {processed} records to {out}")

if __name__=='__main__':
    main()
    
