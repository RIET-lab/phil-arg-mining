import json

# Load the data
with open('test_cleaned.json', 'r') as f:
    data = json.load(f)

rels = data[0]['relations']
print(f"Total relations: {len(rels)}")

# Simple analysis of relation distances based on character positions
print("\nAnalyzing relation distances...")

local_relations = 0  # Same sentence/close proximity
medium_relations = 0  # Different sentences, same paragraph area
distant_relations = 0  # Far apart, likely different paragraphs

for rel in rels:
    distance = abs(rel['head_start'] - rel['tail_start'])
    
    if distance < 200:  # Very close (likely same sentence/clause)
        local_relations += 1
    elif distance < 1000:  # Medium distance (nearby sentences)
        medium_relations += 1
    else:  # Large distance (likely different paragraphs)
        distant_relations += 1

print(f"Local relations (<200 chars): {local_relations} ({local_relations/len(rels)*100:.1f}%)")
print(f"Medium relations (200-1000 chars): {medium_relations} ({medium_relations/len(rels)*100:.1f}%)")
print(f"Distant relations (>1000 chars): {distant_relations} ({distant_relations/len(rels)*100:.1f}%)")

print("\nSample distant relations (potential cross-paragraph chains):")
count = 0
for rel in rels:
    distance = abs(rel['head_start'] - rel['tail_start'])
    if distance > 1000 and count < 5:
        print(f"\nDistance: {distance} characters")
        print(f"Head: \"{rel['head_text'][:100]}...\"")
        print(f"Tail: \"{rel['tail_text'][:100]}...\"")
        print(f"Relation: {rel['label']}")
        count += 1

# Look for potential reasoning chains
print("\n" + "="*50)
print("REASONING CHAIN ANALYSIS")
print("="*50)

# Build a simple graph to look for chains
from collections import defaultdict, deque

# Create adjacency list
graph = defaultdict(list)
adu_to_text = {}

for rel in rels:
    if rel['label'] == 'supports':  # Focus on support chains
        head_key = f"{rel['head_start']}-{rel['head_end']}"
        tail_key = f"{rel['tail_start']}-{rel['tail_end']}"
        graph[head_key].append(tail_key)
        adu_to_text[head_key] = rel['head_text'][:50] + "..."
        adu_to_text[tail_key] = rel['tail_text'][:50] + "..."

# Find chains of length 3 or more
def find_chains(start_node, current_path, all_paths, max_depth=5):
    if len(current_path) >= 3 and len(current_path) <= max_depth:
        all_paths.append(current_path[:])
    
    if len(current_path) < max_depth:
        for next_node in graph[start_node]:
            if next_node not in current_path:  # Avoid cycles
                current_path.append(next_node)
                find_chains(next_node, current_path, all_paths, max_depth)
                current_path.pop()

reasoning_chains = []
for start_node in list(graph.keys()):
    find_chains(start_node, [start_node], reasoning_chains)

print(f"Found {len(reasoning_chains)} potential reasoning chains of length 3+")

# Show the longest chains
if reasoning_chains:
    reasoning_chains.sort(key=len, reverse=True)
    print(f"\nTop 3 longest reasoning chains:")
    for i, chain in enumerate(reasoning_chains[:3]):
        print(f"\nChain {i+1} (length {len(chain)}):")
        for j, node in enumerate(chain):
            print(f"  {j+1}. {adu_to_text.get(node, 'Unknown')}")
            if j < len(chain) - 1:
                print("     ↓ supports")
else:
    print("No reasoning chains of length 3+ found!") 