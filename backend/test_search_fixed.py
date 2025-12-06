"""
Test the fixed relevance calculation
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from services.vector_db import load_vector_db, search_papers

print("Loading vector DB...")
vector_db = load_vector_db()

print("\nSearching for 'cancer treatment'...")
results = search_papers(
    query="cancer treatment",
    n_results=10,
    vector_db=vector_db
)

print(f"\nFound {len(results['ids'])} results")
print("\nFixed relevance calculation (2.0 - distance) / 2.0:")
threshold = 0.3

passing_count = 0
for i, (paper_id, distance) in enumerate(zip(results['ids'], results['distances'])):
    # NEW FORMULA (fixed)
    relevance = (2.0 - distance) / 2.0
    passes = relevance >= threshold

    if passes:
        passing_count += 1

    status = "[PASS]" if passes else "[FAIL]"
    print(f"  {i+1}. {status} ID={paper_id}, distance={distance:.4f}, relevance={relevance:.4f}")

print(f"\n✓ Results passing threshold {threshold}: {passing_count} / {len(results['distances'])}")
