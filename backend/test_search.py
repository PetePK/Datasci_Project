"""
Debug search to see actual distances returned by ChromaDB
"""
import sys
from pathlib import Path

# Add backend to path
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
print("\nDistances returned:")
for i, (paper_id, distance) in enumerate(zip(results['ids'], results['distances'])):
    relevance = 1.0 - distance
    print(f"  {i+1}. ID={paper_id}, distance={distance:.4f}, relevance={relevance:.4f}")

print("\n--- Testing threshold filter ---")
threshold = 0.3
passing = [d for d in results['distances'] if (1.0 - d) >= threshold]
print(f"Threshold: {threshold}")
print(f"Results passing threshold: {len(passing)} / {len(results['distances'])}")

# Test with lower threshold
print("\n--- Testing with threshold=0.0 ---")
threshold = 0.0
passing = [d for d in results['distances'] if (1.0 - d) >= threshold]
print(f"Results passing threshold: {len(passing)} / {len(results['distances'])}")
