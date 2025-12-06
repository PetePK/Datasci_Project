"""
Test if ChromaDB has papers loaded
"""
import chromadb
from pathlib import Path

# Path to vector DB
VECTOR_DB_PATH = Path(__file__).parent.parent / "data" / "vector_db"

print(f"Checking vector DB at: {VECTOR_DB_PATH}")
print(f"Path exists: {VECTOR_DB_PATH.exists()}")

if VECTOR_DB_PATH.exists():
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))

        # Try to get collection
        try:
            collection = client.get_collection(name="papers")
            count = collection.count()
            print(f"[OK] Collection 'papers' found")
            print(f"[OK] Number of papers in ChromaDB: {count}")

            if count > 0:
                # Test a simple query
                results = collection.query(
                    query_texts=["cancer treatment"],
                    n_results=5
                )
                print(f"[OK] Test query returned {len(results['ids'][0])} results")
            else:
                print("[ERROR] Collection is EMPTY!")

        except Exception as e:
            print(f"[ERROR] Collection 'papers' not found: {e}")
            print("\nAvailable collections:")
            collections = client.list_collections()
            for coll in collections:
                print(f"  - {coll.name} ({coll.count()} items)")

    except Exception as e:
        print(f"[ERROR] Failed to connect to ChromaDB: {e}")
else:
    print("[ERROR] Vector DB directory does not exist!")
