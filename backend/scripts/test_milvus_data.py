import sys
import os
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

def test_db():
    print("🔭 Testing Milvus Database Integrity...")
    
    # Path to DB
    db_path = "./milvus_hannom.db"
    if not os.path.exists(db_path):
        print(f"❌ Error: Database file not found at {db_path}")
        return

    # 1. Load Model
    print("🚀 Loading Model...")
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    
    # 2. Connect to Milvus Lite
    print("🔗 Connecting to Milvus...")
    client = MilvusClient(db_path)
    collection_name = "hannom_knowledge"
    
    if not client.has_collection(collection_name):
        print(f"❌ Error: Collection '{collection_name}' not found in DB.")
        return

    # 3. Check stats
    stats = client.get_collection_stats(collection_name)
    print(f"📊 Collection Stats: {stats}")
    
    # 4. Perform Search
    query = "Chữ 'nhất' nghĩa là gì?"
    print(f"🔍 Searching for: '{query}'")
    
    query_vector = model.encode(query).tolist()
    search_res = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        output_fields=["text"]
    )
    
    print("\n✅ Search Results:")
    for i, res in enumerate(search_res[0]):
        text = res['entity'].get('text', 'N/A')
        score = res['distance']
        print(f"[{i+1}] (Score: {score:.4f}): {text}")

if __name__ == "__main__":
    test_db()
