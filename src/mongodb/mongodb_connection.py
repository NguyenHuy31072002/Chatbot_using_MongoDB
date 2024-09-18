import pymongo

def get_mongo_client(mongo_uri):
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Kết nối thành công với MongoDB")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None

def ingest_data(collection, documents):
    collection.delete_many({})
    collection.insert_many(documents)
    print("Dữ liệu đã được lưu trữ vào MongoDB")
