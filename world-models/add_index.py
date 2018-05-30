from pymongo import MongoClient
import numpy as np

client = MongoClient()
db = client.retro_contest
collection = db["1527608256"]

total_doc = collection.find()
length = total_doc.count()
new_idx = np.arange(length)
np.random.shuffle(new_idx)

i = 0
for item in total_doc:
    item["id"] = int(new_idx[i])
    collection.update({'_id': item["_id"]}, {"$set": item})
    i += 1