"""Extract /knowledge.md from the agent's LangGraph state and write it to disk."""

import sys
from pymongo import MongoClient
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

MONGODB_URI = "mongodb://firmware_user:f2970a49-95bc-412c-8983-8c69d0a01e01@mongodb.noreal.solutions:27017/?directConnection=true&serverSelectionTimeoutMS=2000&authSource=admin"
MONGODB_DB = "firmware_analysis"
THREAD_ID = "re-session"
FILE_PATH = "/knowledge.md"
OUTPUT = "knowledge.md"

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
serde = JsonPlusSerializer()

doc = db["checkpoint_writes"].find_one(
    {"channel": "files", "thread_id": THREAD_ID},
    sort=[("_id", -1)],
)

if doc is None:
    print(f"No 'files' channel writes found for thread '{THREAD_ID}'", file=sys.stderr)
    sys.exit(1)

data = serde.loads_typed((doc["type"], doc["value"]))

if FILE_PATH not in data:
    print(f"'{FILE_PATH}' not found. Files present:", file=sys.stderr)
    for p in sorted(data):
        print(f"  {p}", file=sys.stderr)
    sys.exit(1)

file_data = data[FILE_PATH]
content = file_data.get("content", "")
if isinstance(content, list):
    content = "\n".join(content)

with open(OUTPUT, "w") as f:
    f.write(content)

print(f"Written {len(content)} chars to {OUTPUT}")
