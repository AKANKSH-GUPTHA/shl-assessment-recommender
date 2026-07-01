import json

with open("data/raw/catalog_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f, strict=False)

for item in data[:5]:
    print("=" * 50)
    print("NAME:", item.get("name"))
    print("TYPE:", item.get("test_type"))
    print("KEYS:", item.get("keys"))
    print("URL:", item.get("link"))