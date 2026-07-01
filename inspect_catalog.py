import json

with open("data/raw/catalog_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f, strict=False)

print("Total assessments:", len(data))

print("\nFirst assessment:")
print(data[0]["name"])

print("\nFields:")
for field in data[0].keys():
    print(field)