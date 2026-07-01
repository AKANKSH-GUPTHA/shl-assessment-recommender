import json

def load_catalog():
    with open("data/raw/catalog_raw.json", "r", encoding="utf-8") as f:
        data = json.load(f, strict=False)

    return data