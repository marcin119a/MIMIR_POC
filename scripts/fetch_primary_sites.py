import json
from pathlib import Path
import requests
import pandas as pd
DATA_DIR = Path("../data")
SPLITS_PATH = DATA_DIR / "splits.json"
OUTPUT_PATH = DATA_DIR / "primary_sites.json"
GDC_URL = "https://api.gdc.cancer.gov/cases"
BATCH_SIZE = 2000  # safe batch size for GDC
def load_barcodes_from_splits(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    all_barcodes = []
    for key in ("train", "val", "test"):
        if key in splits and isinstance(splits[key], list):
            all_barcodes.extend(splits[key])
    # unique and sorted for readability
    return sorted(set(all_barcodes))
def fetch_primary_sites_for_barcodes(barcodes: list[str]) -> dict[str, str | None]:
    """
    Returns a dict: submitter_id (case_id, 12 chars) -> primary_site (or None if missing).
    """
    # case_id is the first 12 characters of the barcode
    case_ids = sorted(set(b[:12] for b in barcodes))
    result: dict[str, str | None] = {}
    # query in batches
    for i in range(0, len(case_ids), BATCH_SIZE):
        batch = case_ids[i : i + BATCH_SIZE]
        filters = {
            "op": "in",
            "content": {
                "field": "cases.submitter_id",
                "value": batch,
            },
        }
        params = {
            "filters": filters,
            "fields": "submitter_id,primary_site,project.project_id",
            "format": "JSON",
            "size": len(batch),
        }
        response = requests.post(GDC_URL, json=params)
        response.raise_for_status()
        data = response.json()["data"]["hits"]
        df = pd.DataFrame(data)
        # make sure we have required columns
        if "submitter_id" not in df or "primary_site" not in df:
            continue
        df_sites = df[["submitter_id", "primary_site"]]
        # fill result
        for submitter_id, primary_site in df_sites.itertuples(index=False):
            result[submitter_id] = primary_site
    # ensure every case_id has an entry (optional)
    for cid in case_ids:
        result.setdefault(cid, None)
    return result
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    barcodes = load_barcodes_from_splits(SPLITS_PATH)
    print(f"Number of unique barcodes in splits.json: {len(barcodes)}")
    primary_sites = fetch_primary_sites_for_barcodes(barcodes)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(primary_sites, f, ensure_ascii=False, indent=2)
    print(f"Saved primary_site for {len(primary_sites)} patients to: {OUTPUT_PATH}")
if __name__ == "__main__":
    main()