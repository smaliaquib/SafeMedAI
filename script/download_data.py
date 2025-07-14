import requests
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

DOWNLOAD_INDEX_URL = os.environ["DOWNLOAD_INDEX_URL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch metadata
resp = requests.get(DOWNLOAD_INDEX_URL)
resp.raise_for_status()
data = resp.json()

# Navigate into data.results.drug.event.partitions
partitions = (
    data.get("results", {}).get("drug", {}).get("event", {}).get("partitions", [])
)

# Filter for 2025 files
links_2025 = [
    part["file"] for part in partitions if "2025" in part.get("display_name", "")
]

print(f"Found {len(links_2025)} 2025 drug-event files.")


def download_file(url):
    local = os.path.join(OUTPUT_DIR, os.path.basename(url))
    if os.path.exists(local):
        print(f"-> Already exists: {local}")
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(local, "wb") as f, tqdm(
            desc=os.path.basename(local),
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


# Download
for url in links_2025:
    download_file(url)

print("Completed downloading 2025 drug-event files.")
