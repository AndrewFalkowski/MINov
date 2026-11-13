"""Download the public data used for forward validation into data/raw/ (git-ignored).

  * Materials Project releases 2023-11-01 and 2025-09-25 (summary, materials, provenance)
    from the AWS Open Data bucket s3://materialsproject-build (no API key needed)
  * GNoME stable materials (current summary + structures, and the 12/1/2023 summary)
    from the public bucket gs://gdm_materials_discovery

Files that already exist with the right size are skipped, so the script can be re-run.
Usage: uv run python experiments/fetch_data.py
"""

import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "data", "raw")
S3 = "https://materialsproject-build.s3.amazonaws.com"
GCS = "https://storage.googleapis.com/gdm_materials_discovery"

MP_RELEASES = ["2023-11-01", "2025-09-25"]
MP_COLLECTIONS = ["summary", "materials", "provenance"]
GNOME_FILES = [
    "gnome_data/stable_materials_summary.csv",
    "gnome_data/by_id.zip",
    "gnome_data/legacy_versions/12_1_23/stable_materials_summary.csv",
    "external_data/mp_snapshot_summary.csv",
]


def s3_listing(prefix):
    token, out = None, []
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        xml = urllib.request.urlopen(f"{S3}/?{urllib.parse.urlencode(query)}", timeout=60).read().decode()
        out += [(key, int(size)) for key, size in re.findall(r"<Key>(.*?)</Key>.*?<Size>(\d+)</Size>", xml, re.S)]
        match = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not match:
            return out
        token = match.group(1)


def fetch(url, dest, size=None):
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as response, open(tmp, "wb") as f:
        while chunk := response.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def main():
    jobs = []
    for release in MP_RELEASES:
        for collection in MP_COLLECTIONS:
            for key, size in s3_listing(f"collections/{release}/{collection}/"):
                jobs.append((f"{S3}/{urllib.parse.quote(key)}", os.path.join(RAW, "mp", key), size))
    for name in GNOME_FILES:
        jobs.append((f"{GCS}/{name}", os.path.join(RAW, "gnome", name), None))

    total = sum(size or 0 for _, _, size in jobs)
    print(f"{len(jobs)} files, {total / 1e9:.2f} GB from S3 listing (+ GNoME files)", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, n in enumerate(pool.map(lambda job: fetch(*job), jobs), 1):
            done += n
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} files, {done / 1e9:.2f} GB downloaded this run", flush=True)
    print("done")


if __name__ == "__main__":
    main()
