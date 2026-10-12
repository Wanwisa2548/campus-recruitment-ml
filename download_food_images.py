"""Download candidates from a user-configured CSV of direct public image URLs."""
import argparse
import csv
import hashlib
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from dataset_prepare import CLASSES, ROOT, read_rgb

FIELDS = ["filename", "class", "source_url", "source_provider", "license", "notes"]
SEARCH_TERMS = {
    "larb": ["Thai larb", "larb moo", "Thai minced pork salad", "ลาบหมู"],
    "noodle": ["Thai noodle soup", "kuay teow", "Thai noodles", "ก๋วยเตี๋ยว"],
    "pad_kra_pao": ["pad kra pao", "pad kaprao", "Thai basil pork rice", "ผัดกะเพรา"],
    "som_tam": ["som tam", "Thai papaya salad", "green papaya salad Thailand", "ส้มตำ"],
    "tom_yum": ["tom yum", "tom yum goong", "Thai spicy shrimp soup", "ต้มยำกุ้ง"],
}
MAX_BYTES = 20 * 1024 * 1024


def download(input_csv, root=ROOT, target=125, delay=0.5):
    raw = Path(root) / "thai_food_raw"
    tracking = raw / "sources.csv"
    with Path(input_csv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"class", "source_url"}.issubset(reader.fieldnames or []):
            raise ValueError("URL CSV needs class and source_url columns")
        candidates = list(reader)
    raw.mkdir(parents=True, exist_ok=True)
    known_urls = set()
    if tracking.exists():
        with tracking.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing sources.csv has unexpected columns; preserving it")
            known_urls.update(row["source_url"] for row in reader if row["source_url"])
    counts = {}
    for name in CLASSES:
        folder = raw / name
        folder.mkdir(parents=True, exist_ok=True)
        counts[name] = sum(p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                           for p in folder.iterdir())
    added = 0
    with tracking.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if tracking.stat().st_size == 0:
            writer.writeheader()
        for index, row in enumerate(candidates, 1):
            name, url = row.get("class", "").strip(), row.get("source_url", "").strip()
            url = urllib.parse.urldefrag(url)[0]
            if name not in CLASSES:
                print(f"[{index}/{len(candidates)}] Skipped unknown class: {name}")
                continue
            if counts[name] >= target or url in known_urls:
                continue
            if urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
                print(f"Skipped non-HTTP(S) URL: {url}")
                continue
            known_urls.add(url)
            destination = None
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ThaiFoodDatasetPreparation/1.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = response.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise ValueError("Image exceeds 20 MB limit")
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}.get(image.format)
                if extension is None:
                    raise ValueError("Unsupported image format")
                digest = hashlib.sha256(data).hexdigest()
                destination = raw / name / ("download_" + digest + extension)
                # Exclusive creation prevents overwriting even an untracked image.
                try:
                    with destination.open("xb") as image_file:
                        image_file.write(data)
                except FileExistsError:
                    print(f"Skipped existing content: {destination.name}")
                    destination = None
                    continue
                read_rgb(destination)
                writer.writerow({"filename": f"{name}/{destination.name}", "class": name,
                    "source_url": url, "source_provider": row.get("source_provider", "") or "user_url_csv",
                    "license": row.get("license", ""), "notes": row.get("notes", "")})
                stream.flush()
                counts[name] += 1
                added += 1
                print(f"[{index}/{len(candidates)}] {name}: {counts[name]}/{target} candidates")
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                if destination is not None:
                    destination.unlink(missing_ok=True)
                print(f"Download failed ({name}): {exc}")
            finally:
                time.sleep(delay)
    print(f"Downloaded {added} new candidates. Check labels and licenses, then run dataset_prepare.py.")
    print("Images are not automatically copyright-free. Check source terms before redistribution.")
    for name, count in counts.items():
        if count < target:
            print(f"{name}: {count}/{target}; supply more permitted source URLs if needed.")
    return added


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urls", type=Path, help="CSV exported from permitted sources/APIs; see food_image_urls.example.csv")
    parser.add_argument("--target", type=int, default=125, help="Candidate target per class (default 125)")
    parser.add_argument("--search-terms", action="store_true", help="Print predefined English and Thai searches")
    args = parser.parse_args()
    if args.search_terms:
        for name, terms in SEARCH_TERMS.items():
            print(f"{name}: " + " | ".join(terms))
        return 0
    if not args.urls:
        print("No source provider configured. Supply --urls your_urls.csv with permitted direct image URLs.")
        print("No API key needed for CSV mode. See README and food_image_urls.example.csv.")
        return 0
    if args.target < 1:
        parser.error("--target must be positive")
    try:
        download(args.urls, target=args.target)
    except (OSError, ValueError) as exc:
        print(f"Downloader stopped: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
