"""Validate raw Thai food photos, deduplicate globally, and make seeded splits."""
import argparse
import hashlib
import json
import random
import tempfile
import warnings
from pathlib import Path

from PIL import Image, ImageOps

CLASSES = ("larb", "noodle", "pad_kra_pao", "som_tam", "tom_yum")
SPLITS = ("train", "val", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
ROOT = Path(__file__).resolve().parent
MANIFEST = ".dataset_prepare_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path):
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            rgb.load()
            return rgb


def fingerprints(image):
    """Pixel SHA catches format changes; two dHashes catch simple resize/re-encode."""
    pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    gray = image.convert("L")
    hashes = []
    for vertical in (False, True):
        small = gray.resize((8, 9) if vertical else (9, 8), Image.Resampling.LANCZOS)
        values = list(small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                index = y * small.width + x
                other = index + (small.width if vertical else 1)
                bits = (bits << 1) | (values[index] > values[other])
        hashes.append(bits)
    return pixels, tuple(hashes)


def is_near(first, second):
    # Low-detail images give uninformative hashes: use pixel SHA for those.
    if any(h in (0, (1 << 64) - 1) for h in first + second):
        return False
    return all((a ^ b).bit_count() <= 2 for a, b in zip(first, second))


def split_counts(count):
    # Largest remainder allocation, with stable train/val/test tie breaking.
    quotas = [count * fraction for fraction in (0.70, 0.15, 0.15)]
    result = [int(q) for q in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - result[i]), i))
    for i in order[:count - sum(result)]:
        result[i] += 1
    # Small datasets of >=3 still get one image in every split.
    if count >= 3:
        for i in (1, 2):
            if result[i] == 0:
                result[0] -= 1
                result[i] = 1
    return result


def safe_path(root, relative):
    path = root / relative
    parts = Path(relative).parts
    if (len(parts) != 3 or parts[0] not in SPLITS or parts[1] not in CLASSES
            or not parts[2].startswith(parts[1] + "_") or path.suffix != ".jpg"
            or path.resolve().parent != (root / parts[0] / parts[1]).resolve()):
        raise ValueError(f"Unsafe manifest path: {relative}")
    return path


def check_output(output):
    """Never delete manually supplied images or edited generated files."""
    for path in [output, *(output / s for s in SPLITS),
                 *(output / s / c for s in SPLITS for c in CLASSES)]:
        if (path.is_symlink() or getattr(path, "is_junction", lambda: False)()
                or path.resolve() != path.absolute()):
            raise ValueError(f"Refusing linked output directory: {path}")
    manifest_path = output / MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("Refusing linked manifest")
    previous = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1 or not isinstance(manifest.get("files"), dict):
            raise ValueError("Unrecognized generated-file manifest")
        previous = manifest["files"]
    for relative, digest in previous.items():
        path = safe_path(output, relative)
        if path.is_symlink():
            raise ValueError(f"Refusing linked file: {path}")
        if path.exists() and sha256(path) != digest:
            raise ValueError(f"Generated file was modified; preserving it: {path}")
    for split in SPLITS:
        for path in (output / split).rglob("*"):
            if path.is_file() and path.suffix.lower() in EXTENSIONS | {".gif"}:
                if path.relative_to(output).as_posix() not in previous:
                    raise ValueError(f"Unmanaged image preserved: {path}. Move it outside thai_food before preparing.")
    return previous


def prepare(root=ROOT):
    root = Path(root).resolve()
    raw, output = root / "thai_food_raw", root / "thai_food"
    previous = check_output(output)
    print(f"Source: {raw}\nOutput: {output}")
    print("Regenerating only manifest-tracked JPGs in train/val/test; raw and other files are preserved.")
    rows, accepted = {}, {name: [] for name in CLASSES}
    file_hashes, pixel_hashes, perceptual = set(), set(), []
    # Stage every conversion before touching an existing generated dataset.
    with tempfile.TemporaryDirectory(prefix=".dataset-stage-", dir=root) as staging:
        stage = Path(staging)
        for name in CLASSES:
            folder = raw / name
            folder.mkdir(parents=True, exist_ok=True)
            paths = sorted((p for p in folder.iterdir() if p.is_file()
                            and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)
            row = rows[name] = dict(raw=len(paths), valid=0, corrupted=0, duplicate=0)
            for path in paths:
                try:
                    image = read_rgb(path)
                    digest = sha256(path)
                    pixels, dhashes = fingerprints(image)
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                        Image.DecompressionBombWarning) as exc:
                    row["corrupted"] += 1
                    print(f"Skipped unreadable {path.name}: {exc}")
                    continue
                duplicate = (digest in file_hashes or pixels in pixel_hashes
                             or any(is_near(dhashes, h) for h in perceptual))
                # Include rejected hashes too, so A~B~C chains cannot leak.
                file_hashes.add(digest)
                pixel_hashes.add(pixels)
                perceptual.append(dhashes)
                if duplicate:
                    row["duplicate"] += 1
                    print(f"Skipped duplicate: {name}/{path.name} (check labels if across classes)")
                    continue
                target = stage / f"{name}_{len(accepted[name]) + 1:04d}.jpg"
                image.save(target, "JPEG", quality=95)
                accepted[name].append(target)
                row["valid"] += 1
            if row["valid"] < 50:
                print(f"WARNING: {name} has {row['valid']} unique valid images; aim for 100-150 (minimum 50).")
        if not any(accepted.values()) and previous:
            print("No usable raw images; existing generated dataset preserved.")
            print_report(rows)
            return rows
        planned, generated_hashes, generated_pixels, generated_dhashes = {}, set(), set(), []
        for name in CLASSES:
            files = accepted[name][:]
            random.Random(SEED).shuffle(files)
            counts = split_counts(len(files))
            offset = 0
            for split, count in zip(SPLITS, counts):
                rows[name][split] = count
                for source in files[offset:offset + count]:
                    relative = f"{split}/{name}/{source.name}"
                    destination = safe_path(output, relative)
                    if destination.exists() and relative not in previous:
                        raise ValueError(f"Filename collision; preserving {destination}")
                    digest = sha256(source)
                    pixels, dhashes = fingerprints(read_rgb(source))
                    if (digest in generated_hashes or pixels in generated_pixels
                            or any(is_near(dhashes, h) for h in generated_dhashes)):
                        raise ValueError("JPEG conversion produced duplicates; previous dataset preserved. Review raw photos.")
                    generated_hashes.add(digest)
                    generated_pixels.add(pixels)
                    generated_dhashes.append(dhashes)
                    planned[relative] = (source, digest)
                offset += count
        # Recheck before cleanup, in case files changed during validation.
        if check_output(output) != previous:
            raise ValueError("Output manifest changed during preparation")
        for relative in previous:
            safe_path(output, relative).unlink(missing_ok=True)
        for split in SPLITS:
            for name in CLASSES:
                (output / split / name).mkdir(parents=True, exist_ok=True)
        for relative, (source, _) in planned.items():
            source.replace(safe_path(output, relative))
        manifest_temp = output / (MANIFEST + ".tmp")
        manifest_temp.write_text(json.dumps({"version": 1, "seed": SEED,
            "files": {p: value[1] for p, value in planned.items()}}, indent=2) + "\n", encoding="utf-8")
        manifest_temp.replace(output / MANIFEST)
    print_report(rows)
    return rows


def print_report(rows):
    columns = ("raw", "valid", "corrupted", "duplicate", "train", "val", "test")
    print("\n" + f"{'Class':<15}" + "".join(f"{c.title():>11}" for c in columns))
    print("-" * 92)
    for name, row in [*rows.items(), ("TOTAL", {c: sum(r.get(c, 0) for r in rows.values()) for c in columns})]:
        print(f"{name:<15}" + "".join(f"{row.get(c, 0):>11}" for c in columns))
    print("Valid = unique usable photos after duplicate removal. Raw = supported files.")
    print("Review photos for wrong labels and near-duplicates missed by hashing; never train on test images.")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        prepare()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Preparation stopped safely: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
