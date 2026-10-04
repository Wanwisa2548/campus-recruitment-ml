"""Synthetic fixtures verify preparation safety; these are not training images."""
import contextlib
import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import dataset_prepare as prep
import download_food_images as downloader


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in prep.CLASSES:
            (self.root / "thai_food_raw" / name).mkdir(parents=True)

    def photo(self, name, number, mode="RGB", suffix=".png"):
        rng = random.Random(number)
        image = Image.frombytes("RGB", (32, 32), rng.randbytes(32 * 32 * 3)).convert(mode)
        path = self.root / "thai_food_raw" / name / f"{number:03d}{suffix}"
        image.save(path)
        return path

    def run_prepare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return prep.prepare(self.root)

    def manifest(self):
        return json.loads((self.root / "thai_food" / prep.MANIFEST).read_text())["files"]

    def test_empty_and_rounding(self):
        rows = self.run_prepare()
        self.assertEqual(sum(r["raw"] for r in rows.values()), 0)
        for split in prep.SPLITS:
            for name in prep.CLASSES:
                self.assertTrue((self.root / "thai_food" / split / name).is_dir())
        self.assertEqual(prep.split_counts(100), [70, 15, 15])
        for count in range(151):
            result = prep.split_counts(count)
            self.assertEqual(sum(result), count)
            if count >= 3:
                self.assertTrue(all(result))

    def test_duplicates_corruption_rgb_and_reproducibility(self):
        for i in range(10):
            self.photo("larb", i)
        first = self.root / "thai_food_raw" / "larb" / "000.png"
        (first.parent / "exact.jpg").write_bytes(first.read_bytes())
        with Image.open(first) as image:
            image.resize((64, 64)).save(first.parent / "near.png")
            image.save(self.root / "thai_food_raw" / "noodle" / "same.bmp")
        (first.parent / "broken.webp").write_bytes(b"not an image")
        self.photo("som_tam", 100, "RGBA")
        self.photo("som_tam", 101, "L", ".bmp")
        self.photo("som_tam", 102, "RGB", ".webp")
        raw_hash = prep.sha256(first)
        rows = self.run_prepare()
        self.assertEqual(rows["larb"]["corrupted"], 1)
        self.assertEqual(rows["larb"]["duplicate"], 2)
        self.assertEqual(rows["noodle"]["duplicate"], 1)
        self.assertEqual(rows["larb"]["valid"], 10)
        self.assertEqual([rows["som_tam"][s] for s in prep.SPLITS], [1, 1, 1])
        manifest = self.manifest()
        self.assertEqual(len(set(manifest.values())), len(manifest))
        for path in manifest:
            with Image.open(self.root / "thai_food" / path) as image:
                self.assertEqual(image.mode, "RGB")
        self.run_prepare()
        self.assertEqual(self.manifest(), manifest)
        self.assertEqual(prep.sha256(first), raw_hash)

    def test_preserves_unmanaged_and_modified_files(self):
        self.photo("larb", 1)
        self.run_prepare()
        manifest = self.manifest()
        folder = self.root / "thai_food" / "train" / "larb"
        note = folder / "notes.txt"
        note.write_text("keep")
        extra = folder / "manual.png"
        extra.write_bytes(b"manual")
        with self.assertRaisesRegex(ValueError, "Unmanaged"):
            self.run_prepare()
        self.assertEqual(extra.read_bytes(), b"manual")
        extra.unlink()
        self.run_prepare()
        self.assertEqual(note.read_text(), "keep")
        generated = self.root / "thai_food" / next(iter(manifest))
        generated.write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.run_prepare()
        self.assertEqual(generated.read_bytes(), b"edited")

    def test_shrinking_and_empty_raw_preserve(self):
        paths = [self.photo("larb", i) for i in range(5)]
        self.run_prepare()
        paths[0].unlink()
        self.run_prepare()
        self.assertEqual(len(self.manifest()), 4)
        manifest = self.manifest()
        for path in paths[1:]:
            path.unlink()
        self.run_prepare()
        self.assertEqual(self.manifest(), manifest)

    def test_manifest_traversal_refused(self):
        self.run_prepare()
        (self.root / "thai_food" / prep.MANIFEST).write_text(json.dumps(
            {"version": 1, "files": {"../README.md": "bad"}}))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.run_prepare()

    def test_downloader_metadata_failures_and_rerun(self):
        photo = self.photo("larb", 5)
        payload = photo.read_bytes()
        urls = self.root / "urls.csv"
        urls.write_text("class,source_url,source_provider,license,notes\n"
                        "noodle,https://example.test/photo,manual,check,review\n"
                        "noodle,https://example.test/photo#same,manual,check,review\n"
                        "larb,https://example.test/failure,manual,,\n")
        def fetch(request, timeout):
            self.assertEqual(timeout, 20)
            if request.full_url.endswith("failure"):
                raise OSError("test failure")
            return io.BytesIO(payload)
        with patch.object(downloader.urllib.request, "urlopen", side_effect=fetch), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(downloader.download(urls, self.root, delay=0), 1)
            self.assertEqual(downloader.download(urls, self.root, delay=0), 0)
        with (self.root / "thai_food_raw" / "sources.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["license"], "check")
        self.assertTrue((self.root / "thai_food_raw" / rows[0]["filename"]).is_file())


if __name__ == "__main__":
    unittest.main()
