"""Offline integrity tests for the official test-only source reader.

All ZIP/HTTP fixtures here are synthetic; they do not count as official data.
"""

import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_test_corpus.py"
SPEC = importlib.util.spec_from_file_location("recipegen_test_source_reader", SCRIPT)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


class MemoryArchive:
    archive = "test.zip"
    revision = "synthetic-offline-fixture"

    def __init__(self, raw):
        self.raw, self.size = raw, len(raw)

    def read(self, start, end):
        return self.raw[start:end + 1]


def fixture():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("test/recipe/goal.txt", "Synthetic dish")
        archive.writestr("test/recipe/steps.txt", "1. Chop\n2. Cook\n")
        archive.writestr("test/recipe/1.jpg", b"image payload must not be fetched")
        archive.writestr("test/second/goal.txt", "Second synthetic dish")
        archive.writestr("test/second/steps.txt", "Prepare\nCook\n")
        archive.writestr("__MACOSX/test/recipe/._1.jpg", b"AppleDouble metadata")
    client = MemoryArchive(stream.getvalue())
    entries, _ = reader.build_index(client)
    return client, entries


class TestOnlyCorpusTests(unittest.TestCase):
    def test_full_archive_http_response_is_rejected_before_body_read(self):
        class Response:
            status = 200
            headers = {}
            body_reads = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, amount):
                self.body_reads += 1
                raise AssertionError("Full ZIP body must never be read")

        response = Response()
        with tempfile.TemporaryDirectory() as folder:
            client = reader.RangeClient("test.zip", "fixture", 100, Path(folder), 100, reader.tls_context())
            with patch.object(reader, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "exact HTTP range"):
                    client._network(0, 3)
        self.assertEqual(response.body_reads, 0)

    def test_wrong_http_range_and_size_are_rejected(self):
        for header in ("bytes 1-4/100", "bytes 0-3/101", None):
            with self.subTest(header=header), self.assertRaises(ValueError):
                reader.RangeClient._verify(206, {"Content-Range": header}, 0, 3, 100)
        reader.RangeClient._verify(206, {"Content-Range": "bytes 0-3/100"}, 0, 3, 100)

    def test_truncated_range_body_is_rejected(self):
        class Response:
            status = 206
            headers = {"Content-Range": "bytes 0-3/100"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, amount):
                return b"abc"

            def geturl(self):
                return "https://example.invalid/test.zip"

        with tempfile.TemporaryDirectory() as folder:
            client = reader.RangeClient("test.zip", "fixture", 100, Path(folder), 100, reader.tls_context())
            with patch.object(reader, "urlopen", return_value=Response()):
                with self.assertRaisesRegex(ValueError, "Truncated or oversized"):
                    client._network(0, 3)

    def test_training_archive_is_rejected_without_a_network_request(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(reader, "urlopen") as network:
            for name in ("train-image1.zip", "train-image2.zip", "video1.zip", "video4.zip", "../test.zip"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "test-only whitelist"):
                    reader.RangeClient(name, "fixture", 1, Path(folder), 1, reader.tls_context())
            network.assert_not_called()

    def test_text_crc_failure_is_rejected(self):
        client, entries = fixture()
        entry = next(entry for entry in entries if entry["member"].endswith("goal.txt"))
        raw = bytearray(client.read(entry["header_offset"], entry["entry_end"]))
        raw[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC verification failed"):
            reader.parse_text_entry(entry, bytes(raw), entry["header_offset"])

    def test_chunks_merge_text_but_never_span_media(self):
        client, entries = fixture()
        chunks = reader.chunks_for(entries, {}, 1024 * 1024)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk["entries"]) == 2 for chunk in chunks))
        image = next(entry for entry in entries if entry["kind"] == "image")
        self.assertTrue(all(chunk["end"] < image["header_offset"] or chunk["start"] > image["entry_end"] for chunk in chunks))
        texts = {row["member"]: row for chunk in chunks for row in reader.fetch_chunk(client, chunk)}
        records = reader.records_for(client.archive, client.revision, entries, texts)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(records[0]["steps"]), 2)
        self.assertTrue(all(record["split"] == "test" for record in records))
        self.assertTrue(all(record["provenance"]["media_state"] == "metadata_only" for record in records))

    def test_appledouble_is_metadata_and_not_a_recipe_or_image(self):
        self.assertEqual(reader.member_kind("__MACOSX/test/foo/._1.jpg"), "archive_metadata")
        self.assertEqual(reader.member_kind("test/foo/._video.mp4"), "archive_metadata")
        self.assertEqual(reader.member_kind("test/foo/.ipynb_checkpoints/goal.txt"), "archive_metadata")
        self.assertEqual(reader.member_kind("test/foo/1.jpg"), "image")

    def test_missing_steps_record_is_preserved_without_fabricated_fields(self):
        client, entries = fixture()
        entries = [entry for entry in entries if entry["member"] != "test/recipe/steps.txt"]
        goal = next(entry for entry in entries if entry["member"] == "test/recipe/goal.txt")
        row = reader.parse_text_entry(goal, client.read(goal["header_offset"], goal["entry_end"]), goal["header_offset"])
        records = reader.records_for(client.archive, client.revision, [goal], {row["member"]: row})
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["steps"], [])
        self.assertIn("missing_steps_text", records[0]["quality_flags"])
        self.assertIsNone(records[0]["source"]["steps_member"])
        self.assertNotIn("minutes", records[0])
        self.assertNotIn("ingredients", records[0])

    def test_download_budget_stops_before_network(self):
        with tempfile.TemporaryDirectory() as folder:
            client = reader.RangeClient("test.zip", "fixture", 100, Path(folder), 3, reader.tls_context())
            with patch.object(client, "_network") as network, self.assertRaisesRegex(ValueError, "budget exceeded"):
                client.read(0, 3)
            network.assert_not_called()

    def test_truncated_final_checkpoint_line_is_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "texts.jsonl"
            path.write_text('{"member":"test/a/goal.txt","text":"one"}\n{"member":', encoding="utf-8")
            loaded = reader.load_checkpoint(path)
            self.assertEqual(list(loaded), ["test/a/goal.txt"])
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
