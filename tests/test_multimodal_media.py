"""Offline media integrity tests; generated fixtures are not RecipeGen data."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch
from urllib.error import URLError
import zipfile

import pytest

from recipegen.multimodal_media import (
    MediaLimits, MediaPreparer, MediaRangeFile, extract_video_frames,
    file_sha256, plan_clips, read_zip_member, select_records, source_id,
)
from recipegen.test_graph import REPO, uid


REVISION = "a" * 40


class MemoryReader:
    def __init__(self, archive, revision, size, limits, raw=b""):
        self.raw = raw
        self.position = 0
        self.reset_budget()

    def reset_budget(self):
        self.requests = self.downloaded = self.attempted_bytes = self.attempts = 0

    def seek(self, position):
        self.position = position

    def read(self, size):
        result = self.raw[self.position:self.position + size]
        self.position += len(result)
        self.requests += 1
        self.attempts += 1
        self.downloaded += len(result)
        self.attempted_bytes += len(result)
        return result

    def close(self):
        pass


def archive_fixture(compression=zipfile.ZIP_STORED):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w", compression=compression) as z:
        z.writestr("test/recipe/1.jpg", b"offline-synthetic-image-payload" * 100)
    raw = blob.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        info = z.infolist()[0]
        entry = {"member": info.filename, "display_path": info.filename, "kind": "image", "size": info.file_size, "compressed_size": info.compress_size, "crc": f"{info.CRC:08x}", "compression": info.compress_type, "flags": info.flag_bits, "header_offset": info.header_offset, "entry_end": z.start_dir - 1, "source": {"archive": "test.zip", "revision": REVISION, "split": "test"}}
    return raw, entry


def corpus_fixture(root):
    raw, entry = archive_fixture()
    source = root / "data/test_graph/source"
    source.mkdir(parents=True)
    for archive, entries in (("test.zip", [entry]), ("test-video.zip", [])):
        index = source / f"{archive[:-4]}.members.jsonl"
        index.write_text("".join(json.dumps(row) + "\n" for row in entries))
        manifest = {"archive": archive, "split": "test", "revision": REVISION, "status": "complete", "archive_size": len(raw), "member_count": len(entries), "index_sha256": file_sha256(index)}
        (source / f"{archive[:-4]}.manifest.json").write_text(json.dumps(manifest))
    record = {"id": "recipe-fixture", "title": "Offline synthetic dish", "split": "test", "source": {"archive": "test.zip", "split": "test", "revision": REVISION, "steps_member": "test/recipe/steps.txt"}, "steps": [{"order": 1, "text": "Synthetic instruction"}], "media": [entry]}
    (source.parent / "records.jsonl").write_text(json.dumps(record) + "\n")
    return raw, entry, record


def test_graph_id_algorithms_are_preserved():
    assert source_id("test.zip", "test/x/1.jpg") == uid("source", f"{REPO}\ntest.zip\ntest/x/1.jpg")
    assert source_id("test.zip", "test/x/1.jpg") != uid("source", "test.zip\ntest/x/1.jpg")


def test_record_selection_is_stable_and_seeked(tmp_path):
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps({"id": key}) for key in ("z", "a", "m")))
    assert [row["id"] for row in select_records(records)] == ["a", "m", "z"]
    assert [row["id"] for row in select_records(records, limit=1, offset=1)] == ["m"]
    assert [row["id"] for row in select_records(records, recipe_id="z")] == ["z"]
    with pytest.raises(ValueError, match="Unknown"):
        list(select_records(records, recipe_id="missing"))
    with pytest.raises(ValueError):
        list(select_records(records, limit=-1))


def test_train_archive_refused_before_network():
    for archive in ("train-image1.zip", "video1.zip", "../test.zip", "test-other.zip"):
        with pytest.raises(ValueError, match="test-only whitelist"):
            MediaRangeFile(archive, REVISION, 10, MediaLimits())


def test_full_response_and_wrong_range_are_rejected():
    for status, header in ((200, None), (206, "bytes 0-2/10"), (206, "bytes 0-3/11")):
        with pytest.raises(ValueError, match="Range"):
            MediaRangeFile.verify_response(status, {"Content-Range": header}, 0, 3, 10)
    MediaRangeFile.verify_response(206, {"Content-Range": "bytes 0-3/10"}, 0, 3, 10)


def test_http_retry_count_and_budget_are_bounded():
    reader = MediaRangeFile("test.zip", REVISION, 100, MediaLimits(http_attempts=3))
    with patch("recipegen.multimodal_media.urlopen", side_effect=URLError("offline")) as request, patch("recipegen.multimodal_media.time.sleep"):
        with pytest.raises(URLError):
            reader.read(10)
        assert request.call_count == 3
        assert reader.attempted_bytes == 30
    reader = MediaRangeFile("test.zip", REVISION, 100, MediaLimits(max_download_bytes=1))
    with patch("recipegen.multimodal_media.urlopen") as request:
        with pytest.raises(ValueError, match="budget"):
            reader.read(10)
        request.assert_not_called()


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_streaming_zip_crc_sha_and_member_boundary(tmp_path, compression):
    raw, entry = archive_fixture(compression)
    target = tmp_path / "out.jpg"
    result = read_zip_member(MemoryReader("test.zip", REVISION, len(raw), MediaLimits(), raw), entry, target, MediaLimits(chunk_bytes=100))
    assert result["zip_crc_verified"] is True
    assert result["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.stat().st_size == entry["size"]
    bad = {**entry, "entry_end": entry["header_offset"] + 35}
    with pytest.raises(ValueError, match="boundary"):
        read_zip_member(MemoryReader("test.zip", REVISION, len(raw), MediaLimits(), raw), bad, target, MediaLimits())


def test_corrupt_payload_leaves_no_file(tmp_path):
    raw, entry = archive_fixture()
    damaged = bytearray(raw)
    damaged[entry["entry_end"]] ^= 1
    target = tmp_path / "out.jpg"
    with pytest.raises(ValueError, match="CRC"):
        read_zip_member(MemoryReader("test.zip", REVISION, len(raw), MediaLimits(), bytes(damaged)), entry, target, MediaLimits())
    assert not target.exists()
    assert not target.with_suffix(".jpg.part").exists()


def test_member_limit_stops_before_reading(tmp_path):
    raw, entry = archive_fixture()
    reader = MemoryReader("test.zip", REVISION, len(raw), MediaLimits(), raw)
    with pytest.raises(ValueError, match="max_file_bytes"):
        read_zip_member(reader, entry, tmp_path / "a.jpg", MediaLimits(max_file_bytes=1))
    assert reader.requests == 0


def test_real_time_windows_do_not_depend_on_step_count():
    frames = [{"decode_index": n, "pts": 100 + n * 10, "timestamp_sec": 10 + n, "time_base": "1/10"} for n in range(12)]
    metadata = {"duration_sec": 12.0, "timeline_origin_sec": 10.0, "time_base": "1/10"}
    clips = plan_clips("video:test", metadata, frames, MediaLimits())
    assert [(c["start_sec"], c["end_sec"]) for c in clips] == [(0, 5), (5, 10), (10, 12)]
    assert [[f["offset_sec"] for f in c["frames"]] for c in clips] == [[0, 2, 4], [5, 7, 9], [10, 11]]
    assert clips[-1]["quality_flags"] == ["fewer_distinct_frames_than_requested"]
    with pytest.raises(ValueError, match="not silently truncated"):
        plan_clips("video:test", metadata, frames, MediaLimits(max_clips=2))
    with pytest.raises(ValueError, match="cannot be invented"):
        plan_clips("video:test", {**metadata, "duration_sec": 20}, frames, MediaLimits())


def test_prepare_resume_checks_sha_and_cleanup_preserves_evidence(tmp_path):
    raw, entry, record = corpus_fixture(tmp_path)
    factory = lambda *args: MemoryReader(*args, raw=raw)
    with MediaPreparer(tmp_path, reader_factory=factory) as preparer:
        row = preparer.prepare_record(record)
        assert row["status"] == "complete"
        assert row["steps"][0]["id"] == uid("step", "recipe-fixture\n1")
        assert row["images"][0]["media_id"] == uid("image", "test.zip\n" + entry["member"])
        assert row["images"][0]["source_id"] == source_id("test.zip", entry["member"])
        again = preparer.prepare_record(record)
        assert again["status"] == "complete"
        assert again["images"][0]["reused"] is True
        assert again["range"]["test.zip"]["requests"] == 0
        (tmp_path / again["images"][0]["path"]).write_bytes(b"corrupt")
        assert preparer.prepare_record(record)["status"] == "failed"
        cleaned = preparer.cleanup_record(record["id"])
        assert cleaned["storage_state"] == "cleaned"
        assert not preparer._recipe_dir(record["id"]).exists()
        assert preparer.output_path.exists()
        rows = [json.loads(line) for line in preparer.output_path.read_text().splitlines()]
        assert rows[0]["images"][0]["sha256"] == row["images"][0]["sha256"]


def test_corrupt_inventory_fails_without_network(tmp_path):
    _, _, _ = corpus_fixture(tmp_path)
    index = tmp_path / "data/test_graph/source/test.members.jsonl"
    index.write_text(index.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA"):
        MediaPreparer(tmp_path)


def test_orphan_media_does_not_create_recipe(tmp_path):
    raw, entry, _ = corpus_fixture(tmp_path)
    (tmp_path / "data/test_graph/records.jsonl").write_text("")
    with MediaPreparer(tmp_path, reader_factory=lambda *args: MemoryReader(*args, raw=raw)) as helper:
        orphan = next(helper.iter_orphan_media())
        row = helper.prepare_orphan_media(orphan)
        assert row["status"] == "complete"
        assert row["recipe_id"] is None and row["steps"] == []
        assert row["images"][0]["source_id"] == source_id("test.zip", entry["member"])
        cleaned = helper.cleanup_record(row["work_id"])
        assert cleaned["recipe_id"] is None
        assert cleaned["images"][0]["storage_state"] == "cleaned"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe unavailable")
def test_actual_ffmpeg_pts_and_longest_side_on_synthetic_video(tmp_path):
    video = tmp_path / "synthetic.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "1.2", "-c:v", "mpeg4", "-y", str(video)], check=True)
    result = extract_video_frames(video, "video:synthetic", tmp_path / "frames", tmp_path, MediaLimits())
    assert result["clip_count"] == 1
    assert result["duration_sec"] == pytest.approx(1.2)
    assert result["sampled_frame_count"] == 3
    frames = result["clips"][0]["frames"]
    assert [f["timestamp_sec"] for f in frames] == pytest.approx([0, 0.6, 1.1])
    for frame in frames:
        assert frame["pts_verified_by_ffmpeg"] is True
        assert max(frame["width"], frame["height"]) <= 448
        assert file_sha256(tmp_path / frame["path"]) == frame["sha256"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe unavailable")
def test_video_original_nonzero_pts_is_preserved(tmp_path):
    video = tmp_path / "nonzero.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10", "-t", "1.2", "-c:v", "mpeg4", "-output_ts_offset", "2", "-y", str(video)], check=True)
    result = extract_video_frames(video, "video:offset", tmp_path / "frames", tmp_path, MediaLimits())
    assert result["timeline_origin_sec"] == pytest.approx(2)
    assert result["duration_sec"] == pytest.approx(1.2)
    assert result["clips"][0]["start_sec"] == 0
    assert result["clips"][0]["frames"][0]["timestamp_sec"] == pytest.approx(2)
    assert result["clips"][0]["frames"][0]["offset_sec"] == 0
