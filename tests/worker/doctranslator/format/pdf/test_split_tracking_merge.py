"""The split path must leave the quality judge something to read.

Each part writes a complete `translate_tracking.json` into its own working
directory -- and `_run_split_translation` then rmtree's that directory in
its `finally:` block. Nothing ever wrote the parent's copy, so on *every*
PDF large enough to split the judge read a non-existent file, scored the
attempt 0.0, and the attempt loop dutifully re-translated the whole
document on every remaining model in the chain.
"""

import json
import threading
from types import SimpleNamespace

from src.worker.doctranslator.format.pdf.high_level import (
    _merge_part_tracking_into_parent,
)
from src.worker.doctranslator.format.pdf.high_level import _write_parent_tracking


def _page(*texts):
    return {"paragraph": [{"input": t, "output": f"{t}-fr"} for t in texts]}


def _write_part(tmp_path, index, pages, cross_page=None, cross_column=None):
    part_dir = tmp_path / f"part_{index}"
    part_dir.mkdir(parents=True, exist_ok=True)
    (part_dir / "translate_tracking.json").write_text(
        json.dumps(
            {
                "page": pages,
                "cross_page": cross_page or [],
                "cross_column": cross_column or [],
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(working_dir=part_dir)


class TestMergePartTrackingIntoParent:
    def test_overlap_pages_are_dropped(self, tmp_path):
        """`tracker.page[j]` maps 1:1 onto `docs.page[j]`, and ResultMerger
        discards exactly the first `overlap_pages` pages of each part. Keeping
        them would feed the judge the same pages two or three times over."""
        part = _write_part(
            tmp_path, 0, [_page("overlap-a"), _page("overlap-b"), _page("real")]
        )
        merged: dict[int, dict] = {}
        _merge_part_tracking_into_parent(
            0,
            part,
            SimpleNamespace(overlap_pages=2),
            merged,
            threading.Lock(),
        )
        assert len(merged[0]["page"]) == 1
        assert merged[0]["page"][0]["paragraph"][0]["input"] == "real"

    def test_zero_overlap_keeps_every_page(self, tmp_path):
        part = _write_part(tmp_path, 0, [_page("a"), _page("b")])
        merged: dict[int, dict] = {}
        _merge_part_tracking_into_parent(
            0, part, SimpleNamespace(overlap_pages=0), merged, threading.Lock()
        )
        assert len(merged[0]["page"]) == 2

    def test_cross_page_and_cross_column_are_kept_whole(self, tmp_path):
        """Those buckets hold paragraphs deliberately excluded from `page`,
        so they are not duplicates of anything -- and they had never been
        judged on any document, split or not."""
        part = _write_part(
            tmp_path,
            0,
            [_page("p1"), _page("p2")],
            cross_page=[_page("spans-pages")],
            cross_column=[_page("spans-columns")],
        )
        merged: dict[int, dict] = {}
        _merge_part_tracking_into_parent(
            0, part, SimpleNamespace(overlap_pages=1), merged, threading.Lock()
        )
        assert len(merged[0]["page"]) == 1
        assert len(merged[0]["cross_page"]) == 1
        assert len(merged[0]["cross_column"]) == 1

    def test_missing_part_file_is_not_fatal(self, tmp_path):
        merged: dict[int, dict] = {}
        _merge_part_tracking_into_parent(
            3,
            SimpleNamespace(working_dir=tmp_path / "gone"),
            SimpleNamespace(overlap_pages=0),
            merged,
            threading.Lock(),
        )
        assert merged == {}

    def test_unparseable_part_file_is_not_fatal(self, tmp_path):
        part_dir = tmp_path / "part_0"
        part_dir.mkdir()
        (part_dir / "translate_tracking.json").write_text("{not json")
        merged: dict[int, dict] = {}
        _merge_part_tracking_into_parent(
            0,
            SimpleNamespace(working_dir=part_dir),
            SimpleNamespace(overlap_pages=0),
            merged,
            threading.Lock(),
        )
        assert merged == {}


class TestWriteParentTracking:
    def test_parts_are_written_in_index_order(self, tmp_path):
        """Parts complete out of order under `as_completed`.

        Deterministic ordering is what keeps the judge's chunk boundaries
        stable across attempts, which is what makes comparing attempt scores
        a like-for-like comparison.
        """
        merged = {
            2: {"page": [_page("c")], "cross_page": [], "cross_column": []},
            0: {"page": [_page("a")], "cross_page": [], "cross_column": []},
            1: {"page": [_page("b")], "cross_page": [], "cross_column": []},
        }
        config = SimpleNamespace(working_dir=tmp_path)
        _write_parent_tracking(config, merged)

        data = json.loads((tmp_path / "translate_tracking.json").read_text())
        assert [p["paragraph"][0]["input"] for p in data["page"]] == ["a", "b", "c"]

    def test_all_three_buckets_are_written(self, tmp_path):
        merged = {
            0: {
                "page": [_page("p")],
                "cross_page": [_page("x")],
                "cross_column": [_page("y")],
            }
        }
        _write_parent_tracking(SimpleNamespace(working_dir=tmp_path), merged)
        data = json.loads((tmp_path / "translate_tracking.json").read_text())
        assert set(data) == {"page", "cross_page", "cross_column"}
        assert len(data["cross_page"]) == 1
        assert len(data["cross_column"]) == 1

    def test_nothing_merged_writes_nothing(self, tmp_path):
        _write_parent_tracking(SimpleNamespace(working_dir=tmp_path), {})
        assert not (tmp_path / "translate_tracking.json").exists()

    def test_output_is_readable_by_the_judge_extractor(self, tmp_path):
        """End-to-end shape check: what the split path writes must be what
        `extract_attempt_segments` reads."""
        from src.worker.services.quality_judge_service import extract_attempt_segments

        merged = {
            0: {
                "page": [_page("one")],
                "cross_page": [_page("two")],
                "cross_column": [],
            }
        }
        _write_parent_tracking(SimpleNamespace(working_dir=tmp_path), merged)
        assert extract_attempt_segments(tmp_path) == [
            ("one", "one-fr"),
            ("two", "two-fr"),
        ]
