"""A restart continues a stage from the pieces already on disk."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from botocore.exceptions import ClientError

from app.speech.pieces import run_saved, write_piece


class SavedPiecesTests(unittest.TestCase):
    def test_a_written_piece_is_not_produced_again(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            write_piece(folder, 0, {"segment": {"text": "уже"}})
            seen: list[int] = []

            def produce(index: int) -> dict:
                seen.append(index)
                return {"segment": {"text": f"новый {index}"}}

            ratios: list[float] = []
            rows = run_saved(3, folder, produce, ratios.append)
            self.assertEqual(set(seen), {1, 2})
            self.assertEqual(rows[0]["segment"]["text"], "уже")
            self.assertEqual(rows[1]["segment"]["text"], "новый 1")
            self.assertGreater(ratios[0], 0)
            self.assertEqual(ratios[-1], 1)

    def test_correction_keeps_a_finished_line(self) -> None:
        from app.speech.correct import correct_document

        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            write_piece(folder, 0, {"segment": {"text": "готово", "words": []}})
            document = {"segments": [{"text": "старое", "words": []}, {"text": "ещё", "words": []}]}

            def correct(segment, *, model, label):
                self.assertEqual(segment["text"], "ещё")
                return {"text": "исправлено", "words": []}

            with patch("app.speech.correct._correct_segment", side_effect=correct):
                result = correct_document(document, model="x", folder=folder)
            self.assertEqual([item["text"] for item in result["segments"]], ["готово", "исправлено"])

    def test_a_finished_merge_is_not_rebuilt(self) -> None:
        from app.speech.chunks import save_json
        from app.speech.merge import merge_document

        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            save_json(folder / "result.json", {"segments": [{"text": "склеено"}]})
            with patch("app.speech.merge._merge_slice", side_effect=AssertionError("срез заново")):
                result = merge_document({"segments": [{"text": "а"}, {"text": "б"}]}, model="x", folder=folder)
            self.assertEqual(result["segments"][0]["text"], "склеено")

    def test_split_keeps_a_finished_line(self) -> None:
        from app.speech.split import split_document

        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            write_piece(folder, 0, {"pieces": [{"text": "первая"}]})
            document = {"segments": [{"text": "старая"}, {"text": "вторая часть."}]}
            result = split_document(document, folder=folder)
            again = split_document({"segments": [{"text": "другие"}]}, folder=folder)
            self.assertEqual(result["segments"][0]["text"], "первая")
            self.assertEqual(again["segments"], result["segments"])

    def test_merge_keeps_decisions_inside_a_slice(self) -> None:
        from app.speech.merge import merge_document

        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            write_piece(
                folder,
                0,
                {
                    "segments": [{"text": "а б", "start": 0, "end": 2, "speaker": "A"}],
                    "done": False,
                    "cursor": 2,
                },
            )
            document = {
                "segments": [
                    {"text": "а", "start": 0, "end": 1, "speaker": "A"},
                    {"text": "б", "start": 1, "end": 2, "speaker": "A"},
                    {"text": "в", "start": 2, "end": 3, "speaker": "A"},
                ]
            }
            seen: list[str] = []

            def decision(left, right, model):
                seen.append(right["text"])
                return "keep"

            with (
                patch("app.speech.merge.even_spans", return_value=[(0, 3)]),
                patch("app.speech.merge._decision", side_effect=decision),
            ):
                merge_document(document, model="x", folder=folder)
            self.assertEqual(seen, ["в"])

    def test_download_keeps_a_finished_file_and_partials(self) -> None:
        from app import fetch

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            video_id = uuid4()
            folder = root / str(video_id)
            folder.mkdir()
            part = folder / "source.mp4.part"
            part.write_bytes(b"piece")
            (folder / "source.mp4").write_bytes(b"ready-file")
            seen: list[dict] = []

            class Downloader:
                def __init__(self, options):
                    seen.append(options)

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def extract_info(self, url, download=False):
                    return {"title": "Ролик", "duration": 4, "formats": []}

                def download(self, urls):
                    raise AssertionError("скачивание заново")

            with (
                patch.object(fetch, "ensure_ytdlp"),
                patch.object(fetch, "_module", SimpleNamespace(YoutubeDL=Downloader)),
                patch.object(fetch, "pick_format", return_value={"format_id": "18"}),
                patch.object(fetch, "mark_fetched", return_value=True),
                patch.object(fetch, "set_title"),
                patch.object(fetch, "set_progress"),
                patch.object(fetch, "log_event"),
                patch.object(fetch, "WORK_DIR", root),
            ):
                fetch.download_link("https://example.test/v", video_id)
            self.assertTrue(part.is_file())
            self.assertFalse(any(item.get("continuedl") is False for item in seen))

    def test_store_does_not_send_a_finished_part_again(self) -> None:
        from app.storage import upload_file

        class Store:
            def __init__(self) -> None:
                self.sent: list[int] = []
                self.fail_once = True
                self.created = 0

            def head_object(self, **kwargs):
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

            def create_multipart_upload(self, **kwargs):
                self.created += 1
                return {"UploadId": "up-1"}

            def list_parts(self, **kwargs):
                return {
                    "Parts": [{"PartNumber": number, "ETag": f"e{number}"} for number in self.sent],
                    "IsTruncated": False,
                }

            def upload_part(self, **kwargs):
                number = int(kwargs["PartNumber"])
                if number == 2 and self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("обрыв")
                self.sent.append(number)
                return {"ETag": f"e{number}"}

            def complete_multipart_upload(self, **kwargs):
                self.completed = [part["PartNumber"] for part in kwargs["MultipartUpload"]["Parts"]]

            def abort_multipart_upload(self, **kwargs):
                raise AssertionError("сборка сброшена")

        store = Store()
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "play.mp4"
            path.write_bytes(b"0123456789")
            with patch("app.storage.client", return_value=store), patch("app.storage.PART_BYTES", 4):
                with self.assertRaises(RuntimeError):
                    upload_file(path, "videos/one.mp4")
                upload_file(path, "videos/one.mp4")
        self.assertEqual(store.created, 1)
        self.assertEqual(store.sent, [1, 2, 3])
        self.assertEqual(store.completed, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
