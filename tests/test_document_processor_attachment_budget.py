from pathlib import Path


class _UploadHandler:
    def __init__(self, uploads):
        self.uploads = uploads

    def resolve_upload(self, fid, owner=None):
        return self.uploads.get(fid)

    def _inside_upload_dir(self, path):
        return True

    def is_image_file(self, display_name, mime):
        return False

    def is_audio_file(self, display_name, mime):
        return False

    def is_document_file(self, display_name, mime):
        return True


def _text_upload(tmp_path: Path, fid: str, body: str):
    path = tmp_path / f"{fid}.txt"
    path.write_text(body, encoding="utf-8")
    return {
        "path": str(path),
        "name": path.name,
        "mime": "text/plain",
    }


def test_multifile_inline_attachment_budget_keeps_later_files_visible(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 1200)
    monkeypatch.setattr(dp, "MIN_INLINE_ATTACHMENT_SLICE", 200)
    uploads = {
        "a": _text_upload(tmp_path, "a", "alpha\n" + ("A" * 1000)),
        "b": _text_upload(tmp_path, "b", "bravo\n" + ("B" * 1000)),
        "c": _text_upload(tmp_path, "c", "charlie\n" + ("C" * 1000)),
    }

    content = dp.build_user_content(
        "How many files do you see?",
        ["a", "b", "c"],
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    assert "=== File: a.txt ===" in content
    assert "=== File: c.txt ===" not in content
    assert "Attachment omitted from inline context: b.txt" in content
    assert "Attachment omitted from inline context: c.txt" in content
    assert "Ask to inspect this file specifically" in content
    assert "--- BEGIN ATTACHED FILE 1/3: a.txt ---" in content
    assert "--- BEGIN ATTACHED FILE 3/3: c.txt ---" in content
    assert len(content) < 2600


def test_inline_attachment_budget_does_not_truncate_small_batches(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 5000)
    uploads = {
        "a": _text_upload(tmp_path, "a", "alpha"),
        "b": _text_upload(tmp_path, "b", "bravo"),
    }

    content = dp.build_user_content(
        "Summarize these.",
        ["a", "b"],
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    assert "=== File: a.txt ===" in content
    assert "=== File: b.txt ===" in content
    assert "Attachment content truncated" not in content
    assert "--- BEGIN ATTACHED FILE 1/2: a.txt ---" in content
    assert "--- END ATTACHED FILE 2/2: b.txt ---" in content


def test_text_attachment_uses_original_filename_not_upload_id(tmp_path):
    import src.document_processor as dp

    stored_name = "cd1d913f7ded4d76a239bb74723e4a00.md"
    path = tmp_path / stored_name
    path.write_text("# Notes\n\nOriginal file body", encoding="utf-8")
    uploads = {
        stored_name: {
            "path": str(path),
            "name": stored_name,
            "original_name": "project-notes.md",
            "mime": "text/markdown",
            "size": path.stat().st_size,
        }
    }

    content = dp.build_user_content(
        "Name this file.",
        [stored_name],
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    assert "--- BEGIN ATTACHED FILE 1/1: project-notes.md ---" in content
    assert "=== File: project-notes.md ===" in content
    assert stored_name not in content


def test_attachment_display_name_strips_path_and_control_characters(tmp_path):
    import src.document_processor as dp

    stored_name = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.md"
    path = tmp_path / stored_name
    path.write_text("body", encoding="utf-8")
    uploads = {
        "upload-id": {
            "path": str(path),
            "name": stored_name,
            "original_name": "..\\nested\\project\nnotes.md",
            "mime": "text/markdown",
            "size": path.stat().st_size,
        }
    }

    content = dp.build_user_content(
        "Name this file.",
        ["upload-id"],
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    assert "--- BEGIN ATTACHED FILE 1/1: project notes.md ---" in content
    assert "=== File: project notes.md ===" in content
    assert "nested" not in content


def test_ten_text_attachments_are_individually_labeled(tmp_path, monkeypatch):
    import src.document_processor as dp

    monkeypatch.setattr(dp, "MAX_INLINE_ATTACHMENT_CHARS", 20000)
    uploads = {}
    ids = []
    stored_names = []
    for idx in range(10):
        stored_name = f"{idx + 1:032x}.txt"
        original_name = f"notes-{idx + 1}.txt"
        path = tmp_path / stored_name
        path.write_text(f"body {idx + 1}", encoding="utf-8")
        uploads[stored_name] = {
            "path": str(path),
            "name": stored_name,
            "original_name": original_name,
            "mime": "text/plain",
            "size": path.stat().st_size,
        }
        ids.append(stored_name)
        stored_names.append(stored_name)

    content = dp.build_user_content(
        "Name each attached file.",
        ids,
        str(tmp_path),
        _UploadHandler(uploads),
        owner="tester",
    )

    for idx in range(10):
        original_name = f"notes-{idx + 1}.txt"
        assert f"--- BEGIN ATTACHED FILE {idx + 1}/10: {original_name} ---" in content
        assert f"--- END ATTACHED FILE {idx + 1}/10: {original_name} ---" in content
        assert f"=== File: {original_name} ===" in content
        assert f"body {idx + 1}" in content
    for stored_name in stored_names:
        assert stored_name not in content
