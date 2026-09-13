from pathlib import Path


def test_selective_project_context_does_not_load_git_root_agents(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    root = Path(tmp_path)
    (root / "AGENTS.md").write_text("large root instructions", encoding="utf-8")
    monkeypatch.setattr(pb, "_find_git_root", lambda _path: root)

    assert pb._load_agents_md(root, mode="selective") == ""
    assert "large root instructions" in pb._load_agents_md(root, mode="full")
