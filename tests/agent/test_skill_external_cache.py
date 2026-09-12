from pathlib import Path


def _skill(path: Path, name: str, description: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nbody\n",
        encoding="utf-8",
    )


def test_skill_prompt_cache_invalidates_external_skill_changes(tmp_path, monkeypatch):
    from agent import prompt_builder as pb

    local = tmp_path / "local" / "local-skill"
    external = tmp_path / "external" / "external-skill"
    _skill(local, "local-skill", "local description")
    _skill(external, "external-skill", "old external description")
    monkeypatch.setattr(pb, "_skills_prompt_snapshot_path", lambda: tmp_path / "snapshot.json")
    pb._SKILLS_PROMPT_CACHE.clear()

    first = pb._build_skills_system_prompt_inner(
        local.parent, [external.parent], None, None, None, [], index_mode="compact"
    )
    (external / "SKILL.md").write_text(
        "---\nname: external-skill\ndescription: new external description\n---\n\nbody\n",
        encoding="utf-8",
    )
    second = pb._build_skills_system_prompt_inner(
        local.parent, [external.parent], None, None, None, [], index_mode="compact"
    )

    assert "old external description" in first
    assert "new external description" in second
