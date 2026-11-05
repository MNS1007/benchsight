"""Session cleanup: this session's files go when it ends, nothing else ever does.
Temp stores only; the end-of-session hooks are proven in real subprocesses."""
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from benchsight import camera, cleanup, events, server

OLD = time.time() - 30 * 3600


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("BENCHSIGHT_KEEP", raising=False)
    monkeypatch.setattr(cleanup, "_files", {})
    yield


def img():
    return np.full((48, 64, 3), 128, np.uint8)


def decoys(store):
    """Files a user might keep in a store pointed at a real folder."""
    (store / "photo.jpg").write_bytes(b"x")
    (store / "notes.txt").write_text("mine")
    (store / "sub").mkdir()
    (store / "sub" / "20260101-000000-000-look.jpg").write_bytes(b"x")
    return {store / "photo.jpg", store / "notes.txt", store / "sub" / "20260101-000000-000-look.jpg"}


def test_session_end_deletes_this_sessions_files_and_nothing_else(tmp_path):
    store = tmp_path / "store"
    shots = [camera.save(img(), n, store=store).path for n in ("look", "watch", "point")]
    keep = decoys(store)
    other = store / "20991231-235959-999-look.jpg"  # another session's, young: not ours to delete
    other.write_bytes(b"x")
    outside = tmp_path / "20260101-000000-000-look.jpg"  # recorded but not in the store
    outside.write_bytes(b"x")
    cleanup.record(outside, store)
    cleanup.record(store / "notes.txt")  # recorded, but not a name benchsight makes
    size = sum(p.stat().st_size for p in shots)
    assert cleanup.session_end() == (3, size)
    assert not any(p.exists() for p in shots)
    assert all(p.exists() for p in keep | {other, outside})
    assert cleanup.session_end() == (0, 0), "idempotent"


def test_session_end_is_fine_when_the_store_is_gone(tmp_path):
    store = tmp_path / "store"
    camera.save(img(), "look", store=store)
    for p in store.iterdir():
        p.unlink()
    store.rmdir()
    assert cleanup.session_end() == (0, 0)


def test_symlinks_are_never_deleted(tmp_path):
    store, target = tmp_path / "store", tmp_path / "precious.jpg"
    store.mkdir()
    target.write_bytes(b"x")
    link = store / "20260101-000000-000-look.jpg"
    link.symlink_to(target)
    cleanup.record(link, store)
    cleanup.session_end()
    assert link.is_symlink() and target.exists()


def test_keep_env_keeps_them(tmp_path, monkeypatch):
    shot = camera.save(img(), "look", store=tmp_path).path
    monkeypatch.setenv("BENCHSIGHT_KEEP", "1")
    assert cleanup.session_end() == (0, 0) and shot.exists()


def test_the_24_hour_backstop_only_touches_benchsight_files(tmp_path):
    old = tmp_path / "20200101-000000-000-look.jpg"
    foreign = tmp_path / "holiday.jpg"
    for p in (old, foreign):
        p.write_bytes(b"x")
        os.utime(p, (OLD, OLD))
    camera.save(img(), "look", store=tmp_path)
    assert not old.exists() and foreign.exists()


def test_the_events_file_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(camera, "STORE", tmp_path)
    t = server._make_tracker(None)
    t.events.add("still", 0.0)
    t.events.flush(0.0)
    assert (tmp_path / "events.jsonl").exists()
    assert cleanup.session_end()[0] == 1 and not (tmp_path / "events.jsonl").exists()


def test_cleanup_tool_reports_files_and_bytes(tmp_path):
    a = camera.save(img(), "look", store=tmp_path).path
    b = camera.save(img(), "watch", store=tmp_path).path
    size = a.stat().st_size + b.stat().st_size
    got = server.cleanup_files()
    assert got["files"] == 2 and got["bytes"] == size and not a.exists() and not b.exists()


# ── the real hooks, in a subprocess ──

SCRIPT = textwrap.dedent("""
    import sys, time, numpy as np
    from benchsight import camera, cleanup
    cleanup.install()
    p = camera.save(np.zeros((8, 8, 3), np.uint8), "look").path
    print(p, flush=True)
    if sys.argv[1] == "wait":
        time.sleep(30)
""")


def spawn(tmp_path, mode, **env):
    e = {k: v for k, v in os.environ.items() if k != "BENCHSIGHT_KEEP"}
    e.update(BENCHSIGHT_DIR=str(tmp_path), **env)
    return subprocess.Popen([sys.executable, "-c", SCRIPT, mode], env=e, stdout=subprocess.PIPE, text=True)


def saved(proc):
    from pathlib import Path
    return Path(proc.stdout.readline().strip())


def test_deleted_on_normal_exit(tmp_path):
    p = spawn(tmp_path, "exit")
    path = saved(p)
    assert p.wait(20) == 0 and str(path).startswith(str(tmp_path)) and not path.exists()


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_deleted_on_signal(tmp_path, sig):
    p = spawn(tmp_path, "wait")
    path = saved(p)
    assert path.exists()
    p.send_signal(sig)
    p.wait(20)
    assert not path.exists()


def test_kept_with_benchsight_keep(tmp_path):
    p = spawn(tmp_path, "exit", BENCHSIGHT_KEEP="1")
    path = saved(p)
    p.wait(20)
    assert path.exists()


def test_deleted_when_the_mcp_client_closes_stdio(tmp_path):
    script = textwrap.dedent("""
        import numpy as np
        from benchsight import camera, server
        p = camera.save(np.zeros((8, 8, 3), np.uint8), "look").path
        open(__import__("sys").argv[1], "w").write(str(p))
        server.main()
    """)
    note = tmp_path / "path.txt"
    store = tmp_path / "store"
    env = {k: v for k, v in os.environ.items() if k != "BENCHSIGHT_KEEP"}
    p = subprocess.Popen([sys.executable, "-c", script, str(note)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**env, "BENCHSIGHT_DIR": str(store)})
    end = time.time() + 20
    while not note.exists() and time.time() < end:
        time.sleep(0.05)
    path = note.read_text()
    assert os.path.exists(path)
    p.stdin.close()
    p.wait(20)
    assert not os.path.exists(path)


# ── benchsight clean ──

def fill(store, n_old=2, n_new=2):
    store.mkdir(parents=True, exist_ok=True)
    old = [store / f"2020010{i}-000000-000-look.jpg" for i in range(1, n_old + 1)]
    new = [store / f"2099010{i}-000000-000-watch.jpg" for i in range(1, n_new + 1)]
    for p in old + new:
        p.write_bytes(b"x" * 100)
    for p in old:
        os.utime(p, (OLD, OLD))
    (store / "events.jsonl").write_text("{}\n")
    return old, new


def test_clean_deletes_only_benchsight_files_older_than_24_h(tmp_path, capsys):
    from benchsight import cli
    old, new = fill(tmp_path)
    keep = decoys(tmp_path)
    for p in keep:
        os.utime(p, (OLD, OLD))
    cli.main(["clean", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert not any(p.exists() for p in old) and all(p.exists() for p in new + list(keep))
    assert (tmp_path / "events.jsonl").exists()
    assert "2 files deleted, 200 B freed" in out


def test_clean_older_than_and_dry_run(tmp_path, capsys):
    from benchsight import cli
    old, new = fill(tmp_path)
    cli.main(["clean", "--dir", str(tmp_path), "--older-than", "0", "--dry-run"])
    out = capsys.readouterr().out
    assert all(p.exists() for p in old + new) and "5 files" in out and "would delete" in out
    cli.main(["clean", "--dir", str(tmp_path), "--older-than", "0"])
    assert not any(p.exists() for p in old + new)


def test_clean_all_spares_everything_not_benchsight(tmp_path, capsys):
    from benchsight import cli
    fill(tmp_path)
    keep = decoys(tmp_path)
    cli.main(["clean", "--dir", str(tmp_path), "--all"])
    assert set(p for p in tmp_path.rglob("*") if p.is_file()) == keep
    assert "5 files deleted" in capsys.readouterr().out
    cli.main(["clean", "--dir", str(tmp_path / "missing"), "--all"])
    assert "0 files deleted" in capsys.readouterr().out


def fake_claude(home):
    """A fake ~/.claude with benchsight blobs where Claude Code puts them, and decoys."""
    proj = home / ".claude" / "projects" / "-Users-me-bench"
    ours = [proj / "sess-1" / "tool-results" / "mcp-benchsight-blob-1-a.jpg",
            proj / "tool-results" / "mcp-benchsight-blob-2-b.jpg"]
    outside = home / "precious.jpg"
    other = [proj / "sess-1" / "tool-results" / "mcp-other-blob-1.jpg",  # another server's
             proj / "sess-1" / "tool-results" / "toolu_123.txt",
             proj / "sess-1" / "mcp-benchsight-blob-3.jpg",  # not in tool-results
             proj / "sess-1.jsonl",
             home / ".claude" / "settings.json",
             outside]
    for p in ours + other:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"y" * 10)
    link = proj / "sess-1" / "tool-results" / "mcp-benchsight-blob-link.jpg"
    link.symlink_to(outside)
    folder = proj / "sess-1" / "tool-results" / "mcp-benchsight-blob-dir"
    folder.mkdir()
    (folder / "inside.jpg").write_bytes(b"z")
    return ours, other + [link, folder / "inside.jpg"]


def test_clean_agent_cache_deletes_only_benchsight_blobs(tmp_path, monkeypatch, capsys):
    from benchsight import cli
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    ours, others = fake_claude(home)
    store = tmp_path / "store"
    fill(store)
    cli.main(["clean", "--dir", str(store), "--all", "--agent-cache", "--dry-run"])
    out = capsys.readouterr().out
    assert all(p.exists() for p in ours) and all(str(p) in out for p in ours)
    assert not any(str(p) in out for p in others)
    cli.main(["clean", "--dir", str(store), "--all", "--agent-cache"])
    assert not any(p.exists() for p in ours)
    assert all(p.exists() or p.is_symlink() for p in others)
    assert "7 files deleted" in capsys.readouterr().out


def test_agent_cache_respects_age_and_is_off_by_default(tmp_path, monkeypatch, capsys):
    from benchsight import cli
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    ours, _ = fake_claude(home)
    os.utime(ours[0], (OLD, OLD))
    cli.main(["clean", "--dir", str(tmp_path / "store")])
    assert all(p.exists() for p in ours)
    cli.main(["clean", "--dir", str(tmp_path / "store"), "--agent-cache"])
    assert not ours[0].exists() and ours[1].exists(), "a running session's young copies stay"
