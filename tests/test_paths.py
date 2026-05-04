from gpuq import paths


def test_state_dir_uses_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert paths.state_dir() == tmp_path / "gpuq"


def test_state_dir_default(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.state_dir() == tmp_path / ".local" / "state" / "gpuq"


def test_socket_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert paths.socket_path() == tmp_path / "gpuq" / "gpuq.sock"


def test_db_path(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert paths.db_path() == tmp_path / "gpuq" / "state.db"


def test_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert paths.log_dir("j_abc123") == tmp_path / "gpuq" / "logs" / "j_abc123"


def test_ensure_creates_dirs(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    paths.ensure()
    assert (tmp_path / "gpuq" / "logs").is_dir()
