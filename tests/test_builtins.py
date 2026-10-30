from io import StringIO

from rich.console import Console

from agis.cli import DatasetCompleter, try_builtin
from agis.session import Session


class FakeBackend:
    model = "fake"

    def __init__(self):
        self.calls = []

    def generate_json(self, messages, schema):
        self.calls.append(messages)
        return '{"steps": [], "message": ""}'


def _console():
    buffer = StringIO()
    return buffer, Console(file=buffer, highlight=False, force_terminal=False, width=200)


def test_ls_lists_the_directory_without_calling_the_model(tmp_path, capfd):
    (tmp_path / "roads.gpkg").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "notes.txt").write_text("x")
    backend = FakeBackend()
    session = Session(backend, tmp_path)
    _, console = _console()
    assert try_builtin("ls", session, None, console)
    assert try_builtin("ls sub", session, None, console)
    captured = capfd.readouterr().out
    assert "roads.gpkg" in captured
    assert "notes.txt" in captured
    assert backend.calls == []


def test_cd_updates_the_session_and_completer_then_pwd(tmp_path):
    sub = tmp_path / "My Folder"
    sub.mkdir()
    backend = FakeBackend()
    session = Session(backend, tmp_path)
    completer = DatasetCompleter(tmp_path)
    buffer, console = _console()
    assert try_builtin('cd "My Folder"', session, completer, console)
    assert session.cwd == sub.resolve()
    assert completer.cwd == sub.resolve()
    buffer.seek(0)
    buffer.truncate()
    assert try_builtin("pwd", session, completer, console)
    assert str(sub.resolve()) in buffer.getvalue()
    assert backend.calls == []


def test_cd_to_a_missing_path_stays_put(tmp_path):
    session = Session(FakeBackend(), tmp_path)
    start = session.cwd
    (tmp_path / "notes.txt").write_text("x")
    _, console = _console()
    assert try_builtin("cd missing", session, None, console)
    assert try_builtin("cd notes.txt", session, None, console)
    assert session.cwd == start


def test_a_gis_request_is_not_a_builtin(tmp_path):
    session = Session(FakeBackend(), tmp_path)
    _, console = _console()
    assert try_builtin("clip roads", session, None, console) is False
    assert try_builtin("ls | head", session, None, console) is False
