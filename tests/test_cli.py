from agis.cli import argv_for


def test_no_args_opens_the_repl():
    assert argv_for([]) == ["repl"]


def test_a_request_runs_once():
    assert argv_for(["clip", "roads", "to", "county"]) == [
        "run", "clip", "roads", "to", "county"]


def test_flags_only_open_the_repl():
    assert argv_for(["-C", "~/data"]) == ["repl", "-C", "~/data"]
    assert argv_for(["-m", "qwen2.5-coder", "-y"]) == ["repl", "-m", "qwen2.5-coder", "-y"]
    assert argv_for(["--model=qwen2.5-coder"]) == ["repl", "--model=qwen2.5-coder"]


def test_flags_plus_a_request_run_once():
    assert argv_for(["-y", "-m", "qwen2.5-coder", "dissolve", "the", "zones"]) == [
        "run", "-y", "-m", "qwen2.5-coder", "dissolve", "the", "zones"]


def test_run_only_flag_stays_a_run():
    assert argv_for(["--dry-run"]) == ["run", "--dry-run"]
    assert argv_for(["-n", "clip", "roads"]) == ["run", "-n", "clip", "roads"]


def test_subcommands_and_help_are_unchanged():
    assert argv_for(["setup"]) == ["setup"]
    assert argv_for(["repl", "-v"]) == ["repl", "-v"]
    assert argv_for(["--help"]) == ["--help"]
    assert argv_for(["--version"]) == ["--version"]
