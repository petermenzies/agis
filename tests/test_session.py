from io import StringIO

from rich.console import Console

from agis.executor import ResolvedStep
from agis.ops.attributes import Condition, Filter
from agis.plan import Plan
from agis.session import Session, _detail


def test_preview_shows_full_operation_details(tmp_path):
    filename = "very_long_county_boundary_layer_name_that_used_to_be_clipped.gpkg"
    status = "a long planned status value that exceeds the old limit"
    output = tmp_path / "roads_filtered_by_a_very_long_status_name.gpkg"
    args = Filter(
        input=filename,
        where=[Condition(field="status", op="==", value=status)],
    )
    step = ResolvedStep(
        number=1, args=args, inputs={}, output=output, kind="vector",
        temporary=False, stem="roads",
    )
    plan = Plan(steps=[args], message="Keep the planned roads.")
    narrow = _preview(plan, step, tmp_path, width=60)
    assert filename in "".join(narrow.split())
    assert "..." not in narrow
    assert "…" not in narrow

    wide = _preview(plan, step, tmp_path, width=400)
    lines = [line.strip() for line in wide.splitlines() if line.strip()]
    assert "1  op: filter" in lines
    assert f"input: {filename}" in lines
    assert f"details: where=['status', '==', '{status}']" in lines
    assert f"output: {output.name}" in lines
    assert "'field':" not in wide


def test_filter_where_reads_as_a_condition():
    assert _detail("where", [{"field": "sector", "op": "==", "value": "tourism"}]) == (
        "['sector', '==', 'tourism']"
    )
    assert _detail("where", [
        {"field": "sector", "op": "==", "value": "tourism"},
        {"field": "beds", "op": ">=", "value": "10"},
    ]) == "[['sector', '==', 'tourism'], ['beds', '>=', '10']]"
    assert _detail("where", [{"field": "name", "op": "is_null", "value": None}]) == (
        "['name', 'is_null']"
    )
    assert _detail("input", "roads.gpkg") == "roads.gpkg"


def _preview(plan, step, cwd, width: int) -> str:
    buf = StringIO()
    console = Console(file=buf, width=width, force_terminal=False)
    Session(backend=None, cwd=cwd, console=console).preview(plan, [step])
    return buf.getvalue()
