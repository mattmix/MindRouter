"""AST guard: no f-string / .format() / concatenation SQL in the db layer.

Snyk flagged a SQL-injection pattern where an IN() list was built by joining
values into an f-string passed to sqlalchemy.text() (crud.py). The fix switched
both sites to expanding bind parameters. This guard keeps the db layer free of
string-built SQL so the pattern cannot silently return.

Identifiers (e.g. table names) genuinely cannot be bound; those live in
services/retention.py behind an explicit allowlist assertion, out of scope here
— this guard is scoped to backend/app/db/ where the finding was.
"""

import ast
import pathlib

DB_DIR = pathlib.Path(__file__).resolve().parents[2] / "db"


def _is_string_built_text(node: ast.AST) -> bool:
    """True if node is text(<f-string with interpolation | concat | .format>)."""
    if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "text"):
        return False
    if not node.args:
        return False
    arg = node.args[0]
    # f-string that actually interpolates a value (a pure-literal f-string is safe)
    if isinstance(arg, ast.JoinedStr):
        return any(isinstance(v, ast.FormattedValue) for v in arg.values)
    # "...".format(...) or var.format(...)
    if (
        isinstance(arg, ast.Call)
        and isinstance(arg.func, ast.Attribute)
        and arg.func.attr == "format"
    ):
        return True
    # string + variable concatenation
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
        return True
    return False


def test_no_string_built_sql_in_db_layer():
    offenders = []
    for path in sorted(DB_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _is_string_built_text(node):
                offenders.append(f"{path.relative_to(DB_DIR.parents[1])}:{node.lineno}")
    assert not offenders, (
        "SQL text() built from an f-string / .format() / concatenation found in "
        "the db layer — use an expanding bind parameter instead:\n  "
        + "\n  ".join(offenders)
    )
