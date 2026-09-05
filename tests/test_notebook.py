import ast
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["01-t4-build-smoke.ipynb", "02-t4-rmsnorm-correctness.ipynb"])
def test_notebook_cells_are_executable_python_and_outputs_are_clean(name):
    path = Path(__file__).resolve().parents[1] / "notebooks" / name
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
            assert cell["execution_count"] is None and cell["outputs"] == []
