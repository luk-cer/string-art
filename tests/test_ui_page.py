import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "plotter" / "ui_index.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_ui_script_parses(tmp_path):
    """One syntax error kills the whole UI (no controls, no upload), so check it parses."""
    s = PAGE.read_text()
    js = s[s.index("<script>") + len("<script>"):s.rindex("</script>")]
    f = tmp_path / "ui.js"
    f.write_text(js)
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
