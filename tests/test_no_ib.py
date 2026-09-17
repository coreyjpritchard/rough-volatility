"""CI-has-no-IB invariant: importing the maths/plotting modules must not pull
in roughvol.ibkr or ib_async. Not parametrised over impl — this is about
roughvol specifically, the package CI actually ships.
"""

import subprocess
import sys

_CHECK = (
    "import roughvol.paths, roughvol.variation, roughvol.plotting; "
    "import sys; "
    "assert 'roughvol.ibkr' not in sys.modules, 'roughvol.ibkr' in sys.modules; "
    "assert 'ib_async' not in sys.modules, 'ib_async' in sys.modules"
)


def test_no_ib_import():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
