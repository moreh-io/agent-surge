# SPDX-License-Identifier: MIT
"""[audit:misc-core#M8] __main__.py must guard main() under __name__.

Importing ``agentsurge.__main__`` (e.g. via importlib spec, doctest
discovery, or coverage tooling that walks the package) must NOT execute
the CLI parser. The body must be guarded with
``if __name__ == "__main__": main()``.
"""

from __future__ import annotations

import importlib
from pathlib import Path


def test_main_module_does_not_invoke_cli_at_import_time() -> None:
    src = Path(importlib.import_module("agentsurge.__main__").__file__).read_text()
    assert 'if __name__ == "__main__"' in src or "if __name__ == '__main__'" in src, (
        "agentsurge/__main__.py must guard main() with __name__ check; "
        "otherwise importing the module runs the full CLI parser."
    )
