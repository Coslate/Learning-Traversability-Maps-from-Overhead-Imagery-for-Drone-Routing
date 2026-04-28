from __future__ import annotations

import loveda_project


def test_package_imports_with_version() -> None:
    assert isinstance(loveda_project.__version__, str)
    assert loveda_project.__version__
