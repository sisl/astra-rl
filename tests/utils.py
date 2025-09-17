import pytest

mark_gpu = pytest.mark.skipif("not config.getoption('gpu')")
