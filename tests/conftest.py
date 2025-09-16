import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--gpu", action="store_true", dest="gpu", default=False, help="enable GPU tests"
    )


@pytest.fixture()
def cli_app():
    from astra_rl.cli.main import app

    return app
