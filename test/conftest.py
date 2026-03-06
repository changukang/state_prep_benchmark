import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-slow",
        action="store_true",
        default=False,
        help="Skip tests marked as slow.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--no-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Skipped because --no-slow was provided")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
