from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from observers.stores.datasets import DatasetsStore


@pytest.fixture(autouse=True)
def mock_store(monkeypatch):
    """Mock the datasets store for all tests"""

    async def mock_add_async(*args, **kwargs):
        return None

    async def mock_close_async(*args, **kwargs):
        return None

    def mock_add(*args, **kwargs):
        return None

    def mock_close(*args, **kwargs):
        return None

    store_mock = create_autospec(DatasetsStore, spec_set=False, instance=True)
    store_mock.add_async = AsyncMock(side_effect=mock_add_async)
    store_mock.close_async = AsyncMock(side_effect=mock_close_async)
    store_mock.add = MagicMock(side_effect=mock_add)
    store_mock.close = MagicMock(side_effect=mock_close)

    def mock_connect(*args, **kwargs):
        return store_mock

    # Patch both the class and the connect method
    monkeypatch.setattr("observers.stores.datasets.DatasetsStore.connect", mock_connect)
    monkeypatch.setattr(
        "observers.stores.datasets.DatasetsStore", lambda *args, **kwargs: store_mock
    )

    return store_mock
