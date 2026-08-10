from __future__ import annotations

import pytest

from regressa.config import Suite
from regressa.storage import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def mock_suite() -> Suite:
    return Suite.model_validate(
        {
            "name": "unit",
            "targets": [
                {
                    "id": "m",
                    "provider": "mock",
                    "model": "mock-1",
                    "options": {
                        "by_case": {
                            "hello": "Hello there, friend.",
                            "json": '{"a": 1, "b": "two"}',
                        }
                    },
                }
            ],
            "cases": [
                {
                    "id": "hello",
                    "prompt": "say hi",
                    "assertions": [{"type": "contains", "value": "hello"}],
                },
                {
                    "id": "json",
                    "prompt": "give json",
                    "assertions": [
                        {"type": "json_valid"},
                        {"type": "json_path", "path": "a", "equals": 1},
                    ],
                },
            ],
        }
    )
