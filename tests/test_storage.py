from __future__ import annotations

import pytest

from regressa.runner import Runner, RunOptions
from regressa.storage import Store


def test_run_round_trips_through_sqlite(mock_suite, store):
    run = Runner(store).run(mock_suite)
    loaded = store.get_run(run.run_id)
    assert loaded is not None
    assert loaded.total == run.total
    assert loaded.score == pytest.approx(run.score)
    assert [r.key for r in loaded.results] == [r.key for r in run.results]
    assert loaded.results[0].assertions[0].type


def test_baseline_set_and_get(mock_suite, store):
    run = Runner(store).run(mock_suite)
    store.set_baseline(mock_suite.name, run.run_id)
    baseline = store.get_baseline(mock_suite.name)
    assert baseline is not None
    assert baseline.run_id == run.run_id
    assert store.clear_baseline(mock_suite.name) is True
    assert store.get_baseline(mock_suite.name) is None


def test_resolve_run_accepts_latest_and_offsets(mock_suite, store):
    runner = Runner(store)
    first = runner.run(mock_suite)
    second = runner.run(mock_suite)

    assert store.resolve_run("latest", mock_suite.name).run_id == second.run_id
    assert store.resolve_run("latest~1", mock_suite.name).run_id == first.run_id
    assert store.resolve_run(first.run_id).run_id == first.run_id
    assert store.resolve_run("nope") is None


def test_resolve_baseline_reference(mock_suite, store):
    run = Runner(store).run(mock_suite)
    store.set_baseline(mock_suite.name, run.run_id)
    assert store.resolve_run("baseline", mock_suite.name).run_id == run.run_id


def test_trend_is_oldest_first(mock_suite, store):
    runner = Runner(store)
    first = runner.run(mock_suite)
    second = runner.run(mock_suite)
    points = store.trend(mock_suite.name)
    assert [p["run_id"] for p in points] == [first.run_id, second.run_id]
    assert points[0]["pass_rate"] == pytest.approx(1.0)


def test_cache_avoids_a_second_provider_call(tmp_path):
    """The cache is keyed on request content, so a repeat run is free."""
    from regressa.config import Suite

    store = Store(tmp_path / "c.db")
    calls = {"n": 0}

    suite = Suite.model_validate(
        {
            "name": "cached",
            "targets": [{"id": "t", "provider": "anthropic", "model": "claude-opus-5"}],
            "cases": [
                {"id": "c", "prompt": "p",
                 "assertions": [{"type": "contains", "value": "ok"}]}
            ],
        }
    )

    class StubProvider:
        name = "anthropic"

        def complete(self, request):
            from regressa.providers.base import CompletionResponse

            calls["n"] += 1
            return CompletionResponse(text="ok", input_tokens=5, output_tokens=5)

    runner = Runner(store)
    runner._providers["anthropic"] = StubProvider()

    first = runner.run(suite)
    second = runner.run(suite)

    assert calls["n"] == 1
    assert first.results[0].cached is False
    assert second.results[0].cached is True

    assert store.cache_clear() >= 1
    runner.run(suite)
    assert calls["n"] == 2
    store.close()


def test_no_cache_option_forces_a_fresh_call(tmp_path):
    from regressa.config import Suite
    from regressa.providers.base import CompletionResponse

    store = Store(tmp_path / "n.db")
    calls = {"n": 0}

    class StubProvider:
        name = "anthropic"

        def complete(self, request):
            calls["n"] += 1
            return CompletionResponse(text="ok")

    suite = Suite.model_validate(
        {
            "name": "nocache",
            "targets": [{"id": "t", "provider": "anthropic", "model": "claude-opus-5"}],
            "cases": [{"id": "c", "prompt": "p"}],
        }
    )
    runner = Runner(store)
    runner._providers["anthropic"] = StubProvider()
    runner.run(suite, RunOptions(use_cache=False))
    runner.run(suite, RunOptions(use_cache=False))
    assert calls["n"] == 2
    store.close()


def test_cache_survives_concurrent_writes_from_worker_threads(tmp_path):
    """Regression test for a crash that only appeared on real (non-mock) runs.

    The runner calls cache_get/cache_put from its thread pool. With an unguarded
    connection, concurrent `with self.conn:` blocks interleaved their implicit
    transactions and sqlite3 raised `InterfaceError: bad parameter or other API
    misuse` — so every real run at the default concurrency of 4 with caching on
    would die partway through. Mock targets skip the cache entirely, which is why
    no offline test caught it.
    """
    from regressa.config import Suite
    from regressa.providers.base import CompletionResponse

    class StubProvider:
        name = "anthropic"

        def complete(self, request):
            return CompletionResponse(text="ok", input_tokens=5, output_tokens=5)

    suite = Suite.model_validate(
        {
            "name": "concurrency",
            "targets": [{"id": "t", "provider": "anthropic", "model": "claude-opus-5"}],
            "cases": [
                {
                    "id": f"c{i}",
                    "prompt": f"prompt {i}",
                    "assertions": [{"type": "contains", "value": "ok"}],
                }
                for i in range(40)
            ],
        }
    )

    store = Store(tmp_path / "conc.db")
    runner = Runner(store)
    runner._providers["anthropic"] = StubProvider()

    first = runner.run(suite, RunOptions(concurrency=16))
    assert first.errors == 0, [r.error for r in first.results if r.error][:3]
    assert first.passed == 40

    # Every response should now be cached, and re-running must read all 40 back.
    assert store.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 40
    second = runner.run(suite, RunOptions(concurrency=16))
    assert second.errors == 0
    assert sum(1 for r in second.results if r.cached) == 40
    store.close()


def test_store_methods_are_safe_to_call_from_many_threads(tmp_path):
    """Reads and writes interleaved across threads must not corrupt or raise."""
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path / "threads.db")
    errors: list[Exception] = []

    def hammer(n: int) -> None:
        try:
            for i in range(20):
                store.cache_put(f"k{n}-{i}", {"text": "x", "input_tokens": i})
                store.cache_get(f"k{n}-{i}")
                store.suites()
                store.list_runs(limit=5)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(hammer, range(8)))

    assert errors == [], errors
    assert store.conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0] == 160
    store.close()


def test_deleting_a_run_removes_its_results(mock_suite, store):
    run = Runner(store).run(mock_suite)
    assert store.delete_run(run.run_id) is True
    assert store.get_run(run.run_id) is None
