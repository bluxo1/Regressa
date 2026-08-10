"""Run orchestration: expand cases × targets, call providers, evaluate assertions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from regressa import capabilities, templating
from regressa.assertions import EvalContext, evaluate
from regressa.config import Case, Suite, Target
from regressa.providers import CompletionRequest, CompletionResponse, ProviderError, get_provider
from regressa.providers.base import Provider
from regressa.results import CaseResult, RunSummary, Usage
from regressa.storage import Store, git_sha, new_run_id


class ConfigError(Exception):
    """A suite is internally valid but cannot be executed as written."""


@dataclass
class RunOptions:
    concurrency: int = 4
    use_cache: bool = True
    targets: list[str] | None = None
    tags: list[str] | None = None
    cases: list[str] | None = None
    label: str | None = None
    # Called with each CaseResult as it completes, for progress display.
    on_result: Callable[[CaseResult], None] | None = field(default=None, repr=False)


@dataclass
class _Task:
    case: Case
    target: Target
    prompt: str
    variables: dict[str, Any]
    assertions: list[Any]


class Runner:
    def __init__(self, store: Store | None = None) -> None:
        self.store = store
        self._providers: dict[str, Provider] = {}

    def provider_for(self, name: str) -> Provider:
        if name not in self._providers:
            self._providers[name] = get_provider(name)
        return self._providers[name]

    # ------------------------------------------------------------------ #

    def run(self, suite: Suite, options: RunOptions | None = None) -> RunSummary:
        opts = options or RunOptions()

        # Fail before spending anything: a target that sets `temperature` on a
        # model which rejects it would 400 on every single case.
        issues = suite.validate_targets()
        if issues:
            raise ConfigError(
                "suite targets are not callable as configured:\n"
                + "\n".join(f"  [{i.target_id}] {i.message}" for i in issues)
            )

        tasks = list(self._plan(suite, opts))
        if not tasks:
            raise ConfigError(
                "no cases matched the given --case / --tag / --target filters"
            )

        run = RunSummary(
            run_id=new_run_id(),
            suite=suite.name,
            suite_hash=suite.fingerprint(),
            label=opts.label,
            git_sha=git_sha(),
        )

        workers = max(1, opts.concurrency)
        if workers == 1:
            for task in tasks:
                result = self._execute(task, opts)
                run.results.append(result)
                if opts.on_result:
                    opts.on_result(result)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._execute, task, opts): task for task in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    run.results.append(result)
                    if opts.on_result:
                        opts.on_result(result)

        # Concurrency scrambles completion order; sort so reports and diffs are stable.
        run.results.sort(key=lambda r: (r.case_id, r.target_id))
        run.finished_at = datetime.now(timezone.utc)
        run.recompute()

        if self.store is not None:
            self.store.save_run(run)
        return run

    # ------------------------------------------------------------------ #

    def _plan(self, suite: Suite, opts: RunOptions) -> Iterable[_Task]:
        target_filter = set(opts.targets) if opts.targets else None
        tag_filter = set(opts.tags) if opts.tags else None
        case_filter = set(opts.cases) if opts.cases else None

        for case in suite.cases:
            if case_filter and case.id not in case_filter:
                continue
            if tag_filter and not (tag_filter & set(case.tags)):
                continue

            variables = {**suite.vars, **case.vars}
            try:
                prompt = templating.render(case.prompt, variables)
            except templating.MissingVariableError as exc:
                raise ConfigError(f"case {case.id!r}: {exc}") from exc

            assertions = suite.resolved_assertions(case)
            for target in suite.resolved_targets(case):
                if target_filter and target.id not in target_filter:
                    continue
                yield _Task(
                    case=case,
                    target=target,
                    prompt=prompt,
                    variables=variables,
                    assertions=assertions,
                )

    def _execute(self, task: _Task, opts: RunOptions) -> CaseResult:
        target = task.target
        request = CompletionRequest(
            model=target.model,
            prompt=task.prompt,
            max_tokens=target.max_tokens,
            system=target.system,
            effort=target.effort,
            thinking=target.thinking,
            stop_sequences=tuple(target.stop_sequences) if target.stop_sequences else None,
            temperature=target.temperature,
            top_p=target.top_p,
            options=target.options,
            case_id=task.case.id,
        )

        result = CaseResult(
            case_id=task.case.id,
            target_id=target.id,
            passed=False,
            tags=list(task.case.tags),
        )

        try:
            response = self._complete(request, target.provider, opts.use_cache)
        except ProviderError as exc:
            # Errored, not failed: a network blip must never be recorded as a
            # prompt regression. `regressa diff` treats these separately.
            result.error = str(exc)
            return result

        cost = self._cost(target.model, response)
        result.output = response.text
        result.latency_ms = response.latency_ms
        result.cached = response.cached
        result.stop_reason = response.stop_reason
        result.usage = Usage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
            cost_usd=cost,
        )

        ctx = EvalContext(
            output=response.text,
            response=response,
            cost_usd=cost,
            case_id=task.case.id,
            target_id=target.id,
            prompt=task.prompt,
            vars=task.variables,
        )

        outcomes = [evaluate(spec, ctx) for spec in task.assertions]
        result.assertions = outcomes

        # Grading spend belongs on the ledger, not hidden inside the assertion.
        judge_cost = sum(
            float(o.detail.get("judge_cost_usd", 0.0) or 0.0) for o in outcomes
        )
        result.usage.cost_usd = cost + judge_cost

        result.passed = all(o.passed for o in outcomes) if outcomes else True
        result.score = _weighted_score(outcomes)
        return result

    def _complete(
        self, request: CompletionRequest, provider_name: str, use_cache: bool
    ) -> CompletionResponse:
        cacheable = use_cache and self.store is not None and provider_name != "mock"
        key = request.cache_key() if cacheable else None

        if key is not None:
            cached = self.store.cache_get(key)  # type: ignore[union-attr]
            if cached is not None:
                response = CompletionResponse.from_dict(cached)
                response.cached = True
                return response

        response = self.provider_for(provider_name).complete(request)

        if key is not None:
            self.store.cache_put(key, response.to_dict())  # type: ignore[union-attr]
        return response

    @staticmethod
    def _cost(model: str, response: CompletionResponse) -> float:
        try:
            return capabilities.price(
                model,
                response.input_tokens,
                response.output_tokens,
                response.cache_read_tokens,
                response.cache_write_tokens,
            )
        except capabilities.UnknownModelError:
            return 0.0


def _weighted_score(outcomes: list[Any]) -> float:
    if not outcomes:
        return 1.0
    total_weight = sum(o.weight for o in outcomes)
    if total_weight <= 0:
        return 0.0
    return sum(o.score * o.weight for o in outcomes) / total_weight
