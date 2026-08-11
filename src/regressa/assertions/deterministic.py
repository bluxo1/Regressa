"""Deterministic assertions — the regression backbone.

These are free, fast, and never flake, so they carry the load. Judge and
similarity assertions layer on top for outputs whose correctness genuinely can't
be expressed as a string or schema check.
"""

from __future__ import annotations

import json
import re
from typing import Any

from regressa.assertions.base import EvalContext, evaluator, outcome


def _norm(text: str, ignore_case: bool) -> str:
    return text.lower() if ignore_case else text


def _as_list(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


@evaluator("contains")
def _contains(spec: Any, ctx: EvalContext):
    haystack = _norm(ctx.output, spec.ignore_case)
    needles = _as_list(spec.value)
    hits = [n for n in needles if _norm(n, spec.ignore_case) in haystack]
    misses = [n for n in needles if n not in hits]

    if spec.mode == "any":
        passed = bool(hits)
        # Partial credit is meaningful for `any`: it shows how close the output got.
        score = 1.0 if passed else 0.0
        msg = (
            f"found {hits[0]!r}"
            if passed
            else f"none of {needles!r} present in output"
        )
    else:
        passed = not misses
        score = len(hits) / len(needles) if needles else 1.0
        msg = "all substrings present" if passed else f"missing {misses!r}"

    return outcome(spec, passed=passed, score=score, message=msg,
                   detail={"hits": hits, "misses": misses})


@evaluator("not_contains")
def _not_contains(spec: Any, ctx: EvalContext):
    haystack = _norm(ctx.output, spec.ignore_case)
    banned = [n for n in _as_list(spec.value) if _norm(n, spec.ignore_case) in haystack]
    return outcome(
        spec,
        passed=not banned,
        message="no banned substrings present" if not banned else f"found banned {banned!r}",
        detail={"found": banned},
    )


@evaluator("equals")
def _equals(spec: Any, ctx: EvalContext):
    actual, expected = ctx.output, spec.value
    if spec.strip:
        actual, expected = actual.strip(), expected.strip()
    if spec.ignore_case:
        actual, expected = actual.lower(), expected.lower()
    passed = actual == expected
    return outcome(
        spec,
        passed=passed,
        message="exact match" if passed else f"expected {expected!r}, got {actual[:200]!r}",
    )


@evaluator("regex")
def _regex(spec: Any, ctx: EvalContext):
    flags = 0
    if spec.ignore_case:
        flags |= re.IGNORECASE
    if spec.dotall:
        flags |= re.DOTALL
    match = re.search(spec.pattern, ctx.output, flags)
    matched = match is not None
    passed = matched is spec.expect
    if spec.expect:
        msg = f"matched {match.group(0)[:120]!r}" if match else f"pattern {spec.pattern!r} did not match"
    else:
        msg = "pattern absent as required" if not match else f"pattern {spec.pattern!r} matched but should not"
    return outcome(spec, passed=passed, message=msg,
                   detail={"match": match.group(0) if match else None})


@evaluator("starts_with")
def _starts_with(spec: Any, ctx: EvalContext):
    text = _norm(ctx.output.lstrip(), spec.ignore_case)
    passed = text.startswith(_norm(spec.value, spec.ignore_case))
    return outcome(spec, passed=passed,
                   message="prefix matched" if passed else f"output does not start with {spec.value!r}")


@evaluator("ends_with")
def _ends_with(spec: Any, ctx: EvalContext):
    text = _norm(ctx.output.rstrip(), spec.ignore_case)
    passed = text.endswith(_norm(spec.value, spec.ignore_case))
    return outcome(spec, passed=passed,
                   message="suffix matched" if passed else f"output does not end with {spec.value!r}")


def _parse_json(text: str) -> tuple[Any | None, str | None]:
    """Parse JSON, tolerating a fenced code block around it."""
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, str(exc)


@evaluator("json_valid")
def _json_valid(spec: Any, ctx: EvalContext):
    parsed, err = _parse_json(ctx.output)
    return outcome(
        spec,
        passed=err is None,
        message="output is valid JSON" if err is None else f"invalid JSON: {err}",
        detail={"type": type(parsed).__name__ if err is None else None},
    )


@evaluator("json_schema")
def _json_schema(spec: Any, ctx: EvalContext):
    parsed, err = _parse_json(ctx.output)
    if err is not None:
        return outcome(spec, passed=False, message=f"invalid JSON: {err}")

    import jsonschema

    validator = jsonschema.Draft202012Validator(spec.schema_)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.path))
    if not errors:
        return outcome(spec, passed=True, message="output matches schema")

    described = [
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:5]
    ]
    return outcome(
        spec,
        passed=False,
        # Partial credit by error count keeps the trend chart informative while a
        # schema is being converged on, instead of a flat zero the whole way.
        score=max(0.0, 1.0 - len(errors) / 10),
        message=f"{len(errors)} schema violation(s): " + "; ".join(described),
        detail={"errors": described},
    )


_MISSING = object()


@evaluator("json_path")
def _json_path(spec: Any, ctx: EvalContext):
    parsed, err = _parse_json(ctx.output)
    if err is not None:
        return outcome(spec, passed=False, message=f"invalid JSON: {err}")

    value = _dig(parsed, spec.path)
    found = value is not _MISSING

    if not spec.exists:
        return outcome(spec, passed=not found,
                       message=f"{spec.path} absent as required" if not found
                       else f"{spec.path} unexpectedly present")

    if not found:
        return outcome(spec, passed=False, message=f"{spec.path} not found in output")

    if spec.equals is None:
        return outcome(spec, passed=True, message=f"{spec.path} present")

    passed = value == spec.equals
    return outcome(
        spec,
        passed=passed,
        message=f"{spec.path} == {spec.equals!r}" if passed
        else f"{spec.path}: expected {spec.equals!r}, got {value!r}",
        detail={"actual": value},
    )


def _dig(data: Any, dotted: str) -> Any:
    current = data
    for part in dotted.split("."):
        if isinstance(current, dict):
            if part not in current:
                return _MISSING
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        else:
            return _MISSING
    return current


def _count_sentences(text: str) -> int:
    """Count sentences, tolerating the abbreviations that show up in support prose.

    Deliberately simple: this backs a "three sentences maximum" style prompt rule,
    where the interesting signal is 3 vs 8, not 3 vs 4. Protecting common
    abbreviations keeps "e.g." and "Dr." from inflating the count.
    """
    protected = text
    for abbrev in ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Mrs.", "Ms.", "Dr.", "No."):
        protected = protected.replace(abbrev, abbrev.replace(".", "\0"))
    parts = re.split(r"[.!?]+(?=\s|$)", protected)
    return len([p for p in parts if p.strip()])


@evaluator("length")
def _length(spec: Any, ctx: EvalContext):
    if spec.unit == "words":
        size = len(ctx.output.split())
    elif spec.unit == "lines":
        size = len(ctx.output.splitlines())
    elif spec.unit == "sentences":
        size = _count_sentences(ctx.output)
    else:
        size = len(ctx.output)

    problems = []
    if spec.min is not None and size < spec.min:
        problems.append(f"{size} {spec.unit} < min {spec.min}")
    if spec.max is not None and size > spec.max:
        problems.append(f"{size} {spec.unit} > max {spec.max}")

    return outcome(
        spec,
        passed=not problems,
        message=f"{size} {spec.unit}" if not problems else "; ".join(problems),
        detail={"size": size, "unit": spec.unit},
    )


@evaluator("latency")
def _latency(spec: Any, ctx: EvalContext):
    # A cached response says nothing about real latency, so don't pretend it does.
    if ctx.response.cached:
        return outcome(spec, passed=True, message="skipped (cached response)")
    actual = ctx.response.latency_ms
    passed = actual <= spec.max_ms
    return outcome(
        spec,
        passed=passed,
        message=f"{actual:.0f}ms" + ("" if passed else f" > budget {spec.max_ms:.0f}ms"),
        detail={"latency_ms": actual},
    )


@evaluator("cost")
def _cost(spec: Any, ctx: EvalContext):
    passed = ctx.cost_usd <= spec.max_usd
    return outcome(
        spec,
        passed=passed,
        message=f"${ctx.cost_usd:.6f}" + ("" if passed else f" > budget ${spec.max_usd:.6f}"),
        detail={"cost_usd": ctx.cost_usd},
    )


@evaluator("tokens")
def _tokens(spec: Any, ctx: EvalContext):
    problems = []
    if spec.max_output is not None and ctx.response.output_tokens > spec.max_output:
        problems.append(f"output {ctx.response.output_tokens} > {spec.max_output}")
    if spec.max_input is not None and ctx.response.input_tokens > spec.max_input:
        problems.append(f"input {ctx.response.input_tokens} > {spec.max_input}")
    return outcome(
        spec,
        passed=not problems,
        message=(
            f"in={ctx.response.input_tokens} out={ctx.response.output_tokens}"
            if not problems
            else "; ".join(problems)
        ),
    )
