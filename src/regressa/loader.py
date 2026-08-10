"""Loading suites from YAML files and directories."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from regressa.config import Suite

SUITE_GLOBS = ("*.yaml", "*.yml")


class SuiteLoadError(Exception):
    """A suite file could not be parsed or failed schema validation."""


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SuiteLoadError(f"{p}: no such file") from exc
    except yaml.YAMLError as exc:
        raise SuiteLoadError(f"{p}: invalid YAML - {exc}") from exc

    if not isinstance(raw, dict):
        raise SuiteLoadError(f"{p}: expected a YAML mapping at the top level")

    try:
        return Suite.model_validate(raw)
    except ValidationError as exc:
        raise SuiteLoadError(f"{p}: {_format_errors(exc)}") from exc


def discover(paths: list[str] | None, root: str | Path = ".") -> list[Path]:
    """Resolve CLI path arguments into a concrete list of suite files.

    A directory expands to every `*.yaml` / `*.yml` inside it (recursively).
    With no paths given, looks in `./suites` then the working directory.
    """
    if not paths:
        default = Path(root) / "suites"
        search = default if default.is_dir() else Path(root)
        return _glob(search)

    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(_glob(p))
        else:
            found.append(p)
    # Preserve order but drop duplicates.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def _glob(directory: Path) -> list[Path]:
    out: list[Path] = []
    for pattern in SUITE_GLOBS:
        out.extend(sorted(directory.rglob(pattern)))
    return out


def _format_errors(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        lines.append(f"  at {loc or '<root>'}: {err['msg']}")
    return "invalid suite:\n" + "\n".join(lines)
