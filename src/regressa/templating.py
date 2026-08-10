"""Minimal `{{ var }}` interpolation for prompts.

Deliberately not Jinja: a prompt regression suite benefits from prompts being
literal text with obvious substitution points. A missing variable is an error —
silently rendering "None" into a prompt would corrupt the eval without failing it.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")


class MissingVariableError(KeyError):
    pass


def variables(template: str) -> set[str]:
    """Every variable name referenced in the template."""
    return {m.group(1) for m in _PATTERN.finditer(template)}


def render(template: str, values: dict[str, Any]) -> str:
    """Substitute `{{ name }}` with `values["name"]`.

    Dotted names index into nested dicts (`{{ user.name }}`).
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        try:
            return str(_lookup(values, name))
        except KeyError as exc:
            available = ", ".join(sorted(values)) or "(none)"
            raise MissingVariableError(
                f"prompt references {{{{ {name} }}}} but no such variable was "
                f"provided. Available: {available}"
            ) from exc

    return _PATTERN.sub(replace, template)


def _lookup(values: dict[str, Any], dotted: str) -> Any:
    current: Any = values
    for part in dotted.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(dotted)
    return current
