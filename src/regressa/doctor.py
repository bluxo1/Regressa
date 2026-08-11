"""Verify the Anthropic integration against the live API.

Everything else in Regressa is tested offline with stub clients, which is fast and
free but cannot tell you whether the requests it builds are actually accepted or
whether the responses it parses look like real ones. This module makes a handful
of deliberately tiny calls to answer that.

It costs a fraction of a cent (the default model is the cheapest one), and one of
the checks costs nothing at all: a request the API is expected to *reject*, which
validates the capability table that the rest of the tool relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from regossa_compat import *  # type: ignore  # noqa: F401,F403  # placeholder
