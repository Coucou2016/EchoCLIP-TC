"""Config loading with environment-variable + ``~`` expansion (P0-8).

``configs/echonet_dynamic.yaml`` uses shell-style placeholders such as
``${ECHONET_ROOT}``. Plain ``yaml.safe_load`` never expanded them, so paths
silently stayed literal. :func:`load_yaml_config` now expands recursively over
every ``str``/``dict``/``list`` value.

Expansion behavior
------------------
* strings: ``os.path.expandvars`` then ``os.path.expanduser``
* dicts: keys are left as-is, values expanded recursively
* lists/tuples: elements expanded recursively
* unset variables: left as-is **unless** ``strict=True`` (then raise)

Escaping ``$`` uses the POSIX-ish ``$$`` convention (``expandvars`` does not
honor ``$`` escapes, so a literal ``$`` is written as ``$$`` and collapsed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

PathLike = Union[str, Path]

#: Marker applied by :func:`load_yaml_config` so callers can record provenance.
CONFIG_ENV_EXPANDED_KEY = "_config_env_expanded"


def _expand_str(value: str, *, strict: bool) -> str:
    if strict:
        expanded = os.path.expandvars(value)
        if expanded == value and "$" in value:
            # Only complain about ``${NAME}``-style tokens that look unset.
            import re

            for m in re.finditer(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value):
                if m.group(1) not in os.environ:
                    raise KeyError(
                        f"Environment variable {m.group(1)!r} referenced by config "
                        "is not set (strict mode)."
                    )
    else:
        expanded = os.path.expandvars(value)
    # Collapse ``$$`` → ``$`` so literal dollars can be escaped.
    expanded = expanded.replace("$$", "$")
    return os.path.expanduser(expanded)


def expand_config(value: Any, *, strict: bool = False) -> Any:
    """Recursively expand env vars + ``~`` in a config tree."""
    if isinstance(value, str):
        return _expand_str(value, strict=strict)
    if isinstance(value, Mapping):
        return {k: expand_config(v, strict=strict) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [expand_config(v, strict=strict) for v in value]
    return value


def load_yaml_config(
    path: PathLike,
    *,
    strict: bool = False,
    record_provenance: bool = False,
) -> Dict[str, Any]:
    """Load a YAML config with recursive env/``~`` expansion.

    Parameters
    ----------
    path:
        YAML file to read.
    strict:
        Raise ``KeyError`` when a ``${VAR}`` placeholder has no value.
    record_provenance:
        Add ``_config_env_expanded: True`` plus ``_config_env_source`` to the
        returned dict so downstream manifests/metrics can record that expansion
        happened.

    Empty files return ``{}`` (matching the previous ``or {}`` behavior).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, Mapping):
        raise TypeError(
            f"Config {path} must be a YAML mapping, got {type(raw).__name__}"
        )
    expanded = expand_config(dict(raw), strict=strict)
    if record_provenance:
        expanded[CONFIG_ENV_EXPANDED_KEY] = True
        expanded["_config_env_source"] = str(path)
    return expanded


def expand_path_str(value: Optional[str]) -> Optional[str]:
    """Convenience for single optional path strings."""
    if value is None:
        return None
    return _expand_str(str(value), strict=False)


__all__ = [
    "CONFIG_ENV_EXPANDED_KEY",
    "expand_config",
    "expand_path_str",
    "load_yaml_config",
]
