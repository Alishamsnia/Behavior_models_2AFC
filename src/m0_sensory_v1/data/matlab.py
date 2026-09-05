"""MATLAB ``SessionData`` loading for the behavioral task.

This module preserves the recursive conversion strategy used by the previous
``Module.Reader.load_mat`` implementation, but raises informative exceptions
instead of printing an error and returning ``None``.  Failing loudly prevents
a malformed session from silently disappearing from a pooled analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from scipy.io.matlab import mat_struct


def _convert_matlab_value(value: Any) -> Any:
    """Recursively convert a SciPy MATLAB value into Python containers.

    Purpose
    -------
    Convert nested MATLAB structs and arrays into the dictionary/list layout
    expected by the legacy behavioral extraction logic.

    Parameters
    ----------
    value:
        A value returned by :func:`scipy.io.loadmat` with ``squeeze_me=True``.

    Returns
    -------
    Any
        Nested dictionaries/lists with scalar numeric values left unchanged.

    Important edge cases
    --------------------
    ``squeeze_me=True`` can turn a one-element array into a scalar.  This
    function therefore handles structs, arrays, and scalars independently.

    Related tests
    -------------
    Integration is checked by loading real task sessions during the data audit.
    """

    if isinstance(value, mat_struct):
        return {
            field: _convert_matlab_value(getattr(value, field))
            for field in value._fieldnames
        }
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _convert_matlab_value(value.item())
        return [_convert_matlab_value(element) for element in value]
    return value


def load_session_mat(path: str | Path) -> dict[str, Any]:
    """Load the ``SessionData`` object from one behavioral MATLAB file.

    Purpose
    -------
    Reproduce the previous model's MATLAB reading behavior while making file
    and schema failures explicit.

    Parameters
    ----------
    path:
        Path to a ``.mat`` behavioral session.

    Returns
    -------
    dict
        Recursively converted contents of the top-level ``SessionData`` field.

    Constraints
    -----------
    The source file must exist and contain a MATLAB variable named
    ``SessionData`` whose converted representation is a dictionary.

    Numerical stability
    -------------------
    No numeric transformation is performed here. Duration units are validated
    after trial extraction.

    Causal restrictions
    -------------------
    This function exposes recorded task data only. Predictive code must decide
    separately which fields are causally available before a choice.

    Important edge cases
    --------------------
    Unlike the legacy reader, this function never returns ``None`` on failure.

    Related tests
    -------------
    Real-file smoke tests exercise this loader without modifying the raw file.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Behavioral MATLAB file does not exist: {source}")

    loaded = sio.loadmat(source, struct_as_record=False, squeeze_me=True)
    if "SessionData" not in loaded:
        raise KeyError(f"'SessionData' field not found in {source}")

    session = _convert_matlab_value(loaded["SessionData"])
    if not isinstance(session, dict):
        raise TypeError(
            f"Expected SessionData to convert to dict, got {type(session).__name__}"
        )
    return session
