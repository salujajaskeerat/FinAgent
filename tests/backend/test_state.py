"""Finite-state transition tests."""

import pytest

from finagent.core.state import AnalysisState, StateTrace


def test_state_trace_rejects_skipping_the_scope_boundary() -> None:
    trace = StateTrace()

    with pytest.raises(RuntimeError, match="invalid state transition"):
        trace.move(AnalysisState.SYNTHESIZING)
