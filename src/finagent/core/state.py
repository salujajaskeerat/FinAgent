"""Explicit bounded state transitions for one analysis run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AnalysisState(StrEnum):
    """Application states used for tracing and invariant checks."""

    RECEIVED = "received"
    RESOLVING_SCOPE = "resolving_scope"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    CALCULATING = "calculating"
    SYNTHESIZING = "synthesizing"
    VALIDATING = "validating"
    REPAIRING = "repairing"
    COMPLETED = "completed"


_ALLOWED: dict[AnalysisState, set[AnalysisState]] = {
    AnalysisState.RECEIVED: {AnalysisState.RESOLVING_SCOPE},
    AnalysisState.RESOLVING_SCOPE: {AnalysisState.PLANNING, AnalysisState.COMPLETED},
    AnalysisState.PLANNING: {AnalysisState.RETRIEVING},
    AnalysisState.RETRIEVING: {AnalysisState.CALCULATING, AnalysisState.COMPLETED},
    AnalysisState.CALCULATING: {AnalysisState.SYNTHESIZING},
    AnalysisState.SYNTHESIZING: {AnalysisState.VALIDATING},
    AnalysisState.VALIDATING: {AnalysisState.REPAIRING, AnalysisState.COMPLETED},
    AnalysisState.REPAIRING: {AnalysisState.VALIDATING},
    AnalysisState.COMPLETED: set(),
}


@dataclass(slots=True)
class StateTrace:
    """Validated sequence of states for one request."""

    current: AnalysisState = AnalysisState.RECEIVED
    history: list[AnalysisState] = field(
        default_factory=lambda: [AnalysisState.RECEIVED]
    )

    def move(self, target: AnalysisState) -> None:
        """Advance to an allowed state.

        Parameters
        ----------
        target
            Next workflow state.

        Raises
        ------
        RuntimeError
            If the requested transition is invalid.
        """
        if target not in _ALLOWED[self.current]:
            raise RuntimeError(f"invalid state transition: {self.current} -> {target}")
        self.current = target
        self.history.append(target)
