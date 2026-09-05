"""Application errors mapped by the HTTP adapter."""


class FinagentError(Exception):
    """Base class for expected application failures."""

    code = "FINAGENT_ERROR"


class DependencyUnavailableError(FinagentError):
    """A required model or data dependency is unavailable."""

    code = "DEPENDENCY_UNAVAILABLE"


class AnalysisTimeoutError(FinagentError):
    """The global analysis deadline expired."""

    code = "ANALYSIS_TIMEOUT"


class RateLimitError(FinagentError):
    """An upstream dependency rejected the request for rate limiting."""

    code = "RATE_LIMITED"


class GroundingError(FinagentError):
    """A generated answer failed the bounded grounding repair."""

    code = "GROUNDING_FAILED"
