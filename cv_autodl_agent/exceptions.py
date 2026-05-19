class InputValidationError(ValueError):
    """Raised when upstream inputs do not satisfy the contract."""


class CandidateSelectionError(RuntimeError):
    """Raised when no valid candidate can be selected."""


class WorkflowExecutionError(RuntimeError):
    """Raised when the workflow cannot produce a valid deliverable."""
