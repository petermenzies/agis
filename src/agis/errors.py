class AgisError(Exception):
    """An error with a message that is safe and useful to show the user."""


class PlanError(AgisError):
    """The plan produced by the model is invalid for the current files."""

    def __init__(self, problems: list[str], retryable: bool = True):
        self.problems = problems
        # False when asking the model again can't help.
        self.retryable = retryable
        super().__init__("; ".join(problems))


class StepError(AgisError):
    """A step failed while running."""

    def __init__(self, index: int, op: str, cause: Exception):
        self.index = index
        self.op = op
        self.cause = cause
        super().__init__(f"step {index} ({op}) failed: {cause}")


class LLMError(AgisError):
    """The model backend could not be reached or returned unusable output."""
