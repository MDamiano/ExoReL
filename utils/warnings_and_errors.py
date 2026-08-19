"""Custom warnings and exceptions used by ExoReL."""


class InvalidVMRCompositionError(ValueError):
    """Raised when active atmospheric gas VMRs do not form a simplex."""
