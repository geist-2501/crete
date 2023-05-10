class AgentNotFound(Exception):
    """Raised when an agent ID is requested but not registered."""


class TalfileLoadError(Exception):
    """Raised when an error is encountered loading a Tal file."""


class WrapperNotFound(Exception):
    """Raised when a wrapper ID is requested but not registered"""


class ConfigNotFound(Exception):
    """Raised when a requested config isn't found."""


class ProfilePropertyNotFound(Exception):
    """Raised when a required property is missing in a profile config."""
