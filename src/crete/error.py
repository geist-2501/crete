class CreteException(Exception):
    def __init__(self, message: str = None) -> None:
        super().__init__(message)
        self.message = message
        self.name = self.__class__.__name__


class AgentNotFound(CreteException):
    """Raised when an agent ID is requested but not registered."""


class ConcfileLoadError(CreteException):
    """Raised when an error is encountered loading a Tal file."""


class WrapperNotFound(CreteException):
    """Raised when a wrapper ID is requested but not registered"""


class ConfigNotFound(CreteException):
    """Raised when a requested config isn't found."""


class ProfilePropertyNotFound(CreteException):
    """Raised when a required property is missing in a profile config."""

    def __init__(self, prop: str, source_file: str = None) -> None:
        if source_file:
            super().__init__(f"Property {prop} does not exist in `{source_file}`")
        else:
            super().__init__(f"Property {prop} does not exist")
