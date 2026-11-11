class SimulatorException(Exception):
    """Base exception for Digital Twin Client Simulator."""
    pass

class ProfileNotFoundException(SimulatorException):
    """Raised when a requested customer profile is not found."""
    pass

class SimulationNotFoundException(SimulatorException):
    """Raised when a requested simulation session is not found."""
    pass

class LLMProviderException(SimulatorException):
    """Raised when the LLM provider (Groq) or voice transcription fails."""
    pass
