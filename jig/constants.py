"""Configuration constants and defaults."""

from pathlib import Path

# API
API_KEY = "lm-studio"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234
COMMON_PORTS = [1234, 4321, 8080]
DEFAULT_TIMEOUT = 3.0

# Ollama
OLLAMA_DEFAULT_PORT = 11434

# Paths
PAIRINGS_DIR = Path("pairings")

# Generation defaults
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 4096

# Schema creator defaults
CREATOR_TEMPERATURE = 0.7
