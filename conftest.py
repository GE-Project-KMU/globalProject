"""Keep test fixtures and subprocess temporary files inside this workspace."""
from scripts.pytest_workspace import configure

def pytest_configure(config):configure(config)
