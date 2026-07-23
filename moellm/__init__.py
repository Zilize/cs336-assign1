import importlib.metadata

try:
    __version__ = importlib.metadata.version("moellm")
except importlib.metadata.PackageNotFoundError:
    pass
