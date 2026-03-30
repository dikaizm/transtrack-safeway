from pipelines.base import BasePipeline

_REGISTRY: dict[str, type[BasePipeline]] = {}


def register_pipeline(name: str):
    """Decorator to register a pipeline class under a string key."""
    def decorator(cls: type[BasePipeline]):
        if name in _REGISTRY:
            raise ValueError(f"Pipeline '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_pipeline(name: str, config: dict) -> BasePipeline:
    """
    Instantiate a registered pipeline by name.

    Args:
        name: Registered pipeline name (e.g. "yolo").
        config: Merged config dict.

    Returns:
        Instantiated BasePipeline subclass.
    """
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown pipeline '{name}'. Available pipelines: {available}"
        )
    return _REGISTRY[name](config)


def list_pipelines() -> list[str]:
    return list(_REGISTRY.keys())
