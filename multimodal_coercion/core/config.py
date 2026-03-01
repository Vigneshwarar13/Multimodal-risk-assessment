from pathlib import Path
import yaml


def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


from functools import lru_cache


class Config:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.default = load_yaml(self.base_path / "configs" / "default.yaml")
        self.thresholds = load_yaml(self.base_path / "configs" / "thresholds.yaml")
        self.models = load_yaml(self.base_path / "configs" / "models.yaml")


@lru_cache(maxsize=4)
def get_config(base_path: str | None = None) -> "Config":
    """Return a cached :class:`Config` instance for the given project root.

    Calling code should use this helper instead of instantiating ``Config``
    directly.  The object is expensive because it reads YAML files; a
    subsequent call with the same ``base_path`` will return the same
    instance.  ``base_path`` defaults to :func:`project_root` which allows
    callers to avoid passing anything.
    """
    if base_path is None:
        base_path = project_root()
    return Config(base_path)


def project_root():
    return Path(__file__).resolve().parents[1]
