from pathlib import Path
import yaml


def load_yaml(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Config:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.default = load_yaml(self.base_path / "configs" / "default.yaml")
        self.thresholds = load_yaml(self.base_path / "configs" / "thresholds.yaml")
        self.models = load_yaml(self.base_path / "configs" / "models.yaml")


def project_root():
    return Path(__file__).resolve().parents[1]

