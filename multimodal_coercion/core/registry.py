from pathlib import Path


class ModelRegistry:
    def __init__(self, models_config, base_path):
        self.cfg = models_config
        self.base = Path(base_path)

    def path(self, key):
        entry = self.cfg.get(key, {})
        p = entry.get("path")
        if not p:
            return None
        return (self.base / Path(p)).resolve()

    def value(self, key, field):
        entry = self.cfg.get(key, {})
        return entry.get(field)

