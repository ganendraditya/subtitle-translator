import json
import os

DEFAULT = {
    "source_lang": "en",
    "target_lang": "id",
    "crop_bottom": 0.15,
    "conf_thresh": 0.7,
    "capture_crop": {"left": 0.0, "right": 1.0},
    "yolo_model": "models/yolov8s-subtitle.pt",
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")


class Settings:
    _instance = None

    def __init__(self):
        self._data = dict(DEFAULT)
        self.load()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        try:
            with open(CONFIG_FILE) as f:
                self._data.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self._data, f, indent=2)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value
        self.save()

    @property
    def source_lang(self):
        return self._data["source_lang"]

    @source_lang.setter
    def source_lang(self, v):
        self._data["source_lang"] = v
        self.save()

    @property
    def target_lang(self):
        return self._data["target_lang"]

    @target_lang.setter
    def target_lang(self, v):
        self._data["target_lang"] = v
        self.save()
