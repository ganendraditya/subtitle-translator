from PyQt6.QtWidgets import QDialog, QFormLayout, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QMessageBox, QDoubleSpinBox

LANGUAGES = [
    ("English", "en"),
    ("Indonesian", "id"),
    ("Japanese", "ja"),
    ("Chinese (zh)", "zh"),
    ("Korean", "ko"),
    ("French", "fr"),
    ("German", "de"),
    ("Spanish", "es"),
    ("Arabic", "ar"),
]


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subtitle Translator - Settings")
        self.setModal(True)

        from config.settings import Settings
        self._settings = Settings.get_instance()

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self._src_combo = QComboBox()
        for name, code in LANGUAGES:
            self._src_combo.addItem(f"{name} ({code})", code)
        idx = self._src_combo.findData(self._settings.source_lang)
        if idx >= 0:
            self._src_combo.setCurrentIndex(idx)
        self._src_combo.currentIndexChanged.connect(self._validate)
        form.addRow("Source language:", self._src_combo)

        self._tgt_combo = QComboBox()
        for name, code in LANGUAGES:
            self._tgt_combo.addItem(f"{name} ({code})", code)
        idx = self._tgt_combo.findData(self._settings.target_lang)
        if idx >= 0:
            self._tgt_combo.setCurrentIndex(idx)
        self._tgt_combo.currentIndexChanged.connect(self._validate)
        form.addRow("Target language:", self._tgt_combo)

        crop = self._settings.get("capture_crop", {"left": 0.0, "right": 1.0})
        self._crop_left = QDoubleSpinBox()
        self._crop_left.setRange(0.0, 100.0)
        self._crop_left.setSuffix(" %")
        self._crop_left.setValue(crop["left"] * 100)
        form.addRow("Crop left (%):", self._crop_left)

        self._crop_right = QDoubleSpinBox()
        self._crop_right.setRange(0.0, 100.0)
        self._crop_right.setSuffix(" %")
        self._crop_right.setValue((1.0 - crop["right"]) * 100)
        form.addRow("Crop right (%):", self._crop_right)

        self._device_combo = QComboBox()
        self._device_combo.addItem("Auto (detect GPU)", "auto")
        self._device_combo.addItem("CPU", "cpu")
        self._device_combo.addItem("GPU", "gpu")
        device = self._settings.get("device", "auto")
        idx = self._device_combo.findData(device)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)
        form.addRow("Device:", self._device_combo)

        layout.addLayout(form)

        buttons = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addStretch()
        buttons.addWidget(save_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

        self._validate()

    def _validate(self):
        pass

    def _save(self):
        src = self._src_combo.currentData()
        tgt = self._tgt_combo.currentData()
        if src == tgt:
            QMessageBox.warning(self, "Invalid", "Source and target must be different.")
            return
        self._settings.source_lang = src
        self._settings.target_lang = tgt
        left = self._crop_left.value() / 100.0
        right = 1.0 - self._crop_right.value() / 100.0
        if left < right:
            self._settings.set("capture_crop", {"left": left, "right": right})
        self._settings.set("device", self._device_combo.currentData())
        self.accept()
