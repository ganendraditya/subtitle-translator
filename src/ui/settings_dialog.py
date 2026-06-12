from PyQt6.QtWidgets import QDialog, QFormLayout, QComboBox, QPushButton, QHBoxLayout, QVBoxLayout, QMessageBox

LANGUAGES = [
    ("English", "en"),
    ("Indonesian", "id"),
    ("Japanese", "ja"),
    ("Chinese (Simplified)", "zh"),
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
        self.accept()
