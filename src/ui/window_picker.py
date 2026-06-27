"""Window picker dialog for selecting capture target."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QComboBox,
    QPushButton, QLabel,
)

from capture.window import list_windows


class WindowPickerDialog(QDialog):
    def __init__(self, parent=None, current_hwnd: int | None = None):
        super().__init__(parent)
        self.setWindowTitle("Select Capture Window")
        self.setModal(True)
        self.setMinimumWidth(500)

        self._selected_hwnd: int | None = None

        layout = QVBoxLayout(self)

        label = QLabel("Choose which window to capture subtitles from:")
        layout.addWidget(label)

        combo_row = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.setMinimumWidth(400)
        combo_row.addWidget(self._combo, stretch=1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_windows)
        combo_row.addWidget(refresh_btn)
        layout.addLayout(combo_row)

        self._load_windows(current_hwnd)

        buttons = QHBoxLayout()
        buttons.addStretch()
        ok_btn = QPushButton("Select")
        ok_btn.clicked.connect(self._accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)

    def _load_windows(self, select_hwnd: int | None = None) -> None:
        windows = list_windows()
        self._combo.clear()
        self._combo.addItem("Full Screen (all windows)", None)
        select_idx = 0
        for i, w in enumerate(windows):
            self._combo.addItem(f"{w['title']}  [{w['width']}x{w['height']}]", w["hwnd"])
            if select_hwnd and w["hwnd"] == select_hwnd:
                select_idx = i + 1
        self._combo.setCurrentIndex(select_idx)

    def _accept(self) -> None:
        self._selected_hwnd = self._combo.currentData()
        self.accept()

    @property
    def selected_hwnd(self) -> int | None:
        return self._selected_hwnd
