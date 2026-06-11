from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont, QColor, QPainter, QFontMetrics
from PyQt6.QtWidgets import QApplication, QWidget
import numpy as np
from typing import Optional


class OverlayWindow(QWidget):
    """Transparent, always-on-top, click-through overlay window."""

    def __init__(self) -> None:
        super().__init__()
        self._init_ui()
        self._text: str = ""
        self._font_size: int = 24
        self._opacity: float = 0.9
        self._position: str = "bottom_center"

    def _init_ui(self) -> None:
        """Initialize window properties."""
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput  # click-through
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        # Full screen size for positioning flexibility
        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
        
        self.show()

    def set_text(self, text: str) -> None:
        """Update displayed text."""
        self._text = text
        self.update()

    def set_font_size(self, size: int) -> None:
        """Set font size."""
        self._font_size = size
        self.update()

    def set_opacity(self, opacity: float) -> None:
        """Set background opacity (0.0-1.0)."""
        self._opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_position(self, position: str) -> None:
        """Set text position: top, top_center, middle, bottom_center, bottom."""
        if position in ("top", "top_center", "middle", "bottom_center", "bottom"):
            self._position = position
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the overlay text (multi-line supported)."""
        if not self._text:
            return

        lines = self._text.split("\n")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", self._font_size, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        line_height = metrics.height()
        padding_y = 10
        total_height = len(lines) * line_height + padding_y * 2
        max_line_width = max(metrics.boundingRect(l).width() for l in lines) if lines else 0

        x, y = self._calculate_position(max_line_width, total_height)

        # Semi-transparent background
        bg_color = QColor(0, 0, 0)
        bg_color.setAlphaF(self._opacity * 0.6)
        painter.fillRect(
            x - 20, y - padding_y,
            max_line_width + 40, total_height,
            bg_color
        )

        # Draw each line with outline
        for i, line in enumerate(lines):
            line_y = y + i * line_height + metrics.ascent()
            painter.setPen(QColor(0, 0, 0))
            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
                painter.drawText(x + dx, line_y + dy, line)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x, line_y, line)

        painter.end()

    def _calculate_position(self, text_width: int, text_height: int) -> tuple[int, int]:
        """Calculate (x, y) position for text based on self._position."""
        screen_rect = self.geometry()
        sw, sh = screen_rect.width(), screen_rect.height()

        positions = {
            "top": (sw // 2 - text_width // 2, 80),
            "top_center": (sw // 2 - text_width // 2, sh // 4 - text_height // 2),
            "middle": (sw // 2 - text_width // 2, sh // 2 - text_height // 2),
            "bottom_center": (sw // 2 - text_width // 2, sh * 3 // 4 - text_height // 2),
            "bottom": (sw // 2 - text_width // 2, sh - 120),
        }
        return positions.get(self._position, positions["bottom_center"])
