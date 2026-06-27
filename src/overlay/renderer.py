from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor, QPainter, QFontMetrics
from PyQt6.QtWidgets import QApplication, QWidget


class OverlayWindow(QWidget):
    """Transparent, always-on-top, click-through overlay window."""

    def __init__(self) -> None:
        super().__init__()
        self._init_ui()
        self._text: str = ""
        self._font_size: int = 24
        self._opacity: float = 0.9
        self._position: str = "bottom_center"
        self._bbox_cy: float | None = None
        self._smoothed_cy: float | None = None
        self._capture_bounds: tuple[int, int, int, int] | None = None
        self._smooth_alpha = 0.18

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

    def set_capture_bounds(self, bounds: tuple[int, int, int, int] | None) -> None:
        """Set screen-space bounds for the frame that produced the OCR result."""
        if bounds != self._capture_bounds:
            self._capture_bounds = bounds
            self.update()

    def set_bbox_cy(self, cy: float | None) -> None:
        """Set YOLO bbox center Y (0.0-1.0 normalized) for dynamic positioning."""
        if cy is None:
            self._bbox_cy = None
            self._smoothed_cy = None
        elif self._smoothed_cy is None:
            self._bbox_cy = cy
            self._smoothed_cy = cy
        else:
            new_smoothed = self._smooth_alpha * cy + (1 - self._smooth_alpha) * self._smoothed_cy
            changed = abs(new_smoothed - self._bbox_cy) > 0.01
            self._smoothed_cy = new_smoothed
            self._bbox_cy = new_smoothed
            if changed:
                self.update()
            return
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        """Paint the overlay text (multi-line supported)."""
        if not self._text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Segoe UI", self._font_size, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = QFontMetrics(font)

        max_text_width = max(260, min(960, self.width() - 96))
        lines = self._wrap_lines(self._text.split("\n"), metrics, max_text_width)
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

    def _wrap_lines(self, lines: list[str], metrics: QFontMetrics, max_width: int) -> list[str]:
        wrapped: list[str] = []
        for line in lines:
            words = line.split()
            if not words:
                wrapped.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if metrics.boundingRect(candidate).width() <= max_width:
                    current = candidate
                else:
                    wrapped.append(current)
                    current = word
            wrapped.append(current)
        return wrapped

    def _anchor_rect(self) -> tuple[int, int, int, int]:
        """Return local overlay coords for the captured frame bounds."""
        screen_rect = self.geometry()
        sw, sh = screen_rect.width(), screen_rect.height()
        if self._capture_bounds is None:
            return (0, 0, sw, sh)

        left, top, right, bottom = self._capture_bounds
        local_left = max(0, left - screen_rect.x())
        local_top = max(0, top - screen_rect.y())
        local_right = min(sw, right - screen_rect.x())
        local_bottom = min(sh, bottom - screen_rect.y())
        if local_right <= local_left or local_bottom <= local_top:
            return (0, 0, sw, sh)
        return (local_left, local_top, local_right, local_bottom)

    def _clamp_position(self, x: int, y: int, text_width: int, text_height: int) -> tuple[int, int]:
        margin = 24
        sw, sh = self.width(), self.height()
        max_x = max(margin, sw - text_width - margin)
        max_y = max(margin, sh - text_height - margin)
        return (max(margin, min(x, max_x)), max(margin, min(y, max_y)))

    def _calculate_position(self, text_width: int, text_height: int) -> tuple[int, int]:
        """Calculate (x, y) position for text based on self._position and optional YOLO bbox."""
        sw, sh = self.width(), self.height()
        left, top, right, bottom = self._anchor_rect()
        aw = right - left
        ah = bottom - top

        if self._bbox_cy is not None:
            bbox_y = top + int(self._bbox_cy * ah)
            if bbox_y > top + ah // 2:
                y = bbox_y - text_height - 48
            else:
                y = bbox_y + 48
            x = left + aw // 2 - text_width // 2
            return self._clamp_position(x, y, text_width, text_height)

        positions = {
            "top": (left + aw // 2 - text_width // 2, top + 64),
            "top_center": (left + aw // 2 - text_width // 2, top + ah // 4 - text_height // 2),
            "middle": (left + aw // 2 - text_width // 2, top + ah // 2 - text_height // 2),
            "bottom_center": (left + aw // 2 - text_width // 2, top + ah * 3 // 4 - text_height // 2),
            "bottom": (left + aw // 2 - text_width // 2, bottom - text_height - 72),
        }
        x, y = positions.get(self._position, positions["bottom_center"])
        return self._clamp_position(x, y, text_width, text_height)
