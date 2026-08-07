#!/usr/bin/env python3
"""
Granville-Phillips 392 Micro-Ion Plus pressure logger GUI.

A light-mode PyQt6 replacement for the original CLI logger. It keeps the
original serial protocol and CSV logging behavior while adding:

- Serial-port discovery and selection in the GUI.
- Editable gauge address, baud rate, interval, timeout, and reconnect delay.
- Automatic reconnect after serial/communication failures.
- Daily CSV files with timestamp, pressure, unit columns.
- Blank pressure entries for protocol errors, preserving the sample record.
- Live pressure/statistics display and application log.
- Average rate-of-change display tied to the selected graph live window.
- Torr values displayed with four significant figures, including trailing zeros.
- Matplotlib pressure history with pan, zoom, mouse-wheel navigation, and a
  Reset / Follow Live button.
- Recent history preloaded in memory, with older daily CSV files loaded lazily
  when the user pans far enough into the past.

Install dependencies:
    python -m pip install pyserial PyQt6 matplotlib

Run:
    python granville_phillips_392_gui.py

Important:
    This assumes the external RS-485/RS-232 converter performs automatic
    half-duplex transmit/receive direction switching, matching the original
    logger.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import numpy as np
import serial
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PyQt6.QtCore import QPoint, QSize, QSettings, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QFont, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleFactory,
    QStyleOptionComboBox,
    QStyleOptionSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from serial.tools import list_ports

FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$"
)


def is_torr_unit(unit: str) -> bool:
    """Return True when the gauge unit is Torr (case-insensitive)."""
    return unit.strip().upper() == "TORR"


def format_pressure_value(value: float, unit: str) -> str:
    """Format pressure for display, using exactly four significant figures in Torr."""
    if is_torr_unit(unit):
        # Scientific notation guarantees four significant figures, including
        # trailing zeros (for example 1.230E-06 instead of 1.23E-06).
        return f"{value:.3E}"
    return f"{value:.6g}"


def format_rate_value(value: float, unit: str) -> str:
    """Format a signed pressure rate for display."""
    if is_torr_unit(unit):
        return f"{value:+.3E}"
    return f"{value:+.6g}"


class GaugeError(RuntimeError):
    """Base class for gauge communication errors."""


class GaugeProtocolError(GaugeError):
    """The gauge returned no response or an invalid response."""


class GranvillePhillips392:
    """Serial protocol implementation preserved from the original logger."""

    def __init__(
        self,
        port: str,
        address: int = 1,
        baudrate: int = 19200,
        timeout: float = 1.0,
    ) -> None:
        if not 0 <= address <= 0xF:
            raise ValueError("Address must be between 0 and F.")
        self.port_name = port
        self.address = f"{address:02X}"
        self.baudrate = baudrate
        self.timeout = timeout
        self.port: Optional[serial.Serial] = None
        self.unit = "UNKNOWN"

    def connect(self) -> None:
        self.close()
        self.port = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        # Let the converter and line settle, then query the configured unit.
        time.sleep(0.1)
        self.unit = self.query("RU").strip().upper()

    def close(self) -> None:
        if self.port is not None:
            try:
                self.port.close()
            finally:
                self.port = None

    def query(self, command: str) -> str:
        """Send one gauge command and return the response data field."""
        # Keep a local reference so a GUI-requested disconnect can safely
        # close the serial object from another thread without turning
        # self.port into None halfway through this method.
        port = self.port
        if port is None or not port.is_open:
            raise serial.SerialException("Serial port is not open.")
        request_text = f"#{self.address}{command}"
        request = (request_text + "\r").encode("ascii")
        port.reset_input_buffer()
        port.write(request)
        port.flush()
        deadline = time.monotonic() + self.timeout
        expected_ok_prefix = f"*{self.address}"
        expected_error_prefix = f"?{self.address}"
        while time.monotonic() < deadline:
            raw = port.read_until(b"\r", size=128)
            if not raw:
                break
            text = raw.decode("ascii", errors="replace").strip("\r\n")
            # Some RS-232/RS-485 converters echo the transmitted request.
            if text == request_text:
                continue
            if text.startswith(expected_error_prefix):
                raise GaugeProtocolError(f"Gauge error response: {text!r}")
            if text.startswith(expected_ok_prefix):
                return text[len(expected_ok_prefix):].strip()
        raise GaugeProtocolError(
            f"No valid reply to {request_text!r} within {self.timeout:.2f} seconds."
        )

    def read_pressure(self) -> float:
        response = self.query("RD")
        if not FLOAT_PATTERN.fullmatch(response):
            raise GaugeProtocolError(
                f"Pressure response is not numeric: {response!r}"
            )
        pressure = float(response)
        # The module uses 9.99E+09 when it cannot provide valid pressure.
        if math.isclose(pressure, 9.99e9, rel_tol=0.0, abs_tol=1.0):
            raise GaugeProtocolError(
                "Gauge reported 9.99E+09 (no valid pressure available)."
            )
        return pressure


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def local_timestamp_path() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def current_csv_path(base_dir: Path) -> Path:
    return base_dir / f"pressure_log_{local_timestamp_path()}.csv"


def append_csv(
    csv_path: Path,
    timestamp: str,
    pressure: Optional[float],
    unit: str, ) -> None:
    """Append one record using the same CSV format as the original script."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if needs_header:
            writer.writerow(["timestamp", "pressure", "unit"])
        pressure_text = "" if pressure is None else f"{pressure:.9g}"
        writer.writerow([timestamp, pressure_text, unit])
        file.flush()


@dataclass(order=True)
class PressurePoint:
    timestamp: datetime
    pressure: Optional[float]
    unit: str


class HistoryStore:
    """
    Keeps recent samples ready in memory and lazily loads older daily files.
    Startup loads whole newest daily files until at least `recent_target` rows
    are present. When the graph is panned to the oldest loaded region,
    `load_older_until()` pulls earlier daily files into memory only as needed.
    """

    def __init__(self, base_dir: Path, recent_target: int = 15_000) -> None:
        self.base_dir = base_dir
        self.recent_target = recent_target
        self.points: list[PressurePoint] = []
        self.loaded_files: set[Path] = set()
        self.file_index: list[Path] = []
        self.refresh_file_index()

    def set_base_dir(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.points.clear()
        self.loaded_files.clear()
        self.refresh_file_index()
        self.preload_recent()

    def refresh_file_index(self) -> None:
        if not self.base_dir.exists():
            self.file_index = []
            return
        self.file_index = sorted(self.base_dir.glob("pressure_log_*.csv"))

    def _read_file(self, path: Path) -> list[PressurePoint]:
        rows: list[PressurePoint] = []
        try:
            with path.open("r", newline="", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    raw_ts = (row.get("timestamp") or "").strip()
                    if not raw_ts:
                        continue
                    try:
                        timestamp = datetime.fromisoformat(raw_ts)
                    except ValueError:
                        continue
                    raw_pressure = (row.get("pressure") or "").strip()
                    pressure: Optional[float]
                    if not raw_pressure:
                        pressure = None
                    else:
                        try:
                            pressure = float(raw_pressure)
                        except ValueError:
                            pressure = None
                    rows.append(
                        PressurePoint(
                            timestamp=timestamp,
                            pressure=pressure,
                            unit=(row.get("unit") or "UNKNOWN").strip() or "UNKNOWN",
                        )
                    )
        except (OSError, csv.Error):
            return []
        rows.sort(key=lambda point: point.timestamp)
        return rows

    def _merge_points(self, incoming: list[PressurePoint]) -> int:
        if not incoming:
            return 0
        # Timestamp is a natural key because each sampling row gets one ISO
        # timestamp. Preserve the newest copy if a duplicate somehow appears.
        merged = {point.timestamp: point for point in self.points}
        for point in incoming:
            merged[point.timestamp] = point
        self.points = sorted(merged.values(), key=lambda point: point.timestamp)
        return len(incoming)

    def preload_recent(self) -> int:
        self.refresh_file_index()
        loaded_rows = 0
        for path in reversed(self.file_index):
            if path in self.loaded_files:
                continue
            rows = self._read_file(path)
            self.loaded_files.add(path)
            loaded_rows += self._merge_points(rows)
            if len(self.points) >= self.recent_target:
                break
        return loaded_rows

    def append_live(self, point: PressurePoint) -> None:
        # Live data should normally be newer than everything already loaded,
        # making this O(1). If time moves backwards, fall back to a merge.
        if not self.points or point.timestamp > self.points[-1].timestamp:
            self.points.append(point)
        elif point.timestamp == self.points[-1].timestamp:
            self.points[-1] = point
        else:
            self._merge_points([point])

    def has_older_files(self) -> bool:
        self.refresh_file_index()
        if not self.file_index:
            return False
        if not self.loaded_files:
            return bool(self.file_index)
        oldest_loaded = min(self.loaded_files)
        return any(path < oldest_loaded and path not in self.loaded_files for path in self.file_index)

    def load_previous_file(self) -> tuple[Optional[Path], int]:
        self.refresh_file_index()
        candidates = [path for path in self.file_index if path not in self.loaded_files]
        if self.loaded_files:
            oldest_loaded = min(self.loaded_files)
            candidates = [path for path in candidates if path < oldest_loaded]
        if not candidates:
            return None, 0
        path = max(candidates)
        rows = self._read_file(path)
        self.loaded_files.add(path)
        added = self._merge_points(rows)
        return path, added

    def load_older_until(self, target_time: datetime, max_files: int = 14) -> list[tuple[Path, int]]:
        loaded: list[tuple[Path, int]] = []
        for _ in range(max_files):
            if self.points and self.points[0].timestamp <= target_time:
                break
            path, rows = self.load_previous_file()
            if path is None:
                break
            loaded.append((path, rows))
        return loaded

    def arrays(self) -> tuple[list[datetime], np.ndarray]:
        times = [point.timestamp for point in self.points]
        values = np.array(
            [np.nan if point.pressure is None else point.pressure for point in self.points],
            dtype=float,
        )
        return times, values


@dataclass(frozen=True)
class LoggerConfig:
    port: str
    address: int
    baudrate: int
    timeout: float
    interval: float
    reconnect_delay: float
    data_dir: Path


class LoggerThread(QThread):
    """Blocking serial logger running outside the GUI thread."""
    log_message = pyqtSignal(str, str)  # message, level
    reading = pyqtSignal(str, object, str, str)  # timestamp, pressure|None, unit, csv_path
    connection_state = pyqtSignal(str)  # connecting/connected/reconnecting/stopped
    connected_info = pyqtSignal(str, str)  # unit, port

    def __init__(self, config: LoggerConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self._stop_event = threading.Event()
        self._gauge: Optional[GranvillePhillips392] = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._gauge is not None:
            # Closing the port helps unblock a serial read sooner on many
            # platforms. Any resulting SerialException is handled in run().
            try:
                self._gauge.close()
            except Exception:
                pass

    def _sleep_interruptibly(self, seconds: float) -> bool:
        """Return True if stop was requested during the sleep."""
        return self._stop_event.wait(max(0.0, seconds))

    def run(self) -> None:
        cfg = self.config
        gauge = GranvillePhillips392(
            port=cfg.port,
            address=cfg.address,
            baudrate=cfg.baudrate,
            timeout=cfg.timeout,
        )
        self._gauge = gauge
        next_read = time.monotonic()
        try:
            while not self._stop_event.is_set():
                if gauge.port is None or not gauge.port.is_open:
                    self.connection_state.emit("connecting")
                    try:
                        gauge.connect()
                        self.connection_state.emit("connected")
                        self.connected_info.emit(gauge.unit, cfg.port)
                        self.log_message.emit(
                            f"Connected to {cfg.port}; address={cfg.address:X}, "
                            f"baud={cfg.baudrate}, unit={gauge.unit}",
                            "success",
                        )
                        next_read = time.monotonic()
                    except (serial.SerialException, GaugeError) as exc:
                        timestamp = local_timestamp()
                        self.connection_state.emit("reconnecting")
                        self.log_message.emit(
                            f"{timestamp} | connect error: {exc} | retrying in "
                            f"{cfg.reconnect_delay:g}s",
                            "error",
                        )
                        gauge.close()
                        if self._sleep_interruptibly(cfg.reconnect_delay):
                            break
                        continue
                sleep_seconds = next_read - time.monotonic()
                if sleep_seconds > 0 and self._sleep_interruptibly(sleep_seconds):
                    break
                if self._stop_event.is_set():
                    break
                timestamp = local_timestamp()
                csv_path = current_csv_path(cfg.data_dir)
                try:
                    pressure = gauge.read_pressure()
                    append_csv(csv_path, timestamp, pressure, gauge.unit)
                    self.reading.emit(timestamp, pressure, gauge.unit, str(csv_path))
                    self.log_message.emit(
                        f"{timestamp} | {format_pressure_value(pressure, gauge.unit)} "
                        f"{gauge.unit} | appended to {csv_path}",
                        "info",
                    )
                except GaugeProtocolError as exc:
                    # Preserve the sampling record with a blank pressure exactly
                    # as the original CLI logger does.
                    append_csv(csv_path, timestamp, None, gauge.unit)
                    self.reading.emit(timestamp, None, gauge.unit, str(csv_path))
                    self.log_message.emit(
                        f"{timestamp} | read error: {exc}",
                        "error",
                    )
                except serial.SerialException as exc:
                    self.log_message.emit(
                        f"{timestamp} | serial error: {exc}",
                        "error",
                    )
                    self.connection_state.emit("reconnecting")
                    gauge.close()
                next_read += cfg.interval
                now = time.monotonic()
                if next_read < now - cfg.interval:
                    next_read = now + cfg.interval
        finally:
            gauge.close()
            self._gauge = None
            self.connection_state.emit("stopped")
            self.log_message.emit("Logger stopped.", "muted")


def _draw_chevron(widget: QWidget, rect, direction: str) -> None:
    """Paint a small high-contrast chevron over a Qt sub-control."""
    if rect.isEmpty():
        return
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(QColor("#334155" if widget.isEnabled() else "#94a3b8"), 1.8)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    center = rect.center()
    cx, cy = center.x(), center.y()
    half_w = max(3, min(5, rect.width() // 5))
    half_h = max(2, min(3, rect.height() // 5))
    if direction == "up":
        points = (
            QPoint(cx - half_w, cy + half_h),
            QPoint(cx, cy - half_h),
            QPoint(cx + half_w, cy + half_h),
        )
    else:
        points = (
            QPoint(cx - half_w, cy - half_h),
            QPoint(cx, cy + half_h),
            QPoint(cx + half_w, cy - half_h),
        )
    painter.drawLine(points[0], points[1])
    painter.drawLine(points[1], points[2])
    painter.end()


class StyledComboBox(QComboBox):
    """Combo box with a styled popup and an always-visible drop arrow."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        view = QListView(self)
        view.setObjectName("ComboPopupView")
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setView(view)

    def paintEvent(self, event) -> None:
        # Stylesheets can suppress the platform arrow image. Draw the chevron
        # ourselves after Qt paints the control so it remains visible everywhere.
        super().paintEvent(event)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        arrow_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxArrow,
            self,
        )
        _draw_chevron(self, arrow_rect, "down")

    def showPopup(self) -> None:
        # Qt creates the combo popup as a separate top-level frame. Styling
        # that frame explicitly avoids the thin native popup rim that can
        # otherwise appear around dropdown lists on Windows.
        popup = self.view().window()
        popup.setObjectName("ComboPopupWindow")
        popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        popup.setStyleSheet(
            "QFrame#ComboPopupWindow {"
            "background-color: #ffffff;"
            "border: 1px solid #cbd5e1;"
            "border-radius: 6px;"
            "padding: 0px;"
            "}"
        )
        super().showPopup()


class StyledDoubleSpinBox(QDoubleSpinBox):
    """Double spin box with visible up/down buttons under Qt stylesheets."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        style = self.style()
        up_rect = style.subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxUp,
            self,
        )
        down_rect = style.subControlRect(
            QStyle.ComplexControl.CC_SpinBox,
            option,
            QStyle.SubControl.SC_SpinBoxDown,
            self,
        )
        _draw_chevron(self, up_rect, "up")
        _draw_chevron(self, down_rect, "down")

MANUAL_PORT_TOKEN = "__manual_port__"


class StatCard(QFrame):
    def __init__(self, title: str, value: str = "—", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 6, 9, 6)
        layout.setSpacing(1)
        title_label = QLabel(title.upper())
        title_label.setObjectName("StatTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatValue")
        self.value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.value_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class BlackNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib toolbar with black glyphs regardless of the OS palette."""

    _ICON_SIZES = (16, 18, 20, 24, 32, 48)

    def _icon(self, name):
        # NavigationToolbar2QT decides whether to recolor its icons from the
        # palette that exists *while the toolbar is constructed*.  On Windows
        # with a dark system palette, that can create white glyphs before this
        # application's light stylesheet is applied.  Recolor the finished
        # icon by alpha so the glyph is always black on our white toolbar.
        source_icon = super()._icon(name)
        black_icon = QIcon()
        for pixels in self._ICON_SIZES:
            pixmap = source_icon.pixmap(QSize(pixels, pixels))
            if pixmap.isNull():
                continue
            painter = QPainter(pixmap)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(pixmap.rect(), QColor("#000000"))
            painter.end()
            black_icon.addPixmap(pixmap)
        return black_icon if not black_icon.isNull() else source_icon


class PressureGraph(QWidget):
    history_requested = pyqtSignal(object)  # datetime target
    follow_changed = pyqtSignal(bool)
    window_changed = pyqtSignal(str)
    WINDOW_OPTIONS = {
        "15 min": timedelta(minutes=15),
        "1 hour": timedelta(hours=1),
        "6 hours": timedelta(hours=6),
        "24 hours": timedelta(hours=24),
        "3 days": timedelta(days=3),
        "All loaded": None,
    }

    def __init__(self, store: HistoryStore, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.store = store
        self.follow_live = True
        self._changing_limits = False
        self._last_history_request: Optional[datetime] = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("Pressure History")
        title.setObjectName("SectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(QLabel("Live window:"))
        self.window_combo = StyledComboBox()
        self.window_combo.addItems(self.WINDOW_OPTIONS.keys())
        self.window_combo.setCurrentText("1 hour")
        self.window_combo.setMinimumHeight(30)
        self.window_combo.setMinimumWidth(92)
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        header.addWidget(self.window_combo)
        self.reset_button = QPushButton("Reset / Follow Live")
        self.reset_button.setObjectName("PrimaryButton")
        self.reset_button.setMinimumHeight(30)
        self.reset_button.clicked.connect(self.reset_live_view)
        header.addWidget(self.reset_button)
        layout.addWidget(self.header_widget)
        self.figure = Figure(figsize=(7, 4), tight_layout=True)
        self.figure.patch.set_facecolor("#ffffff")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()
        self.toolbar = BlackNavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("GraphToolbar")
        self.toolbar.setIconSize(QSize(18, 18))
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        (self.line,) = self.ax.plot([], [], linewidth=1.6, marker="", color="#2563eb", label="Pressure")
        self.ax.set_ylabel("Pressure")
        self.ax.set_xlabel("Local time")
        self.ax.grid(True, alpha=0.18)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        self.ax.yaxis.set_major_formatter(FuncFormatter(self._format_y_tick))
        self.ax.format_ydata = self._format_y_cursor
        self.ax.callbacks.connect("xlim_changed", self._on_xlim_changed)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_user_navigation)
        self.canvas.mpl_connect("key_press_event", self._on_user_navigation)
        self.refresh_data(preserve_view=False)

    def set_compact_mode(self, compact: bool) -> None:
        """Show only the plot canvas when the main window is in compact mode."""
        self.header_widget.setVisible(not compact)
        self.toolbar.setVisible(not compact)

    def _display_unit(self) -> str:
        """Use the newest known history unit when formatting graph pressure values."""
        for point in reversed(self.store.points):
            if point.unit:
                return point.unit
        return "UNKNOWN"

    def _format_y_tick(self, value: float, _position: object = None) -> str:
        return format_pressure_value(float(value), self._display_unit())

    def _format_y_cursor(self, value: float) -> str:
        return format_pressure_value(float(value), self._display_unit())

    def _on_window_changed(self, text: str) -> None:
        self.reset_live_view()
        self.window_changed.emit(text)

    def _style_axes(self) -> None:
        self.ax.set_facecolor("#ffffff")
        for spine in self.ax.spines.values():
            spine.set_color("#94a3b8")
        self.ax.tick_params(colors="#334155")
        self.ax.xaxis.label.set_color("#334155")
        self.ax.yaxis.label.set_color("#334155")
        self.ax.title.set_color("#0f172a")

    def _on_user_navigation(self, _event) -> None:
        # Clicking/keyboard navigation through Matplotlib should stop automatic
        # following; Reset / Follow Live turns it back on.
        if self.follow_live:
            self.follow_live = False
            self.follow_changed.emit(False)

    def _on_scroll(self, event) -> None:
        if event.xdata is None:
            return
        self.follow_live = False
        self.follow_changed.emit(False)
        xmin, xmax = self.ax.get_xlim()
        span = xmax - xmin
        if span <= 0:
            return
        key = (event.key or "").lower()
        if "shift" in key:
            # Shift + wheel pans horizontally.
            direction = -1.0 if event.button == "up" else 1.0
            delta = span * 0.18 * direction
            self._set_xlim(xmin + delta, xmax + delta)
        else:
            # Wheel zooms horizontally around the mouse cursor.
            scale = 0.80 if event.button == "up" else 1.25
            new_span = span * scale
            cursor = event.xdata
            ratio = (cursor - xmin) / span
            new_min = cursor - new_span * ratio
            new_max = new_min + new_span
            self._set_xlim(new_min, new_max)
        self._autoscale_y_for_visible_data()
        self.canvas.draw_idle()

    def _set_xlim(self, xmin: float, xmax: float) -> None:
        self._changing_limits = True
        try:
            self.ax.set_xlim(xmin, xmax)
        finally:
            self._changing_limits = False
        self._maybe_request_history(xmin, xmax)

    def _on_xlim_changed(self, axes) -> None:
        if self._changing_limits:
            return
        xmin, xmax = axes.get_xlim()
        if not self.follow_live:
            self._maybe_request_history(xmin, xmax)
        self._autoscale_y_for_visible_data()

    def _maybe_request_history(self, xmin: float, xmax: float) -> None:
        if not self.store.points:
            return
        oldest = self.store.points[0].timestamp
        oldest_num = mdates.date2num(oldest)
        visible_span = max(xmax - xmin, 1e-9)
        threshold = oldest_num + visible_span * 0.12
        if xmin > threshold:
            return
        # Ask for enough older data to cover slightly beyond the requested
        # left edge. This avoids loading a file for tiny navigation movements.
        target_num = xmin - visible_span * 0.15
        target_dt = mdates.num2date(target_num, tz=oldest.tzinfo)
        if self._last_history_request is not None:
            if abs((target_dt - self._last_history_request).total_seconds()) < 1.0:
                return
        self._last_history_request = target_dt
        self.history_requested.emit(target_dt)

    def refresh_data(self, preserve_view: bool = True) -> None:
        old_xlim = self.ax.get_xlim()
        times, values = self.store.arrays()
        self.line.set_data(times, values)
        if not times:
            self.ax.set_title("No data loaded yet")
            self.canvas.draw_idle()
            return
        self.ax.set_title(f"{len(times):,} samples loaded in memory")
        self.ax.relim()
        if self.follow_live or not preserve_view:
            self.reset_live_view()
        else:
            self._changing_limits = True
            try:
                self.ax.set_xlim(old_xlim)
            finally:
                self._changing_limits = False
            self._autoscale_y_for_visible_data()
            self.canvas.draw_idle()

    def reset_live_view(self, *_args) -> None:
        times, values = self.store.arrays()
        if not times:
            self.canvas.draw_idle()
            return
        self.follow_live = True
        self.follow_changed.emit(True)
        latest = times[-1]
        selected = self.WINDOW_OPTIONS[self.window_combo.currentText()]
        if selected is None:
            left = times[0]
        else:
            left = latest - selected
            if left < times[0]:
                left = times[0]
        right = latest + timedelta(seconds=1)
        self._changing_limits = True
        try:
            self.ax.set_xlim(left, right)
        finally:
            self._changing_limits = False
        self._autoscale_y_for_visible_data()
        self.canvas.draw_idle()

    def _autoscale_y_for_visible_data(self) -> None:
        """
        Fit the Y axis to the currently visible pressure samples.

        A small amount of padding is kept above and below the data so the
        trace never sits directly against the graph boundary.
        """
        if not self.store.points:
            return

        xmin, xmax = self.ax.get_xlim()
        times, values = self.store.arrays()
        xnums = mdates.date2num(times)

        mask = (
            (xnums >= xmin)
            & (xnums <= xmax)
            & np.isfinite(values)
        )
        visible = values[mask]

        if visible.size == 0:
            return

        ymin = float(np.min(visible))
        ymax = float(np.max(visible))

        if math.isclose(ymin, ymax, rel_tol=1e-12, abs_tol=1e-30):
            # Flat/near-flat traces still need some visible vertical room.
            pad = max(abs(ymin) * 0.05, 1e-12)
        else:
            # 6% breathing room above and below the visible data.
            pad = (ymax - ymin) * 0.06

        self.ax.set_ylim(ymin - pad, ymax + pad)


class MainWindow(QMainWindow):
    NORMAL_LAYOUT_MIN_WIDTH = 1120
    NORMAL_LAYOUT_MIN_HEIGHT = 760
    COMPACT_MIN_WIDTH = 420
    COMPACT_MIN_HEIGHT = 320

    def __init__(self) -> None:
        super().__init__()
        self._compact_mode = False
        self.setWindowTitle("Granville-Phillips 392 Pressure Logger")
        self.resize(1440, 900)
        # 1120x760 is now the responsive breakpoint instead of a hard stop.
        self.setMinimumSize(self.COMPACT_MIN_WIDTH, self.COMPACT_MIN_HEIGHT)
        self.settings = QSettings("GP392Tools", "GP392PressureLogger")
        initial_dir = Path(self.settings.value("data_dir", "./data/", type=str))
        self.history = HistoryStore(initial_dir)
        preloaded = self.history.preload_recent()
        self.logger_thread: Optional[LoggerThread] = None
        self.session_pressures: list[float] = []
        self.session_samples = 0
        self.session_errors = 0
        self.last_csv_path = "—"
        self.current_unit = "UNKNOWN"
        self._build_ui()
        self._apply_light_theme()
        self._load_saved_settings()
        self.refresh_ports(log_result=False)
        self.graph.refresh_data(preserve_view=False)
        self._update_rate_of_change()
        self._update_responsive_layout(force=True)
        if preloaded:
            self.add_log(
                f"Preloaded {preloaded:,} recent rows from {len(self.history.loaded_files)} CSV file(s).",
                "muted",
            )
        else:
            self.add_log("No existing pressure history found in the selected data folder.", "muted")
    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(8)
        self.header_widget = QWidget()
        header = QHBoxLayout(self.header_widget)
        header.setContentsMargins(0, 0, 0, 0)
        title_wrap = QVBoxLayout()
        app_title = QLabel("Granville-Phillips 392")
        app_title.setObjectName("AppTitle")
        subtitle = QLabel("Micro-Ion Plus Pressure Logger")
        subtitle.setObjectName("AppSubtitle")
        title_wrap.addWidget(app_title)
        title_wrap.addWidget(subtitle)
        header.addLayout(title_wrap)
        header.addStretch(1)
        self.header_status = QLabel("DISCONNECTED")
        self.header_status.setObjectName("HeaderStatusDisconnected")
        self.header_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.header_status)
        root.addWidget(self.header_widget)
        # Fixed four-quadrant layout. The previous splitter handles looked like
        # draggable window dividers but were not useful in practice, so the
        # sections now use equal proportions and equal gutters.
        body = QGridLayout()
        self.body_layout = body
        body.setContentsMargins(0, 0, 0, 0)
        body.setHorizontalSpacing(8)
        body.setVerticalSpacing(8)
        body.setColumnStretch(0, 1)
        body.setColumnStretch(1, 1)
        body.setRowStretch(0, 1)
        body.setRowStretch(1, 1)
        # Upper-left: controls/settings.
        self.controls_panel = self._build_controls_panel()
        # Upper-right: large live stats display.
        self.stats_panel = self._build_stats_panel()
        # Lower-left: logs.
        self.logs_panel = self._build_log_panel()
        # Lower-right: graph.
        self.graph = PressureGraph(self.history)
        self.graph_box = QGroupBox("Live && Historical Graph")
        graph_layout = QVBoxLayout(self.graph_box)
        graph_layout.setContentsMargins(10, 12, 10, 10)
        graph_layout.addWidget(self.graph)
        self.graph.history_requested.connect(self._load_older_history)
        self.graph.window_changed.connect(self._update_rate_of_change)
        for widget in (self.controls_panel, self.stats_panel, self.logs_panel, self.graph_box):
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body.addWidget(self.controls_panel, 0, 0)
        body.addWidget(self.stats_panel, 0, 1)
        body.addWidget(self.logs_panel, 1, 0)
        body.addWidget(self.graph_box, 1, 1)
        root.addLayout(body, 1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "body_layout"):
            self._update_responsive_layout()

    def _update_responsive_layout(self, force: bool = False) -> None:
        compact = (
            self.width() < self.NORMAL_LAYOUT_MIN_WIDTH
            or self.height() < self.NORMAL_LAYOUT_MIN_HEIGHT
        )
        if compact == self._compact_mode and not force:
            return
        self._compact_mode = compact

        panels = (
            self.controls_panel,
            self.stats_panel,
            self.logs_panel,
            self.graph_box,
        )
        for panel in panels:
            self.body_layout.removeWidget(panel)

        if compact:
            # Small-window view: current pressure + graph, and literally no
            # connection controls, extra statistics, logs, app header, graph
            # header, or Matplotlib toolbar.
            self.header_widget.hide()
            self.controls_panel.hide()
            self.logs_panel.hide()
            self.stats_cards_widget.hide()
            self.stats_panel.setTitle("")
            self.graph_box.setTitle("")
            self.graph.set_compact_mode(True)

            self.stats_panel.show()
            self.graph_box.show()
            self.body_layout.addWidget(self.stats_panel, 0, 0, 1, 2)
            self.body_layout.addWidget(self.graph_box, 1, 0, 1, 2)
            self.body_layout.setRowStretch(0, 0)
            self.body_layout.setRowStretch(1, 1)
        else:
            self.header_widget.show()
            self.controls_panel.show()
            self.logs_panel.show()
            self.stats_cards_widget.show()
            self.stats_panel.setTitle("Live Gauge Status")
            self.graph_box.setTitle("Live && Historical Graph")
            self.graph.set_compact_mode(False)

            self.body_layout.addWidget(self.controls_panel, 0, 0)
            self.body_layout.addWidget(self.stats_panel, 0, 1)
            self.body_layout.addWidget(self.logs_panel, 1, 0)
            self.body_layout.addWidget(self.graph_box, 1, 1)
            self.body_layout.setRowStretch(0, 1)
            self.body_layout.setRowStretch(1, 1)

        self.body_layout.activate()
        self.graph.canvas.draw_idle()

    def _build_controls_panel(self) -> QWidget:
        panel = QGroupBox("Connection && Logging Controls")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 10)
        layout.setSpacing(7)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        port_row = QHBoxLayout()
        port_row.setSpacing(6)
        self.port_combo = StyledComboBox()
        self.port_combo.setEditable(False)
        self.port_combo.setMinimumWidth(220)
        self.port_combo.setMinimumHeight(30)
        self.port_combo.activated.connect(self._on_port_activated)
        self.refresh_ports_button = QPushButton("Refresh")
        self.refresh_ports_button.setFixedWidth(78)
        self.refresh_ports_button.setMinimumHeight(30)
        self.refresh_ports_button.clicked.connect(self.refresh_ports)
        port_row.addWidget(self.port_combo, 1)
        port_row.addWidget(self.refresh_ports_button)
        form.addRow("Serial port", port_row)
        self.address_combo = StyledComboBox()
        for value in range(16):
            self.address_combo.addItem(f"{value:X}", value)
        self.address_combo.setCurrentIndex(1)
        self.address_combo.setMinimumHeight(30)
        form.addRow("Gauge address", self.address_combo)
        self.baud_combo = StyledComboBox()
        self.baud_combo.addItems(["1200", "2400", "4800", "9600", "19200", "38400"])
        self.baud_combo.setCurrentText("19200")
        self.baud_combo.setMinimumHeight(30)
        form.addRow("Baud rate", self.baud_combo)
        self.interval_spin = StyledDoubleSpinBox()
        self.interval_spin.setRange(0.1, 86400.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSingleStep(1.0)
        self.interval_spin.setValue(10.0)
        self.interval_spin.setSuffix(" s")
        self.interval_spin.setMinimumHeight(30)
        form.addRow("Read interval", self.interval_spin)
        self.timeout_spin = StyledDoubleSpinBox()
        self.timeout_spin.setRange(0.1, 60.0)
        self.timeout_spin.setDecimals(2)
        self.timeout_spin.setValue(1.0)
        self.timeout_spin.setSuffix(" s")
        self.timeout_spin.setMinimumHeight(30)
        form.addRow("Reply timeout", self.timeout_spin)
        self.reconnect_spin = StyledDoubleSpinBox()
        self.reconnect_spin.setRange(0.1, 3600.0)
        self.reconnect_spin.setDecimals(1)
        self.reconnect_spin.setValue(5.0)
        self.reconnect_spin.setSuffix(" s")
        self.reconnect_spin.setMinimumHeight(30)
        form.addRow("Reconnect delay", self.reconnect_spin)
        data_row = QHBoxLayout()
        data_row.setSpacing(6)
        self.data_dir_edit = QLineEdit(str(self.history.base_dir))
        self.data_dir_edit.setMinimumHeight(30)
        self.data_dir_edit.editingFinished.connect(self._on_data_dir_edited)
        self.browse_button = QPushButton("Browse")
        self.browse_button.setFixedWidth(78)
        self.browse_button.setMinimumHeight(30)
        self.browse_button.clicked.connect(self.browse_data_dir)
        data_row.addWidget(self.data_dir_edit, 1)
        data_row.addWidget(self.browse_button)
        form.addRow("CSV data folder", data_row)
        layout.addLayout(form)
        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self.connect_button = QPushButton("Connect && Start Logging")
        self.connect_button.setObjectName("PrimaryButton")
        self.connect_button.clicked.connect(self.start_logging)
        self.disconnect_button = QPushButton("Disconnect")
        self.disconnect_button.setObjectName("DangerButton")
        self.disconnect_button.setEnabled(False)
        self.disconnect_button.clicked.connect(self.stop_logging)
        self.connect_button.setMinimumHeight(32)
        self.disconnect_button.setMinimumHeight(32)
        button_row.addWidget(self.connect_button, 1)
        button_row.addWidget(self.disconnect_button, 1)
        layout.addLayout(button_row)
        utility_row = QHBoxLayout()
        utility_row.setSpacing(6)
        self.open_folder_button = QPushButton("Open Data Folder")
        self.open_folder_button.clicked.connect(self.open_data_folder)
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(lambda: self.log_text.clear())
        self.open_folder_button.setMinimumHeight(30)
        self.clear_log_button.setMinimumHeight(30)
        utility_row.addWidget(self.open_folder_button, 1)
        utility_row.addWidget(self.clear_log_button, 1)
        layout.addLayout(utility_row)
        hint = QLabel(
            "Tip: mouse wheel = horizontal zoom, Shift + wheel = horizontal pan. "
            "The Matplotlib toolbar also provides rectangle zoom and pan."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)
        layout.addStretch(1)
        self.config_widgets = [
            self.port_combo,
            self.refresh_ports_button,
            self.address_combo,
            self.baud_combo,
            self.interval_spin,
            self.timeout_spin,
            self.reconnect_spin,
            self.data_dir_edit,
            self.browse_button,
        ]
        return panel

    def _build_stats_panel(self) -> QWidget:
        panel = QGroupBox("Live Gauge Status")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 14, 12, 10)
        layout.setSpacing(7)
        pressure_container = QFrame()
        pressure_container.setObjectName("PressureHero")
        pressure_layout = QVBoxLayout(pressure_container)
        pressure_layout.setContentsMargins(12, 7, 12, 7)
        pressure_layout.setSpacing(1)
        label = QLabel("CURRENT PRESSURE")
        label.setObjectName("HeroCaption")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pressure_value = QLabel("—")
        self.pressure_value.setObjectName("PressureValue")
        self.pressure_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pressure_unit = QLabel("UNKNOWN")
        self.pressure_unit.setObjectName("PressureUnit")
        self.pressure_unit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pressure_layout.addWidget(label)
        pressure_layout.addWidget(self.pressure_value)
        pressure_layout.addWidget(self.pressure_unit)
        layout.addWidget(pressure_container)
        self.stats_cards_widget = QWidget()
        grid = QGridLayout(self.stats_cards_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for column in range(4):
            grid.setColumnStretch(column, 1)
        self.status_card = StatCard("Connection", "Disconnected")
        self.port_card = StatCard("Port", "—")
        self.last_read_card = StatCard("Last sample", "—")
        self.samples_card = StatCard("Session samples", "0")
        self.errors_card = StatCard("Invalid samples", "0")
        self.min_card = StatCard("Session minimum", "—")
        self.max_card = StatCard("Session maximum", "—")
        self.average_card = StatCard("Session average", "—")
        self.rate_card = StatCard("Rate of change", "—")
        self.file_card = StatCard("Current CSV", "—")
        self.memory_card = StatCard("History in memory", f"{len(self.history.points):,} rows")
        top_cards = [
            self.status_card,
            self.port_card,
            self.last_read_card,
            self.samples_card,
            self.errors_card,
            self.min_card,
            self.max_card,
            self.average_card,
        ]
        for index, card in enumerate(top_cards):
            grid.addWidget(card, index // 4, index % 4)
        # Rate of change is intentionally wide because it includes the signed
        # value, unit/minute, and the currently selected graph window.
        grid.addWidget(self.rate_card, 2, 0, 1, 4)
        # The two potentially wider values share the full last row.
        grid.addWidget(self.file_card, 3, 0, 1, 2)
        grid.addWidget(self.memory_card, 3, 2, 1, 2)
        layout.addWidget(self.stats_cards_widget, 1)
        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QGroupBox("Application Log")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 10)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        self.log_text.setFont(font)
        layout.addWidget(self.log_text)
        return panel
    # ---------- Settings / ports ----------

    def _load_saved_settings(self) -> None:
        self.baud_combo.setCurrentText(self.settings.value("baud", "19200", type=str))
        self.interval_spin.setValue(self.settings.value("interval", 10.0, type=float))
        self.timeout_spin.setValue(self.settings.value("timeout", 1.0, type=float))
        self.reconnect_spin.setValue(self.settings.value("reconnect", 5.0, type=float))
        address = self.settings.value("address", 1, type=int)
        self.address_combo.setCurrentIndex(max(0, min(15, address)))
        self.data_dir_edit.setText(self.settings.value("data_dir", "./data/", type=str))

    def _save_settings(self) -> None:
        self.settings.setValue("baud", self.baud_combo.currentText())
        self.settings.setValue("interval", self.interval_spin.value())
        self.settings.setValue("timeout", self.timeout_spin.value())
        self.settings.setValue("reconnect", self.reconnect_spin.value())
        self.settings.setValue("address", self.address_combo.currentData())
        self.settings.setValue("data_dir", self.data_dir_edit.text().strip())
        port = self.port_combo.currentData()
        if port and port != MANUAL_PORT_TOKEN:
            self.settings.setValue("last_port", str(port))

    def refresh_ports(self, checked: bool = False, log_result: bool = True) -> None:
        del checked
        previous = self.port_combo.currentData()
        if previous == MANUAL_PORT_TOKEN:
            previous = None
        saved = self.settings.value("last_port", "", type=str)
        ports = list(list_ports.comports())
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        if not ports:
            self.port_combo.addItem("No detected serial ports", None)
        else:
            for port in ports:
                description = port.description or "Unknown device"
                self.port_combo.addItem(f"{port.device} — {description}", port.device)
        preferred = previous or saved
        detected_values = {self.port_combo.itemData(i) for i in range(self.port_combo.count())}
        if preferred and preferred not in detected_values:
            # Preserve the original CLI's ability to target an explicit port
            # even when OS discovery does not currently report it.
            self.port_combo.addItem(f"{preferred} — saved/manual port", preferred)
        self.port_combo.insertSeparator(self.port_combo.count())
        self.port_combo.addItem("Enter port manually…", MANUAL_PORT_TOKEN)
        selected = False
        if preferred:
            for index in range(self.port_combo.count()):
                if self.port_combo.itemData(index) == preferred:
                    self.port_combo.setCurrentIndex(index)
                    selected = True
                    break
        if not selected:
            for index in range(self.port_combo.count()):
                data = self.port_combo.itemData(index)
                if data and data != MANUAL_PORT_TOKEN:
                    self.port_combo.setCurrentIndex(index)
                    selected = True
                    break
        self.port_combo.blockSignals(False)
        if log_result:
            if ports:
                self.add_log(f"Detected {len(ports)} serial port(s):", "muted")
                for port in ports:
                    self.add_log(f"  {port.device}: {port.description or 'Unknown device'}", "muted")
            else:
                self.add_log("No serial ports detected. You can still choose Enter port manually…", "warning")

    def _on_port_activated(self, index: int) -> None:
        if self.port_combo.itemData(index) != MANUAL_PORT_TOKEN:
            return
        saved = self.settings.value("last_port", "", type=str)
        text, accepted = QInputDialog.getText(
            self,
            "Enter serial port",
            "Serial port (for example COM5 or /dev/ttyUSB0):",
            text=saved,
        )
        port = text.strip() if accepted else ""
        if port:
            manual_index = self.port_combo.findData(MANUAL_PORT_TOKEN)
            insert_at = manual_index if manual_index >= 0 else self.port_combo.count()
            existing = self.port_combo.findData(port)
            if existing >= 0:
                self.port_combo.setCurrentIndex(existing)
            else:
                self.port_combo.insertItem(insert_at, f"{port} — manual port", port)
                self.port_combo.setCurrentIndex(insert_at)
            return
        # If the prompt is cancelled, return to the first valid port rather
        # than leaving the action item selected.
        for candidate in range(self.port_combo.count()):
            data = self.port_combo.itemData(candidate)
            if data and data != MANUAL_PORT_TOKEN:
                self.port_combo.setCurrentIndex(candidate)
                return
        self.port_combo.setCurrentIndex(0)

    def _on_data_dir_edited(self) -> None:
        text = self.data_dir_edit.text().strip()
        if text:
            self._switch_history_directory(Path(text))

    def browse_data_dir(self) -> None:
        current = self.data_dir_edit.text().strip() or "./data/"
        selected = QFileDialog.getExistingDirectory(self, "Select CSV data folder", current)
        if selected:
            self.data_dir_edit.setText(selected)
            self._switch_history_directory(Path(selected))

    def open_data_folder(self) -> None:
        path = Path(self.data_dir_edit.text().strip() or "./data/").expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        # QDesktopServices avoids platform-specific shell calls.
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _switch_history_directory(self, path: Path) -> None:
        if self.logger_thread is not None and self.logger_thread.isRunning():
            return
        path = path.expanduser()
        if path == self.history.base_dir:
            return
        self.history.set_base_dir(path)
        self.graph.refresh_data(preserve_view=False)
        self.memory_card.set_value(f"{len(self.history.points):,} rows")
        self._update_rate_of_change()
        self.add_log(
            f"History folder changed to {path}; loaded {len(self.history.points):,} recent rows.",
            "muted",
        )
    # ---------- Logger lifecycle ----------

    def _build_config(self) -> Optional[LoggerConfig]:
        port = self.port_combo.currentData()
        if not port or port == MANUAL_PORT_TOKEN:
            QMessageBox.warning(self, "No serial port", "Select a valid serial port first.")
            return None
        data_text = self.data_dir_edit.text().strip()
        if not data_text:
            QMessageBox.warning(self, "No data folder", "Choose a folder for the daily CSV files.")
            return None
        data_dir = Path(data_text).expanduser()
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Data folder error", f"Could not create/access the data folder:\n{exc}")
            return None
        return LoggerConfig(
            port=str(port),
            address=int(self.address_combo.currentData()),
            baudrate=int(self.baud_combo.currentText()),
            timeout=float(self.timeout_spin.value()),
            interval=float(self.interval_spin.value()),
            reconnect_delay=float(self.reconnect_spin.value()),
            data_dir=data_dir,
        )

    def start_logging(self) -> None:
        if self.logger_thread is not None and self.logger_thread.isRunning():
            return
        config = self._build_config()
        if config is None:
            return
        self._switch_history_directory(config.data_dir)
        self._save_settings()
        self.session_pressures.clear()
        self.session_samples = 0
        self.session_errors = 0
        self._update_session_stats()
        thread = LoggerThread(config, self)
        thread.log_message.connect(self.add_log)
        thread.reading.connect(self._on_reading)
        thread.connection_state.connect(self._on_connection_state)
        thread.connected_info.connect(self._on_connected_info)
        thread.finished.connect(self._on_thread_finished)
        self.logger_thread = thread
        for widget in self.config_widgets:
            widget.setEnabled(False)
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.port_card.set_value(config.port)
        self._on_connection_state("connecting")
        self.add_log("Starting logger...", "muted")
        thread.start()

    def stop_logging(self) -> None:
        if self.logger_thread is None:
            return
        self.add_log("Stopping logger...", "muted")
        self.disconnect_button.setEnabled(False)
        self.logger_thread.stop()

    def _on_thread_finished(self) -> None:
        for widget in self.config_widgets:
            widget.setEnabled(True)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        if self.logger_thread is not None:
            self.logger_thread.deleteLater()
        self.logger_thread = None
        self._on_connection_state("stopped")

    def _on_connection_state(self, state: str) -> None:
        labels = {
            "connecting": "Connecting…",
            "connected": "Connected / Logging",
            "reconnecting": "Reconnecting…",
            "stopped": "Disconnected",
        }
        self.status_card.set_value(labels.get(state, state.title()))
        if state == "connected":
            self.header_status.setText("LIVE / LOGGING")
            self.header_status.setObjectName("HeaderStatusConnected")
        elif state in {"connecting", "reconnecting"}:
            self.header_status.setText(state.upper())
            self.header_status.setObjectName("HeaderStatusWarning")
        else:
            self.header_status.setText("DISCONNECTED")
            self.header_status.setObjectName("HeaderStatusDisconnected")
        # Re-polish after changing object name so the stylesheet updates.
        self.header_status.style().unpolish(self.header_status)
        self.header_status.style().polish(self.header_status)

    def _on_connected_info(self, unit: str, port: str) -> None:
        self.current_unit = unit
        self.pressure_unit.setText(unit)
        self.port_card.set_value(port)

    def _on_reading(self, timestamp_text: str, pressure_obj: object, unit: str, csv_path: str) -> None:
        try:
            timestamp = datetime.fromisoformat(timestamp_text)
        except ValueError:
            timestamp = datetime.now().astimezone()
        pressure = pressure_obj if isinstance(pressure_obj, (float, int)) else None
        pressure = float(pressure) if pressure is not None else None
        self.current_unit = unit
        self.last_csv_path = csv_path
        self.session_samples += 1
        if pressure is None:
            self.session_errors += 1
            self.pressure_value.setText("READ ERROR")
            self.pressure_value.setProperty("error", True)
        else:
            self.session_pressures.append(pressure)
            self.pressure_value.setText(format_pressure_value(pressure, unit))
            self.pressure_value.setProperty("error", False)
        self.pressure_value.style().unpolish(self.pressure_value)
        self.pressure_value.style().polish(self.pressure_value)
        self.pressure_unit.setText(unit)
        self.last_read_card.set_value(timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S"))
        self.file_card.set_value(Path(csv_path).name)
        self.history.append_live(PressurePoint(timestamp, pressure, unit))
        self.memory_card.set_value(f"{len(self.history.points):,} rows")
        self._update_session_stats()
        self.graph.refresh_data(preserve_view=True)
        self._update_rate_of_change()

    def _update_session_stats(self) -> None:
        self.samples_card.set_value(f"{self.session_samples:,}")
        self.errors_card.set_value(f"{self.session_errors:,}")
        if not self.session_pressures:
            self.min_card.set_value("—")
            self.max_card.set_value("—")
            self.average_card.set_value("—")
            return
        minimum = min(self.session_pressures)
        maximum = max(self.session_pressures)
        average = sum(self.session_pressures) / len(self.session_pressures)
        unit = self.current_unit
        self.min_card.set_value(f"{format_pressure_value(minimum, unit)} {unit}")
        self.max_card.set_value(f"{format_pressure_value(maximum, unit)} {unit}")
        self.average_card.set_value(f"{format_pressure_value(average, unit)} {unit}")

    def _update_rate_of_change(self, *_args) -> None:
        """Show average pressure change per minute over the selected live window."""
        if not hasattr(self, "rate_card") or not hasattr(self, "graph"):
            return

        window_text = self.graph.window_combo.currentText()
        selected = self.graph.WINDOW_OPTIONS.get(window_text)
        if not self.history.points:
            self.rate_card.set_value(f"— · {window_text}")
            return

        # Match the graph's live-window anchor exactly, including a newest row
        # whose pressure is invalid/blank. That prevents a read error from
        # quietly extending a 15-minute or 1-hour rate window farther back.
        window_end = self.history.points[-1].timestamp
        cutoff = None if selected is None else window_end - selected

        valid_points = [
            point
            for point in self.history.points
            if point.pressure is not None
            and math.isfinite(point.pressure)
            and (cutoff is None or point.timestamp >= cutoff)
            and point.timestamp <= window_end
        ]
        if len(valid_points) < 2:
            self.rate_card.set_value(f"— · {window_text}")
            return

        latest = valid_points[-1]
        unit = latest.unit or self.current_unit

        # Do not mix readings recorded in different pressure units. The newest
        # valid point in the selected window establishes the calculation unit.
        window_points = [
            point
            for point in valid_points
            if point.unit.strip().upper() == unit.strip().upper()
        ]
        if len(window_points) < 2:
            self.rate_card.set_value(f"— · {window_text}")
            return

        first = window_points[0]
        last = window_points[-1]
        elapsed_minutes = (last.timestamp - first.timestamp).total_seconds() / 60.0
        if elapsed_minutes <= 0.0:
            self.rate_card.set_value(f"— · {window_text}")
            return

        rate = (float(last.pressure) - float(first.pressure)) / elapsed_minutes
        self.rate_card.set_value(
            f"{format_rate_value(rate, unit)} {unit}/min · {window_text}"
        )

    # ---------- History / graph ----------

    def _load_older_history(self, target_time: datetime) -> None:
        loaded = self.history.load_older_until(target_time)
        if not loaded:
            return
        total_rows = sum(rows for _, rows in loaded)
        first_name = loaded[-1][0].name
        self.add_log(
            f"Loaded {total_rows:,} older history rows from {len(loaded)} file(s), back through {first_name}.",
            "muted",
        )
        self.memory_card.set_value(f"{len(self.history.points):,} rows")
        self.graph.refresh_data(preserve_view=True)
        self._update_rate_of_change()
    # ---------- Logging / styling / close ----------

    def add_log(self, message: str, level: str = "info") -> None:
        colors = {
            "info": "#334155",
            "success": "#15803d",
            "error": "#dc2626",
            "warning": "#a16207",
            "muted": "#64748b",
        }
        color = colors.get(level, colors["info"])
        safe = (
            message.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        self.log_text.append(f'<span style="color:{color};">{safe}</span>')
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _apply_light_theme(self) -> None:
        # Keep child labels/widgets transparent by default.  Applying a background
        # to every QWidget is what caused the gray rectangles behind text in the
        # previous version on Windows.
        self.setStyleSheet(
            """
            QMainWindow, QWidget#CentralWidget {
                background-color: #f5f7fb;
            }
            QWidget {
                color: #1e293b;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 10pt;
            }
            QLabel {
                background-color: transparent;
            }
            QGroupBox {
                border: 1px solid #cbd5e1;
                border-radius: 9px;
                margin-top: 0px;
                padding: 26px 9px 9px 9px;
                font-weight: 600;
                color: #0f172a;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: border;
                subcontrol-position: top left;
                position: relative;
                left: 11px;
                top: 7px;
                padding: 0px 2px;
                background-color: transparent;
            }
            QLabel#AppTitle {
                font-size: 18pt;
                font-weight: 700;
                color: #0f172a;
            }
            QLabel#AppSubtitle {
                color: #64748b;
                font-size: 9pt;
            }
            QLabel#HeaderStatusDisconnected, QLabel#HeaderStatusConnected, QLabel#HeaderStatusWarning {
                border-radius: 7px;
                padding: 6px 11px;
                min-width: 108px;
                font-weight: 700;
                letter-spacing: 0.6px;
            }
            QLabel#HeaderStatusDisconnected {
                background: #ffffff;
                color: #475569;
                border: 1px solid #cbd5e1;
            }
            QLabel#HeaderStatusConnected {
                background: #f0fdf4;
                color: #166534;
                border: 1px solid #86efac;
            }
            QLabel#HeaderStatusWarning {
                background: #fffbeb;
                color: #92400e;
                border: 1px solid #fcd34d;
            }
            QLabel#SectionTitle {
                font-size: 11pt;
                font-weight: 650;
                color: #0f172a;
            }
            QLabel#HintLabel {
                color: #64748b;
                font-size: 8pt;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 5px 7px;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
                border: 1px solid #3b82f6;
            }
            QComboBox {
                padding-right: 31px;
            }
            QComboBox::drop-down {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 28px;
                background-color: #f1f5f9;
                border: none;
                border-left: 1px solid #cbd5e1;
                border-top-right-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QComboBox::drop-down:hover {
                background-color: #e2e8f0;
            }
            QComboBox::drop-down:pressed {
                background-color: #dbeafe;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
            }
            QSpinBox, QDoubleSpinBox {
                padding-right: 31px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 28px;
                background-color: #f1f5f9;
                border: none;
                border-left: 1px solid #cbd5e1;
                border-bottom: 1px solid #dbe3ed;
                border-top-right-radius: 5px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 28px;
                background-color: #f1f5f9;
                border: none;
                border-left: 1px solid #cbd5e1;
                border-top: 1px solid #dbe3ed;
                border-bottom-right-radius: 5px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #e2e8f0;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #dbeafe;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
            }
            QComboBox QAbstractItemView, QListView#ComboPopupView {
                background-color: #ffffff;
                color: #1e293b;
                border: none;
                outline: 0;
                padding: 3px;
                selection-background-color: #dbeafe;
                selection-color: #1e3a8a;
            }
            QComboBox QAbstractItemView::item, QListView#ComboPopupView::item {
                min-height: 26px;
                padding: 3px 7px;
                border-radius: 4px;
            }
            QComboBox QAbstractItemView::item:hover, QListView#ComboPopupView::item:hover {
                background-color: #eff6ff;
            }
            QPushButton {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 5px 9px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f8fafc;
                border-color: #94a3b8;
            }
            QPushButton:pressed {
                background-color: #f1f5f9;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background-color: #ffffff;
                border-color: #e2e8f0;
            }
            QPushButton#PrimaryButton {
                background-color: #2563eb;
                border-color: #1d4ed8;
                color: white;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton#PrimaryButton:disabled {
                background-color: #bfdbfe;
                border-color: #bfdbfe;
                color: #ffffff;
            }
            QPushButton#DangerButton {
                background-color: #fff7f7;
                color: #991b1b;
                border-color: #fca5a5;
            }
            QPushButton#DangerButton:hover {
                background-color: #fee2e2;
            }
            QPushButton#DangerButton:disabled {
                background-color: #ffffff;
                color: #94a3b8;
                border-color: #e2e8f0;
            }
            QFrame#PressureHero {
                background-color: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 9px;
            }
            QLabel#HeroCaption {
                color: #64748b;
                font-size: 8pt;
                font-weight: 700;
                letter-spacing: 0.8px;
            }
            QLabel#PressureValue {
                color: #0f172a;
                font-size: 28pt;
                font-weight: 700;
                padding: 0;
            }
            QLabel#PressureValue[error="true"] {
                color: #dc2626;
                font-size: 21pt;
            }
            QLabel#PressureUnit {
                color: #2563eb;
                font-size: 11pt;
                font-weight: 600;
            }
            QFrame#StatCard {
                background-color: #ffffff;
                border: 1px solid #dbe3ed;
                border-radius: 7px;
            }
            QLabel#StatTitle {
                color: #64748b;
                font-size: 7.5pt;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            QLabel#StatValue {
                color: #1e293b;
                font-size: 9.5pt;
                font-weight: 600;
            }
            QToolBar#GraphToolbar, QWidget#GraphToolbar {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                spacing: 2px;
            }
            QToolBar#GraphToolbar QToolButton {
                background-color: #ffffff;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 3px;
                min-width: 24px;
                min-height: 24px;
            }
            QToolBar#GraphToolbar QToolButton:hover {
                background-color: #f1f5f9;
                border-color: #cbd5e1;
            }
            QToolTip {
                background-color: #ffffff;
                color: #1e293b;
                border: 1px solid #94a3b8;
                padding: 4px;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #ffffff;
                border: none;
                margin: 0;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #cbd5e1;
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #94a3b8;
            }
            """
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        if self.logger_thread is not None and self.logger_thread.isRunning():
            self.logger_thread.stop()
            if not self.logger_thread.wait(2500):
                answer = QMessageBox.question(
                    self,
                    "Logger still stopping",
                    "The serial logger has not stopped yet. Close the application anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer == QMessageBox.StandardButton.No:
                    event.ignore()
                    return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName("Granville-Phillips 392 Pressure Logger")
    app.setOrganizationName("GP392Tools")
    window = MainWindow()
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())