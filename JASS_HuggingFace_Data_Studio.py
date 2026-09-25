#!/usr/bin/env python3
"""
JASS Hugging Face Data Studio v0.2
Local-first Hugging Face dataset research workspace.

v0.2 adds:
- Dataset intelligence and Hub Viewer API metadata
- Split/config discovery
- Local Parquet statistics
- Clean structured record inspector (binary payloads hidden)
- Proper audio controls for embedded WAV/RIFF bytes
- Column type/role detection
- Safer bounded preview
"""

import csv
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QTextEdit, QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QTabWidget, QMessageBox, QFileDialog, QStatusBar, QHeaderView,
    QProgressBar, QGroupBox, QGridLayout, QSlider, QAbstractItemView,
    QScrollArea, QFrame
)

APP_NAME = "JASS Hugging Face Data Studio"
APP_VERSION = "v0.2"
DEFAULT_DATASET = (
    "https://huggingface.co/datasets/"
    "dianavdavidson/Vaani-garo-mizo-majority-lg-English-no-transcript0"
)
HF_API = "https://datasets-server.huggingface.co"


def parse_repo(value):
    s = value.strip().rstrip("/")
    s = re.sub(r"^https?://huggingface\.co/datasets/", "", s)
    s = s.split("/tree/")[0].split("/blob/")[0]
    if s.startswith("datasets/"):
        s = s[8:]
    if "/" not in s:
        raise ValueError("Enter a Hugging Face dataset URL or org/dataset ID.")
    return s


def compact(value, limit=180):
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return f"<binary: {len(value):,} bytes>"
    if isinstance(value, dict):
        safe = {}
        for k, v in value.items():
            if isinstance(v, (bytes, bytearray)):
                safe[k] = f"<binary: {len(v):,} bytes>"
            else:
                safe[k] = v
        try:
            value = json.dumps(safe, ensure_ascii=False)
        except Exception:
            value = str(safe)
    elif isinstance(value, (list, tuple)):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    s = str(value).replace("\n", " ")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def safe_json(value):
    if isinstance(value, (bytes, bytearray)):
        return f"<binary: {len(value):,} bytes>"
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    return value


def media_kind(name, value):
    n = name.lower()
    if "audio" in n or "speech" in n or "sound" in n:
        return "audio"
    if "image" in n or "picture" in n or "photo" in n:
        return "image"
    if "video" in n or "movie" in n:
        return "video"
    if isinstance(value, dict):
        if any(k in value for k in ("bytes", "array")) and "audio" in n:
            return "audio"
    return None


def format_seconds(seconds):
    try:
        s = float(seconds)
        m, sec = divmod(int(s), 60)
        return f"{m:02d}:{sec:02d}"
    except Exception:
        return "00:00"


class Loader(QThread):
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, repo_id, filename=None, parent=None):
        super().__init__(parent)
        self.repo_id = repo_id
        self.filename = filename

    def run(self):
        try:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            info = api.dataset_info(self.repo_id)
            files = api.list_repo_files(self.repo_id, repo_type="dataset")
            parquets = [f for f in files if f.lower().endswith(".parquet")]
            local_path = None
            if self.filename:
                local_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    repo_type="dataset",
                )
            hub = self.hub_calls()
            self.loaded.emit({
                "info": info,
                "files": files,
                "parquets": parquets,
                "selected": self.filename,
                "local_path": local_path,
                "hub": hub,
            })
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def hub_calls(self):
        result = {}
        for endpoint, key in (
            (f"/splits?dataset={urllib.parse.quote(self.repo_id, safe='')}", "splits"),
            (f"/size?dataset={urllib.parse.quote(self.repo_id, safe='')}", "size"),
        ):
            try:
                with urllib.request.urlopen(HF_API + endpoint, timeout=15) as r:
                    result[key] = json.loads(r.read().decode("utf-8"))
            except Exception as e:
                result[key] = {"error": str(e)}
        return result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1580, 950)
        self.setMinimumSize(1150, 720)

        self.repo_id = ""
        self.info = None
        self.files = []
        self.parquets = []
        self.df = None
        self.table_rows = []
        self.visible_rows = []
        self.pf = None
        self.parquet_path = None
        self.current_filename = ""
        self.current_audio = None
        self.current_image = None
        self.current_image_name = 'image.png'
        self.temp_files = []
        self.player = None
        self.audio_output = None

        self.build_ui()
        self.apply_theme()
        self.url_edit.setText(DEFAULT_DATASET)

    def build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 11, 14, 9)
        root.setSpacing(9)

        title = QLabel(f"{APP_NAME}")
        title.setObjectName("title")
        subtitle = QLabel(
            "A local-first research workspace for Hugging Face datasets — "
            "metadata, splits, schema, statistics, records and multimodal data."
        )
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        bar = QHBoxLayout()
        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("Hugging Face dataset URL or org/dataset")
        self.load_btn = QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self.load_dataset)
        bar.addWidget(self.url_edit, 1)
        bar.addWidget(self.load_btn)
        root.addLayout(bar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        root.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.nav = QTreeWidget()
        self.nav.setHeaderHidden(True)
        self.nav.itemClicked.connect(self.nav_clicked)
        ll.addWidget(self.nav)
        splitter.addWidget(left)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()

        self.overview = QTextEdit()
        self.overview.setReadOnly(True)

        self.stats = QTextEdit()
        self.stats.setReadOnly(True)

        self.schema = QTableWidget(0, 4)
        self.schema.setHorizontalHeaderLabels(["Column", "Detected Type", "Role", "Sample"])
        self.schema.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.schema.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.schema.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.schema.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)

        data_tab = QWidget()
        dl = QVBoxLayout(data_tab)
        controls = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search loaded preview…")
        self.search_edit.textChanged.connect(self.filter_table)
        self.column_combo = QComboBox()
        self.column_combo.addItem("All columns")
        self.column_combo.currentTextChanged.connect(self.filter_table)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(25, 1000)
        self.rows_spin.setValue(100)
        self.rows_spin.setPrefix("Rows: ")
        self.refresh_btn = QPushButton("Load Preview")
        self.refresh_btn.clicked.connect(self.load_preview)
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self.export_csv)
        controls.addWidget(self.search_edit, 1)
        controls.addWidget(self.column_combo)
        controls.addWidget(self.rows_spin)
        controls.addWidget(self.refresh_btn)
        controls.addWidget(self.export_btn)
        dl.addLayout(controls)

        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.record_selected)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        dl.addWidget(self.table)

        self.files_view = QTreeWidget()
        self.files_view.setHeaderLabels(["Repository file", "Type"])
        self.files_view.itemDoubleClicked.connect(self.file_double_clicked)

        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.stats, "Statistics")
        self.tabs.addTab(self.schema, "Schema")
        self.tabs.addTab(data_tab, "Data Explorer")
        self.tabs.addTab(self.files_view, "Files")
        cl.addWidget(self.tabs)
        splitter.addWidget(center)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Selected Record")
        bl = QVBoxLayout(box)
        self.record_label = QLabel("Select a row to inspect it.")
        self.record_label.setWordWrap(True)
        bl.addWidget(self.record_label)
        self.record_view = QTextEdit()
        self.record_view.setReadOnly(True)
        bl.addWidget(self.record_view, 1)
        rl.addWidget(box, 1)

        media = QGroupBox("Multimodal Preview")
        ml = QVBoxLayout(media)
        self.media_info = QLabel(
            "Select a record. JASS will recognize common audio, image and video fields."
        )
        self.media_info.setWordWrap(True)
        ml.addWidget(self.media_info)

        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(210)
        self.image_label.setFrameShape(QFrame.StyledPanel)
        self.image_label.setScaledContents(False)
        ml.addWidget(self.image_label)

        image_buttons = QHBoxLayout()
        self.save_image_btn = QPushButton("Save Image")
        self.save_image_btn.setEnabled(False)
        self.save_image_btn.clicked.connect(self.save_image)
        image_buttons.addWidget(self.save_image_btn)
        ml.addLayout(image_buttons)

        self.audio_slider = QSlider(Qt.Horizontal)
        self.audio_slider.setRange(0, 0)
        self.audio_slider.sliderMoved.connect(self.seek_audio)
        self.audio_time = QLabel("00:00 / 00:00")
        self.audio_time.setAlignment(Qt.AlignCenter)
        ml.addWidget(self.audio_slider)
        ml.addWidget(self.audio_time)

        buttons = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.stop_btn = QPushButton("■ Stop")
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_audio)
        self.stop_btn.clicked.connect(self.stop_audio)
        buttons.addWidget(self.play_btn)
        buttons.addWidget(self.stop_btn)
        ml.addLayout(buttons)
        rl.addWidget(media)

        splitter.addWidget(right)
        splitter.setSizes([225, 900, 410])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready — enter a Hugging Face dataset URL.")

    def apply_theme(self):
        self.setStyleSheet("""
            QWidget { background:#0d1117; color:#e6edf3; font-size:13px; }
            QLineEdit, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTreeWidget {
                background:#111821; border:1px solid #303946; border-radius:7px;
                padding:5px; selection-background-color:#244b6b;
            }
            QPushButton {
                background:#1f6feb; color:white; border:0; border-radius:7px;
                padding:8px 14px; font-weight:600;
            }
            QPushButton:hover { background:#388bfd; }
            QPushButton:disabled { background:#30363d; color:#8b949e; }
            QTabBar::tab { background:#161b22; padding:9px 16px; margin-right:2px; }
            QTabBar::tab:selected { background:#21262d; border-bottom:2px solid #58a6ff; }
            QHeaderView::section { background:#161b22; color:#c9d1d9; padding:6px; border:0; }
            QTreeWidget::item { padding:7px 4px; }
            QTreeWidget::item:selected { background:#1c3550; }
            QGroupBox { border:1px solid #303946; border-radius:8px; margin-top:10px; padding-top:12px; }
            QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; color:#79c0ff; }
            QLabel#title { font-size:25px; font-weight:700; color:#f0f6fc; }
            QLabel#subtitle { color:#8b949e; font-size:13px; }
            QProgressBar { border:0; height:4px; background:#21262d; }
            QProgressBar::chunk { background:#1f6feb; }
        """)

    def load_dataset(self):
        try:
            self.repo_id = parse_repo(self.url_edit.text())
        except ValueError as e:
            QMessageBox.warning(self, "Invalid dataset", str(e))
            return
        self.set_busy(True)
        self.nav.clear()
        self.overview.clear()
        self.stats.clear()
        self.schema.setRowCount(0)
        self.files_view.clear()
        self.statusBar().showMessage(f"Loading {self.repo_id}…")
        self.loader = Loader(self.repo_id, parent=self)
        self.loader.loaded.connect(self.dataset_loaded)
        self.loader.failed.connect(self.load_failed)
        self.loader.start()

    def set_busy(self, busy):
        self.load_btn.setEnabled(not busy)
        self.progress.setVisible(busy)

    def load_failed(self, msg):
        self.set_busy(False)
        self.statusBar().showMessage("Load failed")
        QMessageBox.critical(self, "Hugging Face error", msg)

    def dataset_loaded(self, payload):
        self.set_busy(False)
        self.info = payload["info"]
        self.files = payload["files"]
        self.parquets = payload["parquets"]
        self.hub = payload.get("hub", {})

        root = QTreeWidgetItem(["DATASET"])
        root.setExpanded(True)
        for name in ["Overview", "Statistics", "Schema", "Data Explorer", "Files"]:
            QTreeWidgetItem(root, [name])
        self.nav.addTopLevelItem(root)

        self.overview.setHtml(self.make_overview_html())
        self.populate_files()
        self.update_stats_from_hub()

        if self.parquets:
            self.download_parquet(self.parquets[0])
        else:
            self.statusBar().showMessage("Dataset loaded — no Parquet file found.")
            self.tabs.setCurrentIndex(0)

    def make_overview_html(self):
        tags = getattr(self.info, "tags", None) or []
        sha = getattr(self.info, "sha", "") or ""
        card = getattr(self.info, "card_data", None)
        return f"""
        <h2>{self.repo_id}</h2>
        <p><b>Author:</b> {getattr(self.info, 'author', '') or '—'}</p>
        <p><b>Last modified:</b> {getattr(self.info, 'last_modified', '') or '—'}</p>
        <p><b>Revision:</b> <code>{sha}</code></p>
        <p><b>Repository files:</b> {len(self.files)}
           &nbsp;&nbsp; <b>Parquet:</b> {len(self.parquets)}</p>
        <p><b>Tags:</b> {compact(tags, 700)}</p>
        <h3>JASS workspace</h3>
        <p>This dataset is inspected locally. Selected files are cached through the
        Hugging Face Hub client; the source repository is not modified.</p>
        <h3>Dataset card metadata</h3>
        <pre>{compact(card, 5000) if card else 'No card_data exposed by the Hub API.'}</pre>
        """

    def update_stats_from_hub(self):
        splits = self.hub.get("splits", {})
        size = self.hub.get("size", {})
        html = "<h2>Dataset Intelligence</h2>"
        if "error" not in splits:
            html += "<h3>Configurations / Splits</h3><table>"
            html += "<tr><th>Config</th><th>Split</th><th>Rows</th></tr>"
            rows = splits.get("splits", [])
            if isinstance(rows, list):
                for item in rows:
                    html += (
                        f"<tr><td>{item.get('config','')}</td>"
                        f"<td>{item.get('split','')}</td>"
                        f"<td>{item.get('num_rows','—')}</td></tr>"
                    )
            html += "</table>"
        else:
            html += "<p>Hub split information unavailable.</p>"
        if "error" not in size:
            html += f"<h3>Hub size information</h3><pre>{compact(size, 3000)}</pre>"
        self.stats.setHtml(html)

    def populate_files(self):
        self.files_view.clear()
        for f in self.files:
            typ = "Parquet" if f.lower().endswith(".parquet") else (
                Path(f).suffix.lstrip(".").upper() or "File"
            )
            QTreeWidgetItem(self.files_view, [f, typ])

    def file_double_clicked(self, item, col):
        name = item.text(0)
        if name.lower().endswith(".parquet"):
            self.download_parquet(name)
        else:
            QMessageBox.information(
                self, "File",
                f"{name}\n\nv0.2 opens Parquet files for structured exploration."
            )

    def download_parquet(self, filename):
        self.statusBar().showMessage(f"Caching {filename}…")
        self.set_busy(True)
        self.loader = Loader(self.repo_id, filename=filename, parent=self)
        self.loader.loaded.connect(self.parquet_loaded)
        self.loader.failed.connect(self.load_failed)
        self.loader.start()

    def parquet_loaded(self, payload):
        self.set_busy(False)
        if not payload.get("local_path"):
            return
        self.parquet_path = payload["local_path"]
        self.current_filename = payload["selected"]
        try:
            import pyarrow.parquet as pq
            self.pf = pq.ParquetFile(self.parquet_path)
            self.populate_schema()
            self.calculate_local_stats()
            self.load_preview()
            self.tabs.setCurrentIndex(3)
            self.statusBar().showMessage(
                f"Ready: {self.current_filename} — "
                f"{self.pf.metadata.num_rows:,} rows, "
                f"{self.pf.metadata.num_row_groups:,} row groups"
            )
        except Exception as e:
            QMessageBox.critical(self, "Parquet error", f"{type(e).__name__}: {e}")

    def detect_role(self, field):
        n = field.name.lower()
        t = str(field.type).lower()
        if any(x in n for x in ("audio", "speech", "sound")):
            return "Audio"
        if any(x in n for x in ("image", "photo", "picture")):
            return "Image"
        if any(x in n for x in ("video", "movie")):
            return "Video"
        if "string" in t:
            return "Text / categorical"
        if "list" in t or "struct" in t or "map" in t:
            return "Nested"
        if any(x in t for x in ("int", "float", "double", "decimal")):
            return "Numeric"
        if "bool" in t:
            return "Boolean"
        return "Other"

    def populate_schema(self):
        fields = list(self.pf.schema_arrow)
        self.schema.setRowCount(len(fields))
        self.column_combo.clear()
        self.column_combo.addItem("All columns")
        for i, field in enumerate(fields):
            self.schema.setItem(i, 0, QTableWidgetItem(field.name))
            self.schema.setItem(i, 1, QTableWidgetItem(str(field.type)))
            self.schema.setItem(i, 2, QTableWidgetItem(self.detect_role(field)))
            try:
                arr = self.pf.read_row_group(0, columns=[field.name]).column(0)
                sample = arr[0].as_py() if len(arr) else ""
            except Exception:
                sample = ""
            self.schema.setItem(i, 3, QTableWidgetItem(compact(sample)))
            self.column_combo.addItem(field.name)

    def calculate_local_stats(self):
        try:
            fields = list(self.pf.schema_arrow)
            nulls = {f.name: 0 for f in fields}
            nonnull = {f.name: 0 for f in fields}
            distinct = {f.name: set() for f in fields}
            max_distinct = 50000
            for rg in range(self.pf.num_row_groups):
                table = self.pf.read_row_group(rg)
                for f in fields:
                    arr = table.column(f.name)
                    try:
                        py = arr.to_pylist()
                    except Exception:
                        continue
                    for v in py:
                        if v is None:
                            nulls[f.name] += 1
                        else:
                            nonnull[f.name] += 1
                            if len(distinct[f.name]) < max_distinct:
                                try:
                                    distinct[f.name].add(str(v))
                                except Exception:
                                    pass
            total = self.pf.metadata.num_rows
            size_mb = os.path.getsize(self.parquet_path) / (1024 * 1024)
            html = f"""
            <h2>Local Parquet Statistics</h2>
            <p><b>File:</b> {self.current_filename}</p>
            <p><b>Rows:</b> {total:,}
               &nbsp;&nbsp; <b>Columns:</b> {len(fields)}
               &nbsp;&nbsp; <b>Row groups:</b> {self.pf.num_row_groups}
               &nbsp;&nbsp; <b>Cached size:</b> {size_mb:.2f} MB</p>
            <table><tr><th>Column</th><th>Non-null</th><th>Null</th><th>Distinct</th></tr>
            """
            for f in fields:
                d = len(distinct[f.name])
                suffix = " (capped)" if d >= max_distinct else ""
                html += (
                    f"<tr><td>{f.name}</td><td>{nonnull[f.name]:,}</td>"
                    f"<td>{nulls[f.name]:,}</td><td>{d:,}{suffix}</td></tr>"
                )
            html += "</table>"
            self.stats.setHtml(html)
        except Exception as e:
            self.stats.setPlainText(f"Statistics error: {type(e).__name__}: {e}")

    def load_preview(self):
        if self.pf is None:
            return
        try:
            import pandas as pd
            n = self.rows_spin.value()
            frames = []
            total = 0
            for rg in range(self.pf.num_row_groups):
                df = self.pf.read_row_group(rg).to_pandas()
                frames.append(df)
                total += len(df)
                if total >= n:
                    break
            self.df = pd.concat(frames, ignore_index=True).head(n) if frames else pd.DataFrame()
            self.table_rows = self.df.to_dict(orient="records")
            self.render_table(self.table_rows)
            self.statusBar().showMessage(
                f"Preview loaded: {len(self.df):,} rows from {self.current_filename}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Preview error", f"{type(e).__name__}: {e}")

    def render_table(self, rows):
        self.visible_rows = rows
        self.table.clear()
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        cols = list(rows[0].keys())
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(len(rows))
        for r, record in enumerate(rows):
            for c, col in enumerate(cols):
                self.table.setItem(r, c, QTableWidgetItem(compact(record.get(col), 240)))
        self.table.resizeRowsToContents()

    def filter_table(self):
        if not self.table_rows:
            return
        q = self.search_edit.text().strip().casefold()
        col = self.column_combo.currentText()
        rows = []
        for rec in self.table_rows:
            if col != "All columns":
                ok = q in compact(rec.get(col), 1000).casefold()
            else:
                ok = q in " ".join(compact(v, 1000) for v in rec.values()).casefold()
            if ok:
                rows.append(rec)
        self.render_table(rows)

    def record_selected(self):
        items = self.table.selectedItems()
        if not items or self.df is None:
            return
        row = items[0].row()
        if row >= len(self.visible_rows):
            return
        rec = self.visible_rows[row]
        self.record_label.setText(f"Record {row + 1} — {self.current_filename}")
        self.record_view.setPlainText(self.pretty_record(rec))
        self.detect_media(rec)

    def pretty_record(self, rec):
        lines = []
        for key, value in rec.items():
            kind = media_kind(key, value)
            if kind == "audio":
                lines.append(f"🎵 {key}\n  [Audio data hidden — use Media controls]")
            elif kind == "image":
                lines.append(f"🖼 {key}\n  [Image field detected]")
            elif kind == "video":
                lines.append(f"🎬 {key}\n  [Video field detected]")
            elif isinstance(value, dict) and any(
                isinstance(v, (bytes, bytearray)) for v in value.values()
            ):
                cleaned = {}
                for k, v in value.items():
                    cleaned[k] = f"<binary: {len(v):,} bytes>" if isinstance(v, (bytes, bytearray)) else v
                lines.append(f"{key}:\n{json.dumps(cleaned, ensure_ascii=False, indent=2, default=str)}")
            else:
                lines.append(f"{key}:\n{json.dumps(safe_json(value), ensure_ascii=False, indent=2, default=str)}")
        return "\n\n".join(lines)

    def detect_media(self, rec):
        self.stop_audio()
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        self.current_audio = None
        self.current_image = None
        self.current_image_name = "image.png"
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("No image selected")
        hints = []

        for key, value in rec.items():
            kind = media_kind(key, value)

            if kind == "audio":
                data = None
                name = key
                if isinstance(value, dict):
                    raw = value.get("bytes") or value.get("data")
                    if isinstance(raw, (bytes, bytearray)):
                        data = bytes(raw)
                    name = value.get("path") or value.get("filename") or key
                elif isinstance(value, (bytes, bytearray)):
                    data = bytes(value)
                if data:
                    self.current_audio = (data, str(name))
                    hints.append(f"🎵 {key}: embedded audio ({len(data):,} bytes)")
                else:
                    hints.append(f"🎵 {key}: audio field detected")

            elif kind == "image":
                data = None
                name = key
                if isinstance(value, dict):
                    raw = value.get("bytes") or value.get("data")
                    if isinstance(raw, (bytes, bytearray)):
                        data = bytes(raw)
                    name = value.get("path") or value.get("filename") or key
                elif isinstance(value, (bytes, bytearray)):
                    data = bytes(value)
                elif isinstance(value, str) and Path(value).is_file():
                    try:
                        data = Path(value).read_bytes()
                    except Exception:
                        data = None

                if data:
                    from PySide6.QtGui import QImage
                    image = QImage.fromData(data)
                    if not image.isNull():
                        self.current_image = data
                        self.current_image_name = str(name)
                        pixmap = QPixmap.fromImage(image)
                        scaled = pixmap.scaled(
                            self.image_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.image_label.setPixmap(scaled)
                        self.image_label.setText("")
                        self.save_image_btn.setEnabled(True)
                        hints.append(
                            f"🖼 {key}: image preview "
                            f"({image.width()} × {image.height()})"
                        )
                    else:
                        hints.append(f"🖼 {key}: image detected but could not decode")
                else:
                    hints.append(f"🖼 {key}: image field detected")

            elif kind == "video":
                hints.append(f"🎬 {key}: video field detected")

        if self.current_audio:
            self.play_btn.setEnabled(True)

        if hints:
            self.media_info.setText("\n".join(hints))
        else:
            self.media_info.setText("No supported media payload detected in this record.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            from PySide6.QtGui import QImage
            image = QImage.fromData(self.current_image)
            if not image.isNull():
                pixmap = QPixmap.fromImage(image)
                self.image_label.setPixmap(
                    pixmap.scaled(
                        self.image_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )

    def save_image(self):
        if not self.current_image:
            return
        suffix = Path(self.current_image_name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
            suffix = ".png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Image",
            f"selected_image{suffix}",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)"
        )
        if not path:
            return
        try:
            Path(path).write_bytes(self.current_image)
            self.statusBar().showMessage(f"Image saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save image", str(e))

    def play_audio(self):
        if not self.current_audio:
            return
        try:
            from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
            data, name = self.current_audio
            suffix = Path(name).suffix.lower()
            if not suffix:
                suffix = ".wav" if data[:4] == b"RIFF" else ".bin"
            fd, path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            Path(path).write_bytes(data)
            self.temp_files.append(path)

            self.audio_output = QAudioOutput()
            self.audio_output.setVolume(1.0)
            self.player = QMediaPlayer()
            self.player.setAudioOutput(self.audio_output)
            self.player.positionChanged.connect(self.audio_position)
            self.player.durationChanged.connect(self.audio_duration)
            self.player.mediaStatusChanged.connect(self.audio_status)
            self.player.setSource(QUrl.fromLocalFile(path))
            self.player.play()
            self.play_btn.setText("❚❚ Playing")
            self.stop_btn.setEnabled(True)
            self.media_info.setText(f"Playing {Path(path).name}")
        except Exception as e:
            QMessageBox.warning(self, "Audio playback", f"Could not play this audio.\n\n{e}")

    def audio_position(self, position):
        self.audio_slider.setValue(position)
        self.update_audio_time()

    def audio_duration(self, duration):
        self.audio_slider.setRange(0, max(0, duration))
        self.update_audio_time()

    def update_audio_time(self):
        pos = self.audio_slider.value() / 1000
        dur = self.audio_slider.maximum() / 1000
        self.audio_time.setText(f"{format_seconds(pos)} / {format_seconds(dur)}")

    def audio_status(self, status):
        if str(status).endswith("EndOfMedia"):
            self.play_btn.setText("▶ Play")
            self.stop_btn.setEnabled(False)

    def seek_audio(self, position):
        if self.player:
            self.player.setPosition(position)

    def stop_audio(self):
        if self.player:
            self.player.stop()
            self.player.deleteLater()
            self.player = None
        self.play_btn.setText("▶ Play")
        self.stop_btn.setEnabled(False)
        self.audio_slider.setValue(0)
        self.audio_time.setText("00:00 / 00:00")

    def export_csv(self):
        if self.df is None or self.df.empty:
            QMessageBox.information(self, "Export", "Load a dataset preview first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "hf_preview.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            self.df.to_csv(path, index=False)
            self.statusBar().showMessage(f"Exported {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def nav_clicked(self, item, column):
        mapping = {
            "Overview": 0,
            "Statistics": 1,
            "Schema": 2,
            "Data Explorer": 3,
            "Files": 4,
        }
        if item.text(0) in mapping:
            self.tabs.setCurrentIndex(mapping[item.text(0)])


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("DejaVu Sans", 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
