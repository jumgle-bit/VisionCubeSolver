"""PySide6 desktop interface for scanning and solving a cube."""

from __future__ import annotations

import sys
from threading import Event

import cv2
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vision_cube_solver.app.session import CubeSession
from vision_cube_solver.cube.enums import (
    COLOR_FACES,
    DISPLAY_COLORS,
    FACE_COLORS,
    FACE_ORDER,
    TOP_REFERENCE,
    CubeColor,
    Face,
)
from vision_cube_solver.cube.model import FaceGrid
from vision_cube_solver.solver.pattern_database import PatternDatabaseManager
from vision_cube_solver.solver.two_phase import (
    SolverCancelled,
    SolverError,
    SolveResult,
    TwoPhaseIDASolver,
)
from vision_cube_solver.vision.camera import CameraError, CameraService
from vision_cube_solver.vision.sampler import GridSampler


class FaceGridWidget(QGroupBox):
    sticker_changed = Signal(object, int, object)

    def __init__(self, face: Face) -> None:
        super().__init__(f"{face.value} / {FACE_COLORS[face].value.title()}")
        self.face = face
        self.grid = FaceGrid.solved(face)
        self.captured = False
        layout = QGridLayout(self)
        self.buttons: list[QPushButton] = []
        for index in range(9):
            button = QPushButton()
            button.setFixedSize(34, 34)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, i=index: self._choose_color(i))
            layout.addWidget(button, index // 3, index % 3)
            self.buttons.append(button)
        self.set_grid(None)

    def set_grid(self, grid: FaceGrid | None) -> None:
        self.captured = grid is not None
        self.grid = grid or FaceGrid.solved(self.face)
        for index, button in enumerate(self.buttons):
            color = self.grid.colors[index]
            confidence = self.grid.confidences[index]
            shown = DISPLAY_COLORS[color] if self.captured else "#5d6470"
            border = "#e53935" if self.captured and confidence < 0.30 else "#222"
            button.setStyleSheet(
                f"background:{shown}; border:2px solid {border}; border-radius:3px;"
            )
            button.setToolTip(
                f"{color.value.title()} | confidence {confidence:.0%}"
                if self.captured
                else "Not captured"
            )
            button.setEnabled(self.captured and index != 4)

    def _choose_color(self, index: int) -> None:
        menu = QMenu(self)
        for color in CubeColor:
            action = QAction(color.value.title(), menu)
            action.triggered.connect(
                lambda _checked=False, selected=color: self.sticker_changed.emit(
                    self.face, index, selected
                )
            )
            menu.addAction(action)
        menu.exec(self.buttons[index].mapToGlobal(self.buttons[index].rect().bottomLeft()))


class SolveWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(str, int, int)

    def __init__(self, cube: object, manager: PatternDatabaseManager) -> None:
        super().__init__()
        self.cube = cube
        self.manager = manager
        self.cancel_event = Event()

    def run(self) -> None:
        try:
            self.manager.ensure_ready(
                lambda label, current, total: self.progress.emit(label, current, total)
            )
            solver = TwoPhaseIDASolver(self.manager)
            result = solver.solve(self.cube, timeout_seconds=30, cancel_event=self.cancel_event)
            self.completed.emit(result)
        except SolverCancelled:
            self.failed.emit("求解已取消")
        except (SolverError, OSError, ValueError) as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("VisionCubeSolver - 黄底绿朝前")
        self.resize(1280, 820)
        self.camera = CameraService()
        self.sampler = GridSampler()
        self.session = CubeSession()
        self.database_manager = PatternDatabaseManager()
        self.last_frame = None
        self.guide_rect = None
        self.worker: SolveWorker | None = None
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._read_camera)
        self._refresh_devices()
        self._update_state()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        title = QLabel("固定方向：黄面朝下，绿面朝向自己")
        title.setStyleSheet("font-size:18px; font-weight:600; padding:6px;")
        root_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, 1)

        capture_panel = QWidget()
        capture_layout = QVBoxLayout(capture_panel)
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        refresh = QPushButton("刷新摄像头")
        refresh.clicked.connect(self._refresh_devices)
        self.camera_button = QPushButton("启动摄像头")
        self.camera_button.clicked.connect(self._toggle_camera)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(refresh)
        device_row.addWidget(self.camera_button)
        capture_layout.addLayout(device_row)

        self.preview = QLabel("请选择并启动摄像头")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(560, 420)
        self.preview.setStyleSheet("background:#17191d; color:#ddd; border:1px solid #444;")
        capture_layout.addWidget(self.preview, 1)

        capture_controls = QHBoxLayout()
        self.center_combo = QComboBox()
        for color in CubeColor:
            self.center_combo.addItem(color.value.title(), color)
        self.center_combo.currentIndexChanged.connect(self._update_orientation_hint)
        self.confirm_button = QPushButton("确认当前面")
        self.confirm_button.clicked.connect(self._capture_face)
        self.remove_button = QPushButton("重新采集当前面")
        self.remove_button.clicked.connect(self._remove_current_face)
        capture_controls.addWidget(QLabel("中心块颜色"))
        capture_controls.addWidget(self.center_combo)
        capture_controls.addWidget(self.confirm_button)
        capture_controls.addWidget(self.remove_button)
        capture_layout.addLayout(capture_controls)
        self.orientation_hint = QLabel()
        self.orientation_hint.setStyleSheet("padding:5px; background:#263238; color:white;")
        capture_layout.addWidget(self.orientation_hint)
        splitter.addWidget(capture_panel)

        state_panel = QWidget()
        state_layout = QVBoxLayout(state_panel)
        faces_group = QGroupBox("已采集色块（点击非中心块可手动修改）")
        faces_layout = QGridLayout(faces_group)
        self.face_widgets: dict[Face, FaceGridWidget] = {}
        positions = {
            Face.U: (0, 1),
            Face.L: (1, 0),
            Face.F: (1, 1),
            Face.R: (1, 2),
            Face.B: (1, 3),
            Face.D: (2, 1),
        }
        for face in FACE_ORDER:
            widget = FaceGridWidget(face)
            widget.sticker_changed.connect(self._change_sticker)
            self.face_widgets[face] = widget
            faces_layout.addWidget(widget, *positions[face])
        state_layout.addWidget(faces_group)

        validation_row = QHBoxLayout()
        self.validate_button = QPushButton("校验六面")
        self.validate_button.clicked.connect(self._validate)
        self.solve_button = QPushButton("求解")
        self.solve_button.clicked.connect(self._solve)
        self.cancel_button = QPushButton("取消求解")
        self.cancel_button.clicked.connect(self._cancel_solve)
        validation_row.addWidget(self.validate_button)
        validation_row.addWidget(self.solve_button)
        validation_row.addWidget(self.cancel_button)
        state_layout.addLayout(validation_row)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding:6px; border:1px solid #666;")
        state_layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        state_layout.addWidget(self.progress)
        self.steps = QListWidget()
        self.steps.setAlternatingRowColors(True)
        state_layout.addWidget(self.steps, 1)
        splitter.addWidget(state_panel)
        splitter.setSizes([620, 660])
        self._update_orientation_hint()

    def _refresh_devices(self) -> None:
        was_running = self.camera.running
        if was_running:
            self._stop_camera()
        self.device_combo.clear()
        devices = self.camera.list_devices()
        for device in devices:
            self.device_combo.addItem(device.name, device.device_id)
        if not devices:
            self.device_combo.addItem("未发现摄像头", None)
        self._update_state()

    def _toggle_camera(self) -> None:
        if self.camera.running:
            self._stop_camera()
            return
        device_id = self.device_combo.currentData()
        if device_id is None:
            QMessageBox.warning(self, "摄像头", "未发现可用摄像头")
            return
        try:
            self.camera.start(device_id)
            self.timer.start(33)
            self.camera_button.setText("停止摄像头")
        except CameraError as exc:
            QMessageBox.critical(self, "摄像头错误", str(exc))
        self._update_state()

    def _stop_camera(self) -> None:
        self.timer.stop()
        self.camera.stop()
        self.last_frame = None
        self.preview.setText("摄像头已停止")
        self.camera_button.setText("启动摄像头")
        self._update_state()

    def _read_camera(self) -> None:
        try:
            frame = self.camera.read_frame()
        except CameraError as exc:
            self._stop_camera()
            QMessageBox.critical(self, "摄像头错误", str(exc))
            return
        self.last_frame = frame
        self.guide_rect = self.sampler.guide_rect(frame)
        shown = self.sampler.draw_guide(frame, self.guide_rect)
        height, width, channels = shown.shape
        image = QImage(
            shown.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_BGR888,
        ).copy()
        self.preview.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _capture_face(self) -> None:
        if self.last_frame is None or self.guide_rect is None:
            QMessageBox.warning(self, "采集", "请先启动摄像头")
            return
        color = self._selected_color()
        try:
            patches = self.sampler.extract_patches(self.last_frame, self.guide_rect)
            result = self.session.capture(color, patches)
        except (ValueError, cv2.error) as exc:
            QMessageBox.critical(self, "识别错误", str(exc))
            return
        self._refresh_face_widgets()
        message = f"{color.value.title()} 面采集完成，最低置信度 {result.minimum_confidence:.0%}"
        if result.low_confidence_indices:
            message += "；红框色块建议手动检查"
        self.status_label.setText(message)
        self._update_state()

    def _remove_current_face(self) -> None:
        color = self._selected_color()
        self.session.remove(COLOR_FACES[color])
        self._refresh_face_widgets()
        self._update_state()

    def _change_sticker(self, face: Face, index: int, color: CubeColor) -> None:
        try:
            self.session.set_sticker(face, index, color)
        except ValueError as exc:
            QMessageBox.warning(self, "修改色块", str(exc))
        self._refresh_face_widgets()
        self._update_state()

    def _refresh_face_widgets(self) -> None:
        faces = self.session.faces
        for face, widget in self.face_widgets.items():
            widget.set_grid(faces.get(face))

    def _validate(self) -> bool:
        result = self.session.validate()
        if result.valid:
            self.status_label.setText("魔方状态合法，可以求解。")
        else:
            self.status_label.setText("校验失败：\n" + "\n".join(result.errors))
        self.solve_button.setEnabled(result.valid and self.worker is None)
        return result.valid

    def _solve(self) -> None:
        if not self._validate():
            return
        cube = self.session.to_facelet_cube().to_cubie_cube()
        self.steps.clear()
        self.worker = SolveWorker(cube, self.database_manager)
        self.worker.progress.connect(self._show_progress)
        self.worker.completed.connect(self._solve_completed)
        self.worker.failed.connect(self._solve_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()
        self.status_label.setText("正在准备模式数据库并求解，请稍候……")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._update_state()

    def _cancel_solve(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.status_label.setText("正在取消求解……")

    def _show_progress(self, label: str, current: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.progress.setFormat(f"{label}: %p%")

    def _solve_completed(self, result: SolveResult) -> None:
        self.status_label.setText(
            f"求解完成：{len(result.moves)} 步，第一阶段 {result.phase1_length} 步，"
            f"搜索 {result.nodes_searched:,} 个节点，用时 {result.elapsed_seconds:.2f} 秒。"
        )
        for index, move in enumerate(result.moves, start=1):
            phase = "阶段一" if index <= result.phase1_length else "阶段二"
            self.steps.addItem(f"{index:02d}. {move.notation:<2}  {phase}  （黄底绿朝前）")

    def _solve_failed(self, message: str) -> None:
        self.status_label.setText(f"求解失败：{message}")

    def _worker_finished(self) -> None:
        self.worker = None
        self.progress.setVisible(False)
        self._update_state()

    def _update_orientation_hint(self) -> None:
        color = self._selected_color()
        top = TOP_REFERENCE[color]
        self.orientation_hint.setText(
            f"当前采集 {color.value.title()} 面：请让 {top.value.title()} 色面位于画面顶部。"
        )

    def _selected_color(self) -> CubeColor:
        return CubeColor(self.center_combo.currentData())

    def _update_state(self) -> None:
        solving = self.worker is not None
        self.confirm_button.setEnabled(self.camera.running and not solving)
        self.remove_button.setEnabled(not solving)
        self.validate_button.setEnabled(not solving)
        self.cancel_button.setEnabled(solving)
        self.solve_button.setEnabled(
            not solving and self.session.complete and self.session.validate().valid
        )

    def closeEvent(self, event: object) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(2000)
        self.camera.stop()
        event.accept()


def run_app() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
