import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from vision_cube_solver.app.main_window import MainWindow
from vision_cube_solver.vision.camera import CameraService


def test_main_window_starts_without_camera(monkeypatch) -> None:
    monkeypatch.setattr(CameraService, "list_devices", lambda _self: [])
    app = QApplication.instance() or QApplication([])

    window = MainWindow()

    assert window.windowTitle().startswith("VisionCubeSolver")
    assert window.device_combo.currentText() == "未发现摄像头"
    assert not window.confirm_button.isEnabled()
    window.close()
    app.processEvents()
