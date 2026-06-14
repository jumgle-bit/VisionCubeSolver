"""Application entry point."""

from __future__ import annotations


def main() -> int:
    try:
        from vision_cube_solver.app.main_window import run_app
    except ImportError as exc:
        if exc.name in {"PySide6", "cv2", "numpy"}:
            raise SystemExit(
                "Missing desktop dependencies. Install them with: "
                'python -m pip install -e ".[dev]"'
            ) from exc
        raise
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())

