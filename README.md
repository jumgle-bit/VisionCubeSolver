# VisionCubeSolver

VisionCubeSolver 是一个离线桌面应用，通过摄像头采集标准配色 3x3 魔方，
校验六面状态，并使用两阶段 IDA* 与模式数据库生成求解步骤。

固定方向为：

- 黄面朝下
- 绿面朝向用户
- `U=White, R=Red, F=Green, D=Yellow, L=Orange, B=Blue`

## 安装与启动

需要 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m vision_cube_solver
```

第一次求解前会生成模式数据库并缓存在 `data/pdb/`。生成过程可能需要几分钟，
后续启动会直接加载缓存。

## 使用步骤

1. 选择并启动摄像头。
2. 选择当前面的中心颜色，按照顶部颜色提示摆放魔方。
3. 点击“确认当前面”，检查并按需修改识别结果。
4. 完成六面后点击“校验”。
5. 校验通过后点击“求解”，按列表中的步骤旋转魔方。

## 开发检查

```powershell
pytest
ruff check .
```

## 构建 Windows EXE

```powershell
.\build_exe.ps1
```

构建完成后双击 `dist\VisionCubeSolver.exe`。应用会自动启动电脑内置摄像头
`Camera 0`，也可以点击“刷新摄像头”查找其他设备。

详细设计见 [docs/PLAN.md](docs/PLAN.md)。
