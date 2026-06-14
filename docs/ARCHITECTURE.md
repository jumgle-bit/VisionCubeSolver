# VisionCubeSolver 架构与算法

## 数据流

```text
CameraService
  -> OpenCV BGR frame
  -> GridSampler (9 个 Lab/HSV/BGR 中位数样本)
  -> ColorClassifier (中心块校准 + 最近颜色)
  -> CubeSession (六面采集与手动修改)
  -> FaceletCube.validate()
  -> CubieCube
  -> TwoPhaseIDASolver
  -> Move 列表
```

用户拍摄每一面时必须遵守界面顶部颜色提示。这个约定使九个色块能直接写入
`URFDLB` 标准面块顺序，无需在识别后猜测面的旋转方向。

## 视觉接口

```python
class CameraService:
    def list_devices(self, maximum_devices: int = 8) -> list[CameraDevice]: ...
    def start(self, device_id: int) -> None: ...
    def read_frame(self) -> np.ndarray: ...
    def stop(self) -> None: ...

class GridSampler:
    def guide_rect(self, frame: np.ndarray, size_ratio: float = 0.62) -> Rect: ...
    def extract_patches(self, frame: np.ndarray, guide_rect: Rect) -> tuple[PatchStats, ...]: ...

class ColorClassifier:
    def register_center_sample(self, color: CubeColor, sample: PatchStats) -> None: ...
    def classify(
        self,
        patches: tuple[PatchStats, ...],
        forced_center: CubeColor | None = None,
    ) -> FaceRecognitionResult: ...
```

每个采样块只使用格子中间区域，并对像素求中位数以降低边框、阴影和高光影响。
分类距离主要使用 Lab 色彩空间，降低亮度权重，并对低饱和度白色进行额外约束。

## 状态校验

`FaceletCube` 保存 54 个可见色块，`CubieCube` 保存：

```python
corner_permutation: tuple[int, ...]  # 8 个角块的位置
corner_orientation: tuple[int, ...]  # 8 个角块的方向
edge_permutation: tuple[int, ...]    # 12 个棱块的位置
edge_orientation: tuple[int, ...]    # 12 个棱块的方向
```

转换为 cubie 状态后执行以下校验：

```text
validate(cube):
    确认每种颜色恰好出现九次
    确认六个中心颜色符合固定方向
    确认每个角块和棱块组合存在且仅出现一次
    确认 sum(corner_orientation) mod 3 == 0
    确认 sum(edge_orientation) mod 2 == 0
    确认 corner_parity == edge_parity
```

## 两阶段 IDA*

求解器不依赖第三方魔方求解库。它使用六个基础面动作的 cubie 变换、坐标移动表
和四张组合模式数据库。

第一阶段坐标：

- 角块方向：`3^7 = 2187`
- 棱块方向：`2^11 = 2048`
- 中层棱块组合：`C(12,4) = 495`

第一阶段启发值：

```text
max(
    distance[corner_orientation, slice_combination],
    distance[edge_orientation, slice_combination]
)
```

第二阶段坐标：

- 角块排列：`8! = 40320`
- 上下层棱块排列：`8! = 40320`
- 中层棱块排列：`4! = 24`

第二阶段只使用 `U U2 U' D D2 D' R2 L2 F2 B2`，启发值为：

```text
max(
    distance[corner_permutation, slice_permutation],
    distance[edge_permutation, slice_permutation]
)
```

IDA* 核心流程：

```text
ida_star(start, heuristic, moves, maximum_depth):
    bound = heuristic(start)
    while bound <= maximum_depth:
        if depth_first_search(start, depth=0, bound):
            return current_path
        bound += 1

depth_first_search(state, depth, bound):
    if depth + heuristic(state) > bound:
        return false
    if state == goal:
        return true
    for move in allowed_moves:
        跳过连续同面动作和可交换的重复同轴动作
        if depth_first_search(apply(state, move), depth + 1, bound):
            return true
    return false
```

首次求解会广度优先生成移动表和模式数据库，写入
`data/pdb/two_phase_v1.pkl`。缓存包含版本号并在加载时检查尺寸和已解状态；损坏
或版本不匹配时自动重建。

