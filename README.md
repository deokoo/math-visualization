# 高斯积分可视化 · Gaussian Integral Replica

用 Python 复刻「高斯积分 ∫e^(−x²)dx = √π」的数学原理动画。原视频来自微信视频号（作者：架构师研究会），本项目基于对原片逐帧分析后，用代码重建每一处数学元素与动画逻辑。

## 数学主线

整个动画讲的是**高斯积分的经典极坐标证明**：

```
I  = ∫ e^(−x²) dx                                  (一维高斯积分)

I² = ∬ e^(−(x²+y²)) dx dy                          (平方 → 二维)

x = r·cosθ, y = r·sinθ                              (极坐标变换)

J  = det[ ∂x/∂r   ∂x/∂θ ] = det[ cosθ   −r sinθ ] = r
        [ ∂y/∂r   ∂y/∂θ ]       [ sinθ    r cosθ  ]

I² = ∫₀^{2π} ∫₀^∞ e^(−r²)·r dr dθ = ∫₀^{2π} ½ dθ = π

I  = √π
```

**画面元素与数学含义一一对应：**

- 顶部 3D 线框钟形曲面 = 二维高斯 `e^(−(x²+y²))`；两侧青/红边缘曲线分别是 `e^(−x²)`、`e^(−y²)`
- 左下极坐标网 + 红色扇形 = 面积元 `r dr dθ`
- 右上雅可比矩阵 = 在当前 `(r, θ)` 处计算 `det J = r`，红色结果恒等于当前半径
- `I² ≈ ...` 进度条 = 角度积分从 `0` 累积到 `2π`，终点为 `π`
- 约 16.15s 处循环重置：标题闪烁、`I²` 归零

## 两个实现版本

| 版本 | 入口脚本 | 说明 |
|------|---------|------|
| **Manim 版（推荐）** | `gaussian_scene_clean.py` | 最终成片版本。静态公式 + 动态数值，3D 线框曲面、极坐标微元、雅可比数值、`I²` 进度条逐帧驱动。 |
| PIL 版 | `make_video.py` | 早期纯 Python/ Pillow 逐帧绘制版本，无 Manim 依赖，作为对照保留。 |

其余脚本：

- `gaussian_scene.py` — Manim 基础工具库（无 LaTeX 的 mathtext 公式、纯内存动态数字 `PixelNumber`、折线工具）
- `gaussian_scene_exact.py` — 逐帧参数对齐的过渡版本
- `add_bgm.py` — 合成背景音乐并混入成片
- `analyze_original.py` — 逐帧总览图与运动镜头分析
- `extract_params.py` — 从原片反解 `I²(t)`、极坐标半径与角度，输出 `params.csv`

## 环境依赖

全程**无需 LaTeX**（本机离线环境）。公式用 matplotlib mathtext 预渲染为 PNG，动态数字用 PIL 内存绘制，零文件 IO。

```bash
pip install -r requirements.txt
```

需要系统已安装 `ffmpeg`（并加入 PATH）。

## 运行

```bash
# 1) 渲染 Manim 场景（1080x1350, 30fps, ~21.3s）
manim -pql gaussian_scene_clean.py GaussianClean

# 2) 合成背景音乐
python add_bgm.py <渲染出的视频路径> output.mp4
```

`formulas/` 是预渲染的公式图片，已随仓库提供。`gaussian_scene_clean.py` 中的 `ROOT` 与 `PARAMS_CSV` 目前是本机绝对路径，克隆到别的机器后需改成你的本地路径；`params.csv` 由 `extract_params.py` 对原片逐帧反解生成（原片 mp4 因版权与体积不随仓库分发）。

## 说明

本项目是**基于原片逐帧分析的代码重建**，数值、时序、元素位置均已对齐原片，但字体与细微透视并非像素级一致。要像素级还原只能回放原帧或取得原作者工程。
