# 高斯积分可视化（Gaussian Integral / ∫e^(−x²)dx = √π）

本合集的第 1 个案例。基于对原视频号动画（作者：架构师研究会）的逐帧分析，用代码重建每一处数学元素与动画逻辑。脚本位于仓库根目录（见下方文件表），本目录只放说明。

## 数学主线

```
I  = ∫ e^(−x²) dx                                  (一维高斯积分)

I² = ∬ e^(−(x²+y²)) dx dy                          (平方 → 二维)

x = r·cosθ, y = r·sinθ                              (极坐标变换)

J  = det[ ∂x/∂r   ∂x/∂θ ] = det[ cosθ   −r sinθ ] = r
        [ ∂y/∂r   ∂y/∂θ ]       [ sinθ    r cosθ  ]

I² = ∫₀^{2π} ∫₀^∞ e^(−r²)·r dr dθ = ∫₀^{2π} ½ dθ = π

I  = √π
```

## 画面元素与数学含义一一对应

- 顶部 3D 线框钟形曲面 = 二维高斯 `e^(−(x²+y²))`；两侧青/红边缘曲线分别是 `e^(−x²)`、`e^(−y²)`
- 左下极坐标网 + 红色扇形 = 面积元 `r dr dθ`
- 右上雅可比矩阵 = 在当前 `(r, θ)` 处计算 `det J = r`，红色结果恒等于当前半径
- `I² ≈ ...` 进度条 = 角度积分从 `0` 累积到 `2π`，终点为 `π`
- 约 16.15s 处循环重置：标题闪烁、`I²` 归零

## 脚本说明（仓库根目录）

| 文件 | 说明 |
|------|------|
| [`gaussian_scene_clean.py`](../../gaussian_scene_clean.py) | **最终成片场景（推荐）**。静态公式 + 动态数值，3D 线框曲面、极坐标微元、雅可比数值、`I²` 进度条逐帧驱动。 |
| [`gaussian_scene.py`](../../gaussian_scene.py) | 公共工具库：无 LaTeX 公式缓存、纯内存动态数字 `PixelNumber`、折线工具。 |
| [`gaussian_scene_exact.py`](../../gaussian_scene_exact.py) | 逐帧参数对齐的过渡版本。 |
| [`make_video.py`](../../make_video.py) | 早期纯 Python / Pillow 逐帧绘制版本，无 Manim 依赖，作为对照保留。 |
| [`add_bgm.py`](../../add_bgm.py) | 合成背景音乐并混入成片。 |
| [`analyze_original.py`](../../analyze_original.py) | 逐帧总览图与运动镜头分析。 |
| [`extract_params.py`](../../extract_params.py) | 从原片反解 `I²(t)`、极坐标半径与角度，输出 `params.csv`。 |
| [`params.csv`](../../params.csv) | 643 帧逐帧反解数据（时间、`I²`、半径、角度），供场景插值使用。 |

## 运行

```bash
pip install -r requirements.txt
manim -pql gaussian_scene_clean.py GaussianClean
python add_bgm.py <渲染出的视频路径> output.mp4
```

`extract_params.py` / `analyze_original.py` 需要原片 mp4 及逐帧图片（因版权与体积不随仓库分发）；直接用现成的 `params.csv` 即可渲染成片。
