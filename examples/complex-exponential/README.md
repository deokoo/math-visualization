# 复指数映射 w = e^z（共形映射）

`math-visualization` 的第二个动画案例。与「高斯积分」成片采用同一套视觉规范：**公式全程固定，只动参数、网格与动画**。

## 数学主线

```text
z = x + iy
w = e^z = e^x (cos y + i sin y)
u = e^x cos y ,  v = e^x sin y

J = det[ ∂u/∂x   ∂u/∂y ] = det[ e^x cos y   -e^x sin y ] = e^(2x)
        [ ∂v/∂x   ∂v/∂y ]       [ e^x sin y    e^x cos y ]

f'(z) = e^z ≠ 0   ⇒   conformal（保角）
```

## 画面元素与数学含义

- 顶部 3D 线框**对数螺线面** = `w = e^z` 的像；青/红边缘是 `ρ=e^(-1)`、`ρ=e¹` 两条边界螺旋线
- 左侧 **z 平面直角网格**：青色竖线是 `x = const`，红色横线是 `y = const`
- 右侧 **w 平面极坐标网格**：青色圆是 `ρ = e^x`，红色射线是 `φ = y`
- 中段数值：`x`、`ρ=e^x`、`y=φ`、`det J = e^(2x)` 逐帧联动
- 约 12.4s 后直角网格连续**弯曲成极坐标网格**；16.15s 循环重置，重新扫描
- 底部固定公式链，末行 `f'(z)=e^z≠0 ⇒ conformal` 在高亮下划线中定格

## 技术栈

| 工具 | 用途 |
|------|------|
| **Manim Community 0.22** | 场景渲染、逐帧时间轴动画 |
| **matplotlib mathtext** | 无 LaTeX 环境下把公式渲染成透明 PNG |
| **Pillow** | 公式贴图、纯内存动态数字渲染 |
| **numpy** | 曲面采样、正交投影、BGM 波形合成 |
| **ffmpeg** | H.264 编码、AAC 音轨合成 |

全程无需安装 LaTeX。

## 文件

| 文件 | 说明 |
|------|------|
| `complex_exp_scene.py` | 最终场景（推荐）：3D 螺旋面 + 双平面 + 网格变形 + 固定公式链 |
| `add_bgm.py` | 合成明亮科技脉冲 BGM（Dm–Bb–F–C，低音心跳 + 铃音琶音）并混入成片 |

`formulas/` 是公式 PNG 缓存，**无需手动准备**：脚本在图片不存在时会用 mathtext 自动生成。

## 运行

```bash
pip install -r ../../requirements.txt

# 1) 渲染（1080x1350, 30fps, ~22.3s）
manim -pqh complex_exp_scene.py ComplexExponential

# 2) 合成背景音乐（系统需有 ffmpeg，或把 imageio-ffmpeg 的二进制放在 PATH）
python add_bgm.py media/videos/complex_exp_scene/1350p30/ComplexExponential.mp4 output.mp4
```
