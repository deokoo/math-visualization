# 傅里叶级数合成方波

`math-visualization` 的第三个动画案例。沿用统一视觉规范：**公式全程固定，只动参数、圆链与波形**。

## 数学主线

```text
f(x) = 4/π · Σ_{n=0}^∞  sin((2n+1)x) / (2n+1)

s_N(x) = 4/π · Σ_{n=0}^{N-1}  sin((2n+1)x) / (2n+1)

s_N(x) → 方波            (N → ∞)
max s_N ≈ 1.18          （吉布斯现象，约 9% 超调）
```

第 n 个旋转圆的半径是 `4/(π(2n+1))`，角速度是 `2n+1`；圆链末端的高度正好等于部分和 `s_N(φ)`，所以右侧波形是圆链"画"出来的。

## 画面元素与数学含义

- 左侧**旋转圆链**：青色大圆是基波，灰色小圆依次是 3、5、7… 次谐波
- 右侧**部分和波形**：从正弦逐步逼近方波，跳变点附近有可见的波纹与超调
- 灰色水平参考线 `±1` = 目标方波幅值
- 数值行：`N`（当前谐波数）、`φ`（相位）、`overshoot ≈ 1.18` 逐帧联动
- 约 16.15s 循环重置，重新从基波开始扫描
- 底部固定公式链，末行 `lim s_N(x) = 1` 在高亮下划线中定格

## 技术栈

| 工具 | 用途 |
|------|------|
| **Manim Community 0.22** | 场景渲染、逐帧时间轴动画 |
| **matplotlib mathtext** | 无 LaTeX 环境下把公式渲染成透明 PNG |
| **Pillow** | 公式贴图、纯内存动态数字渲染 |
| **numpy** | 部分和计算、波形采样、BGM 波形合成 |
| **ffmpeg** | H.264 编码、AAC 音轨合成 |

全程无需安装 LaTeX。

## 文件

| 文件 | 说明 |
|------|------|
| `fourier_scene.py` | 最终场景（推荐）：旋转圆链 + 部分和波形 + 吉布斯超调 + 固定公式链 |
| `add_bgm.py` | 合成明亮科技脉冲 BGM 并混入成片 |

`formulas/` 是公式 PNG 缓存，**无需手动准备**：脚本在图片不存在时会用 mathtext 自动生成。

## 运行

```bash
pip install -r ../../requirements.txt

# 1) 渲染（1080x1350, 30fps, ~22.3s）
manim -pqh fourier_scene.py FourierSeries

# 2) 合成背景音乐（系统需有 ffmpeg，或把 imageio-ffmpeg 的二进制放在 PATH）
python add_bgm.py media/videos/fourier_scene/1350p30/FourierSeries.mp4 output.mp4
```
