# Math Visualization

用 Python + Manim 生成的数学动画合集，每日一篇：1080×1350 竖屏、30fps、公式固定只动参数与动画、无 LaTeX、配乐每片不同、全程无需联网。

> Python/Manim math animations: one new visualization every day, rendered offline without LaTeX, each with its own non-looping synthesized score.

## 已收录作品（17）

主表只做索引，**每个案例的详细说明（公式链、分镜、技术要点、运行命令）都在各自的 `examples/<主题>/README.md` 里**，新增案例时只加一行，主 README 不会越写越长。

| # | 主题 | 一句话看点 | 详情 |
|---|------|-----------|------|
| 1 | 高斯积分 | `∫e^(−x²)dx=√π` 的极坐标证明完整复刻 | [README](examples/gaussian-integral/README.md) |
| 2 | 复指数映射 | `w=eᶻ` 的共形映射：直角网格弯成极坐标网格 | [README](examples/complex-exponential/README.md) |
| 3 | 傅里叶级数 | 旋转圆链合成方波与吉布斯现象 | [README](examples/fourier-series/README.md) |
| 4 | 泰勒展开 | `sin x` 的多项式逼近与余项收敛 | [README](examples/taylor-series/README.md) |
| 5 | 最速降线 | 变分法推出摆线：四轨竞速 1.39s 胜出 | [README](examples/brachistochrone/README.md) |
| 6 | 元胞自动机 | 局部规则涌现混沌、分形、图灵完备与滑翔机枪 | [README](examples/cellular-automata/README.md) |
| 7 | Dijkstra 最短路 | 逐节点松弛，五亿数据般的图一步步锁定答案 | [README](examples/dijkstra/README.md) |
| 8 | 微分方程相图 | 迹与行列式决定鞍点、结点、焦点、中心 | [README](examples/phase-portrait/README.md) |
| 9 | 二分查找 | 五亿条有序数据 29 步锁定目标：每步砍一半 | [README](examples/binary-search/README.md) |
| 10 | 格林公式 | 边界环流 = 区域旋度和：小环流两两抵消 | [README](examples/green-theorem/README.md) |
| 11 | 梯度与方向导数 | 沿等高线变化为零、沿梯度上升最快 | [README](examples/gradient/README.md) |
| 12 | 曲面积分 | 切向量叉出法向，通量化为参数域二重积分 | [README](examples/surface-integral/README.md) |
| 13 | 斯托克斯公式 | 闭合边界环流 = 任意张成曲面的旋度通量 | [README](examples/stokes-theorem/README.md) |
| 14 | 曲率与曲率圆 | `κ=|dα/ds|`、`R=1/κ`：二阶接触的贴合圆 | [README](examples/curvature-circle/README.md) |
| 15 | 拉普拉斯变换 | `f(t)e^(−st)` 的面积累积成 `F(s)` 与收敛域 | [README](examples/laplace-transform/README.md) |
| 16 | 高斯散度定理 | 六个面的通量之和 = 内部散度体积分 | [README](examples/divergence-theorem/README.md) |
| 17 | 雅可比行列式与换元 | 网格弯成极坐标，`J=r` 给出局部面积放大率 | [README](examples/jacobian-determinant/README.md) |

后续会持续加入更多数学可视化主题（线性代数、微分方程、离散数学、组合数学等），每日一篇。

### 新增案例时这样做

1. 新建 `examples/<主题英文>/`，放入自包含场景源码、公式 PNG、BGM 脚本、`precheck.py`、分镜 README。
2. **只在上表追加一行**（主题 + 一句话看点 + 详情链接），不要在主 README 里新增「已收录作品：…」长段落。
3. 更新 [`BGM_REGISTRY.md`](BGM_REGISTRY.md) 的配乐签名与相似度实测值。

## 技术栈 / Toolchain

| 工具 | 用途 |
|------|------|
| **Manim Community 0.22** | 主力渲染引擎：3D 线框曲面、公式对象、逐帧时间轴动画 |
| **matplotlib mathtext** | 在**无 LaTeX** 环境下把公式渲染成 PNG（本机离线关键的避坑方案） |
| **Pillow (PIL)** | 逐帧图像绘制、公式贴图、纯内存动态数字渲染 |
| **numpy** | 数值计算、曲面采样、BGM 波形合成 |
| **ffmpeg**（imageio-ffmpeg 二进制） | H.264 视频编码、AAC 音轨合成 |

全程**无需安装 LaTeX**，适合离线 / 受限环境。

## 配乐差异化约定

系列铁律：**每部动画的配乐必须彼此不同**——调性 / 音色 / 节奏结构至少换两样，仅换随机种子不算。新片配乐与库中任意一片的波形余弦相似度须 < 0.1，且自身无循环。已用配乐签名与下一部选曲建议见 [`BGM_REGISTRY.md`](BGM_REGISTRY.md)。

## 安装

```bash
pip install -r requirements.txt
```

另需系统安装 `ffmpeg` 并加入 PATH（本仓库的渲染流程用 imageio-ffmpeg 自带的二进制即可）。

## 运行

通用流程见任一案例的 README；以最近的雅可比行列式为例：

```bash
cd examples/jacobian-determinant
python precheck.py
manim -pql preview_scene.py JacobianChange
python check_frames.py
manim -pqh jacobian_scene.py JacobianChange
python add_bgm.py
```

高斯积分的公式 PNG 缓存目录 `formulas/` **不需要手动准备**；场景脚本在图片不存在时会用 mathtext 自动生成，且全部使用相对路径，克隆到任意目录即可运行。

## 说明

成片是**基于数学推导的可视化重建**，数值与时序与公式严格一致，但字体与细微透视不做像素级还原；早期高斯案例另按原片逐帧分析做了对齐。
