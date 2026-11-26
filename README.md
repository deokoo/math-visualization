# Math Visualization

一个用 Python 生成数学动画的代码集。目前收录十七个案例：

1. **高斯积分** `∫e^(−x²)dx = √π` 极坐标证明的完整复刻
2. **复指数映射** `w = eᶻ` 的共形映射可视化
3. **傅里叶级数** 旋转圆链合成方波与吉布斯现象
4. **泰勒展开** `sin x` 的多项式逼近与余项收敛
5. **最速降线** 变分法求最快下滑曲线：摆线与四轨竞速
6. **元胞自动机** 局部规则如何涌现复杂世界：混沌、分形、图灵完备与滑翔机枪
7. **Dijkstra 最短路** 逐节点松弛，一步步锁定最短距离
8. **微分方程相图** 迹与行列式如何决定平衡点的类型：鞍点、结点、焦点、中心
9. **二分查找** 五亿条有序数据 29 步锁定目标：每步砍一半与 log₂n 收缩曲线
10. **格林公式** 沿闭合边界绕一圈的环流 = 区域内部旋度之和：小环流抵消、旋度热图与双数值收敛
11. **梯度与方向导数** 沿等高线走变化为零、沿梯度走变化最快：旋转方向向量与 D_u f = |∇f| cosθ 实时联动
12. **曲面积分** 切向量叉出法向、通量密度 F·n 逐块累加：曲面上的积分化为参数域二重积分
13. **斯托克斯公式** 闭合边界环流 = 任意张成曲面的旋度通量：换曲面答案不变
14. **曲率与曲率圆** `κ=|dα/ds|`、`R=1/κ`：半径连出圆心，曲率圆与曲线二阶接触
15. **拉普拉斯变换** `f(t)e^(-st)` 的面积累积成 `F(s)`：极点与收敛域的可视化
16. **高斯散度定理** 六个面的 `F·n` 通量之和 = 内部 `∇·F` 的体积分：移动源点总和不变
17. **雅可比行列式与换元** 矩形网格弯成极坐标网格：偏导边向量张成的平行四边形给出局部面积放大率 `J=r`，于是 `dx dy = r dr dθ`

后续会持续加入更多数学可视化主题（线性代数、微分方程、离散数学、组合数学等），每日一篇。

> A collection of Python-generated math animations: Gaussian integral proof, conformal mapping `w = e^z`, Fourier series square-wave synthesis, Taylor expansion of sin x, the brachistochrone problem, cellular automata (emergence from local rules), Dijkstra's shortest-path algorithm, ODE phase portraits (how trace and determinant classify equilibria), binary search (locating a target among 500 million sorted values in 29 halving steps), Green's theorem (circulation along a closed boundary equals the sum of curl over its enclosed region), gradient & directional derivative (zero change along a contour, steepest ascent along ∇f, with D_u f = |∇f| cos θ), surface integrals (flux through a parametrized surface, reduced to a double integral over its parameter domain), Stokes' theorem (circulation along a closed boundary equals curl flux through any spanning surface), curvature & the osculating circle (`κ=|dα/ds|`, `R=1/κ` and second-order contact), the Laplace transform (accumulated area of `f(t)e^(-st)` becomes `F(s)`, with pole and ROC), and the divergence theorem (sum of `F·n` flux over six faces equals the volume integral of `∇·F`, invariant under moving the source), and the Jacobian determinant & change of variables (a rectangular grid bends into polar coordinates; the parallelogram spanned by the two derivative vectors gives the local area scale `J=r`, hence `dx dy = r dr dθ`). One new visualization every day.

## 技术栈 / Toolchain

| 工具 | 用途 |
|------|------|
| **Manim Community 0.22** | 主力渲染引擎：3D 线框曲面、公式对象、逐帧时间轴动画 |
| **matplotlib mathtext** | 在**无 LaTeX** 环境下把公式渲染成 PNG（本机离线关键的避坑方案） |
| **Pillow (PIL)** | 逐帧图像绘制、公式贴图、纯内存动态数字渲染 |
| **numpy** | 数值计算、曲面采样、BGM 波形合成 |
| **ffmpeg**（imageio-ffmpeg 二进制） | H.264 视频编码、AAC 音轨合成 |

全程**无需安装 LaTeX**，适合离线 / 受限环境。

## 已收录作品：高斯积分可视化

基于对原视频号动画（作者：架构师研究会）的逐帧分析，用代码重建每一处数学元素与动画逻辑。

### 数学主线

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

### 脚本说明

| 文件 | 说明 |
|------|------|
| `gaussian_scene_clean.py` | **最终成片场景（推荐）**。静态公式 + 动态数值，3D 线框曲面、极坐标微元、雅可比数值、`I²` 进度条逐帧驱动。 |
| `gaussian_scene.py` | 公共工具库：无 LaTeX 公式缓存、纯内存动态数字 `PixelNumber`、折线工具。 |
| `gaussian_scene_exact.py` | 逐帧参数对齐的过渡版本。 |
| `make_video.py` | 早期纯 Python / Pillow 逐帧绘制版本，无 Manim 依赖，作为对照保留。 |
| `add_bgm.py` | 合成背景音乐并混入成片。 |
| `analyze_original.py` | 逐帧总览图与运动镜头分析。 |
| `extract_params.py` | 从原片反解 `I²(t)`、极坐标半径与角度，输出 `params.csv`。 |
| `params.csv` | 643 帧逐帧反解数据（时间、`I²`、半径、角度），供场景插值使用。 |

## 已收录作品：复指数映射 w = eᶻ

第二个案例，沿用同一套视觉规范：**公式全程固定，只动参数、网格与动画**。画面包含 3D 对数螺线面、z 平面直角网格向 w 平面极坐标网格的连续变形、`det J = e^(2x)` 数值联动，以及末行 `f'(z) = eᶻ ≠ 0 ⇒ conformal` 的定格高亮。

详见 [`examples/complex-exponential/README.md`](examples/complex-exponential/README.md)。

## 已收录作品：傅里叶级数合成方波

第三个案例：旋转圆链的第 1、3、5… 次谐波逐次加入，部分和波形从正弦逼近方波，跳变附近始终保留约 9% 超调（吉布斯现象）。数值行实时显示谐波数 `N`、相位 `φ` 与超调量。

详见 [`examples/fourier-series/README.md`](examples/fourier-series/README.md)。

## 已收录作品：泰勒展开逼近 sin x

第四个案例：3D 瀑布曲面展示整族泰勒多项式随阶数 n（0→5）逐渐收敛到 sin x；2D 收敛图的红色 P_n 曲线逐帧贴近青色目标，橙色误差竖线与青色 `|R_n| ≤ 0.12` 区间条同步变化，直观看到"逼近"与"收敛区间扩大"。数值行实时显示阶数、扫描点、多项式值与余项。

详见 [`examples/taylor-series/README.md`](examples/taylor-series/README.md)。

## 已收录作品：最速降线（摆线竞速）

第五个案例：把时间写成泛函 `T = ∫ds/v`，用 Beltrami 恒等式推出 `y(1+y′²) = 2a`，其解为摆线。画面上半是 3D 摆线族曲面（调参数 a 使曲线穿过终点 B），下半是直线 / 折线 / 抛物线 / 摆线四球竞速——摆线 1.39 s 最先到达，比直线 1.63 s 快 14.8%。

详见 [`examples/brachistochrone/README.md`](examples/brachistochrone/README.md)。

## 已收录作品：元胞自动机（涌现）

第六个案例：顶部是一维初等元胞自动机时空图，从单个细胞出发，规则 30 / 90 / 110 / 184 依次逐行生长——分别呈现混沌三角、谢尔宾斯基分形、可图灵完备的复杂结构与车流模型；中部是康威生命游戏的高斯帕滑翔机枪，按 `B3/S23` 持续发射滑翔机。数值行实时显示世代 `t`、规则号、当前行活细胞数 `N_live` 与生命游戏代数 `gen`。整条动画体现"简单局部规则 → 复杂全局行为"的涌现思想。

详见 [`examples/cellular-automata/README.md`](examples/cellular-automata/README.md)。

## 已收录作品：Dijkstra 最短路

第七个案例：8 节点加权图从源点出发，逐节点执行"选当前距离最小的点 → 松弛邻边"。节点颜色演化（灰=未访问、橙=探索、红=当前、青=已锁定），青色最短路树逐步生长，橙色高亮正在松弛的边；每个节点下方 `d(v)` 从 `∞` 实时降到最终值，终点 `d(H)=13`。数值行联动 `|S|` / `relax` / `d(H)` / `updates`。是离散数学图论方向的开篇。

详见 [`examples/dijkstra/README.md`](examples/dijkstra/README.md)。

## 已收录作品：微分方程相图

第八个案例：二维线性系统 `x' = Ax` 的相平面。灰青箭头是向量场，青色是解轨迹，红色是关键轨迹；参数矩阵 A(t) 从鞍点连续演化到稳定结点、稳定焦点、中心，右侧类型标签、左侧 a/b/c/d 数值以及 `τ=tr A`、`Δ=det A`、`τ²−4Δ`、`λ₁,₂` 全部实时联动。判别式为负时特征值自动切换为共轭复根显示。是微分方程 / 动力系统方向的开篇。

详见 [`examples/phase-portrait/README.md`](examples/phase-portrait/README.md)。

## 已收录作品：二分查找（五亿数据 29 步）

第九个案例：五亿条有序数据的抽象可视化。顶部 GLOBAL 全局数据带与中部 ZOOM 放大带用青色标出当前候选区间、红竖线标中点 mid、橙竖线标目标 T（314159265，π 前 9 位）；每步比较 A[mid] 与 T 后砍掉一半，决策文字实时切换，候选从五亿收敛到 1；左下对数轴收缩曲线画出剩余量指数下降，数值行联动 step / left / mid / T。是算法 / 离散数学方向的第二篇。

详见 [`examples/binary-search/README.md`](examples/binary-search/README.md)。

## 已收录作品：格林公式（边界环流 = 区域旋度和）

第十个案例：多元微积分「边界 ↔ 区域」的经典桥梁。中央青色闭合曲线 C 围住区域 D，红点沿边界逆时针走一圈，左侧环流从 0 累加到 10.401；区域切成小格后公共边上的小环流两两抵消（黄色 × 标出抵消点），红色旋度热图按极角铺满 D，右侧区域旋度和也累加到 10.401，等号亮起。底部 5 行固定公式链从 `∮_C Pdx+Qdy` 走到 `∬_D(∂Q/∂x−∂P/∂y)dA`。热图用区域蒙版严格裁到曲线内，避免方格缺口；是本系列第一个多元积分学主题。

详见 [`examples/green-theorem/README.md`](examples/green-theorem/README.md)。

## 已收录作品：梯度与方向导数

第十一个案例：多元微积分「方向 → 变化率」的核心关系。中央是三个高斯钟形叠加成的等高线地形，青色梯度箭头 ∇f 永远垂直于等高线、指向上升最快处，橙色直角标记把这个几何事实钉住；红色单位向量 u 绕定点 P 旋转，橙色角弧标出与梯度的夹角 θ，右下 cos 条随符号在青/红间切换。左侧梯度模长恒为 1.237，右侧方向导数随 cosθ 实时跳动：沿等高线归零、逆梯度跌到 −1.237、沿梯度升到 +1.237。底部 5 行固定公式链走完 `∇f → D_u f = ∇f·u → |∇f|cosθ → max/0/min`。是梯度 / 场论方向的开篇。

详见 [`examples/gradient/README.md`](examples/gradient/README.md)。

## 已收录作品：曲面积分（通量）

第十二个案例：把曲面上的积分化为参数域上的二重积分。中央是参数曲面 `Σ: r(u,v)=(u,v,f(u,v))` 的 3D 线框，先在面元上画出 r_u、r_v 两条切向量并叉出法向 r_u×r_v=(−f_u,−f_v,1)；青色场箭头浮上曲面，红色箭头标出穿面通量；随后面元沿 u+v 对角前沿逐块点亮，右侧 Φ 从 0 累加到 11.278。底部固定公式链从参数化走到 `∬_Σ F·n dS = ∬_D F(r)·(r_u×r_v) du dv`。是场论 / 曲面积分方向的开篇。

详见 [`examples/surface-integral/README.md`](examples/surface-integral/README.md)。

## 已收录作品：斯托克斯公式（环流 = 旋度通量）

第十三个案例：场论里最漂亮的"降维"结论。橙色闭合边界 C 上红色切向箭头绕行，左侧环流从 0 累加到 10.619；青色旋度场 ∇×F=(x,−y,2) 浮上抛物面帽，196 个曲面片按极角逐扇区点亮，右侧旋度通量追上同一个数。随后曲面连续压平为共享同一条边界的平圆盘（原帽留淡轮廓），两个数值纹丝不动——积分只认边界。底部固定公式链从边界参数化走到 `∮_∂Σ F·dr = ∬_Σ (∇×F)·n dS`。

详见 [`examples/stokes-theorem/README.md`](examples/stokes-theorem/README.md)。

## 已收录作品：曲率与曲率圆（Curvature / Osculating Circle）

第十四个案例：单变量微积分里"弯得多快"的几何答案。顶部双语标题下是 `f(x)=1.2 sin x` 的图像：两点切线夹角 Δα 除以收缩的弧长 Δs 给出曲率定义；红点沿曲线滑行时曲线按 κ 大小由灰转青，κ 与 R=1/κ 实时联动。橙色半径连出圆心 C(a,b)，青色曲率圆在峰顶收紧、在两翼放松；最后同屏对比 R=0.83 的小圆与 R=2.30 的大圆——共享切线却二阶接触才是曲率圆。底部固定公式链从 `κ=|dα/ds|` 走到 `(x−a)²+(y−b)²=R²`。

详见 [`examples/curvature-circle/README.md`](examples/curvature-circle/README.md)。

## 已收录作品：拉普拉斯变换（Laplace Transform / Time to Frequency）

第十五个案例：从时间域走向复频域的经典桥梁。左侧红色 `f(t)=e^(-at)` 与青色衰减核 `e^(-st)` 同屏，橙色的乘积面积随积分上限 T 从 0 扫到 6 而累积，`F(T)` 逼近 `1/(s+a)=0.500`；右侧 s 平面用红色 × 标出极点 `s=-a`，青色阴影标出收敛域 `Re(s)>-a`。当 s 穿过极点时读数切换为 ∞ 并出现 DIVERGES 警告，直观解释"为什么收敛域不能丢"。底部固定公式链从定义走到 ROC。**本片起标题不再显示打字机竖线，历史场景也同步修正。**

详见 [`examples/laplace-transform/README.md`](examples/laplace-transform/README.md)。

## 已收录作品：高斯散度定理（Divergence Theorem / Flux = Source）

第十六个案例：场论「边界 ↔ 内部」的第二座桥。斜投影立方体 `Ω=[−a,a]³` 的可见三面按通量密度 `F·n` 着色（青=流出、红=流入）并长出穿出箭头，右侧六条通量条按 `±x/±y/±z` 逐个填充；下方「边界通量总和」从 0 累加到 27.783，「内部体积分」随后追上同一个数。最后把源点 p 移出立方体，各面通量重新分配、总和纹丝不动——散度恒定、体积不变，答案就不变。底部固定公式链从 `∮_∂Ω F·n dS = ∭_Ω ∇·F dV` 走到 `Σ(±a−pᵢ)(2a)² = 24a³`。

详见 [`examples/divergence-theorem/README.md`](examples/divergence-theorem/README.md)。

## 配乐差异化约定

系列铁律：**每部动画的配乐必须彼此不同**——调性 / 音色 / 节奏结构至少换两样，仅换随机种子不算。新片配乐与库中任意一片的波形余弦相似度须 < 0.1。已用配乐签名与下一部选曲建议见 [`BGM_REGISTRY.md`](BGM_REGISTRY.md)。

## 安装

```bash
pip install -r requirements.txt
```

另需系统安装 `ffmpeg` 并加入 PATH（本仓库的渲染流程用 imageio-ffmpeg 自带的二进制即可）。

## 运行

```bash
# 1) 渲染 Manim 场景（1080x1350, 30fps, ~21.3s）
manim -pql gaussian_scene_clean.py GaussianClean

# 2) 合成背景音乐
python add_bgm.py <渲染出的视频路径> output.mp4
```

`formulas/` 目录是公式 PNG 缓存，**不需要手动准备**：`gaussian_scene.py` 的 `formula()` 在图片不存在时会用 mathtext 自动生成。脚本已全部使用相对路径，克隆到任意目录即可运行。

`extract_params.py` / `analyze_original.py` 需要原片 mp4 及逐帧图片（因版权与体积不随仓库分发）；直接用现成的 `params.csv` 即可渲染成片。

## 说明

成片是**基于原片逐帧分析的代码重建**，数值、时序、元素位置均已对齐原片，但字体与细微透视并非像素级一致。要像素级还原只能回放原帧或取得原作者工程。
