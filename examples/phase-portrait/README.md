# 微分方程相图（Phase Portrait）

二维线性微分方程组的相平面可视化：一个矩阵如何决定所有解轨迹的命运。

## 数学主线

```
ẋ = a x + b y ,   ẏ = c x + d y          (二维线性系统 x' = A x)

τ = tr A = a + d ,   Δ = det A = a d − b c

det(A − λI) = λ² − τ λ + Δ = 0          (特征方程)

λ₁,₂ = ( τ ± √(τ² − 4Δ) ) / 2

Δ < 0                 → 鞍点 saddle
Δ > 0 , τ² > 4Δ       → 结点 node（τ<0 稳定）
Δ > 0 , τ² < 4Δ       → 焦点 focus（共轭复根，轨迹旋转）
τ = 0 , Δ > 0         → 中心 center（纯虚根，轨迹闭合）
```

## 画面结构

- **双语标题**（顶部）：中文「微分方程相图」+ 英文数学标题「PHASE PORTRAIT / LINEAR SYSTEMS」，右下角 `AIMaker-Lab` 小水印；中文由 `cn_text()` 用微软雅黑预渲染。
- **中央**：相平面。灰青小箭头 = 向量场；青色 = 解轨迹；红色 = 关键轨迹；中心红点 = 平衡点。矩阵 A(t) 在四种平衡点类型间连续演化：鞍点 → 稳定结点 → 稳定焦点 → 中心，随后快速重演。
- **左侧**：参数矩阵 A 的 a、b、c、d 实时数值。
- **右侧**：当前平衡点类型与稳定性标签。
- **数值行**：τ、Δ、τ²−4Δ、λ₁,₂（判别式为负时显示共轭复根）。
- **底部**：5 行固定公式链。

## 技术要点

- 无 LaTeX：公式由 matplotlib mathtext 预渲染成透明 PNG（`formulas/` 目录按需自动生成）。
- 相平面用 `PhaseField(ImageMobject)`：numpy 向量化 RK4 积分生成解轨迹 + PIL 整块绘制箭头场，每帧原位更新 `pixel_array`，避免上千个 Manim 图元。
- 动态数字用纯内存 `PixelNumber`，`set_value` 原地重写像素，零文件 IO。
- 所有动态层必须放进 Timeline 持有的 Group，否则会被 Manim 烘焙成静态层不再更新。
- 配乐签名 **PHASOR / F#-Dorian**（seed 201120）：F# 多利亚调式 + 减法滤波锯齿扫频 pad + 钢片琴非谐波泛音 + 低通噪声打击 + 5/4 拍黄金比间隔音序，全程不循环。

## 运行

```bash
manim -pql phase_portrait_scene.py PhasePortrait   # 预览
manim -pqh phase_portrait_scene.py PhasePortrait   # 1080x1350 成片
python add_bgm.py                                   # 合成并混入 BGM
```
