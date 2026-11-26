# 雅可比行列式与换元（Jacobian Determinant / Change of Variables）

换元积分里那个 `du dv` 为什么会变成 `r dr dθ`？答案藏在“网格被弯曲时，一小块面积放大了多少倍”里。

> 标题双语，整体淡入后常亮，无打字机竖线 `|`；品牌水印统一 `AIMaker-Lab`。

## 固定公式链（画面底部五行，全程位置不变）

```
∫∫_D f dx dy = ∫∫_{D'} f |J| du dv                 ← 换元公式
x = u cos v,  y = u sin v   (u=r, v=θ)              ← 极坐标换元
J = ∂(x,y)/∂(u,v) = ∂x/∂u·∂y/∂v − ∂x/∂v·∂y/∂u      ← 行列式定义
J = cos v·u cos v − (−u sin v)·sin v = u            ← 代入化简
ΔA_image = u0·Δu·Δv  ⇒  dx dy = r dr dθ             ← 局部面积放大率
```

本片具体取：

- 形变 `T_s(u,v) = ((1−s)u + s·u cos v, s·u sin v)`，`s: 0→1`，矩形网格连续弯成极坐标网格；
- 参数微元初值 `u0=0.90, v0=0.82, Δu=0.44, Δv=0.62`；
- 行列式阶段 `u0: 0.72→1.72`，读数同步跟上；
- 收缩阶段 `Δu: 0.44→0.12, Δv: 0.62→0.18`，`v0: 0.82→1.92`，放大率始终等于 `u0=r`。

## 动画分镜（22.33 秒）

| 时间（成片） | 阶段 | 画面 | 中文讲述行 |
|------|------|------|-----------|
| 0–2.2s | intro | 双语标题淡入；参数平面矩形网格浮现 | 换元时，一小块面积会被放大多少倍？ |
| 2.2–5.9s | morph | 右侧网格随 s 从恒等形变弯成同心圆 + 放射线 | 令 (u,v) 变成 (u cos v, u sin v)：矩形网格弯成极坐标网格 |
| 5.9–10.7s | patch | 左侧小矩形点亮；右侧映成青色曲边四边形，长出青/红偏导边向量与橙色线性近似平行四边形 | 盯住参数平面里的小矩形：它的两条边分别映成青、红两个向量 |
| 10.7–15.7s | jac | u0 从 0.72 增到 1.72；像面积/原面积与 |J| 两个读数同步上升 | 两个向量张成的平行四边形面积，就是局部面积放大率 |
| 15.7–19.5s | shrink | 微元缩小、v0 平移；读数纹丝不动 | 缩小并重放微元：放大率只取决于 r，与 v 和微元大小无关 |
| 19.5–20.8s | final | 换元公式点亮 | 这个放大率就是雅可比行列式：dx dy = |J| du dv = r dr dθ |
| 20.8–22.3s | replay | 定格 | 网格怎么弯不重要，重要的是每一点的局部面积怎么变 |

## 技术要点

- `JacobianPlane(ImageMobject)`：静态底图（分栏 + 参数网格 + 标签）预渲染一次，每帧原位更新 `pixel_array`；右侧形变网格、微元、偏导向量、平行四边形均在同一 numpy 像素缓冲上绘制。
- 预检同时验证：解析面积 `((u0+Δu/2)²−(u0−Δu/2)²)·Δv/2` 除以 `ΔuΔv` 精确等于 `u0`；四偏导数的中心差分与 `det J=u` 吻合；最终网格与微元全部落在画布内。
- 公式 PNG 由 matplotlib mathtext 预渲染（无 LaTeX）；`\int\!\!\int` 代替 `\iint`，避免部分版本的字形缺失。
- 中文标题/讲述行用微软雅黑预渲染，数学符号全部走 mathtext，避免豆腐块。
- 配乐签名 **MELLOW / Eb-Mixolydian-Rhodes**（seed 2611262，用户反馈重做）：Eb 混合利底亚 + FM 电钢琴（颤音 + 微合唱）+ 竖琴泛音琶音 + 软刷镲 + 5/7/11/6/9/13/8/10 非对称乐句分组；不循环、无打字机噪声与音乐锯滑音。实测 vs 库内 18 部旧片最大 |cos| = 0.0521，自相关峰 0.355。

## 运行

> 本机 `python` 为裸解释器、不含 manim；请用
> `C:/Users/admin/.workbuddy/binaries/python/envs/default/Scripts/python.exe` 执行 `.py` 脚本。

```bash
python precheck.py                                       # 公式与数值预检
manim -pql preview_scene.py JacobianChange               # 半分辨率预览
python check_frames.py                                   # 抽关键帧拼图检查
manim -pqh jacobian_scene.py JacobianChange              # 1080×1350 成片
python add_bgm.py                                        # 合成并混入 BGM
```
