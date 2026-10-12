# 术语表

[← 目录](README.md)

按第一次出现的章节排序。

| 中文 | 英文 | 一句话解释 | 首次出现 |
|---|---|---|---|
| 位姿 | pose | 相机在哪（平移 t）+ 朝哪看（旋转 R） | 第 0 章 |
| 轨迹 | trajectory | 一段视频所有帧位姿连成的线 | 第 0 章 |
| 流形 | manifold | 局部看是平的、整体是弯的空间 | 1.2 |
| 切空间 | tangent space, $T_pM$ | 在流形某点贴的一张"平纸"，可以在上面做线性代数 | 1.3 |
| 黎曼度量 | Riemannian metric | 每个切空间上的内积，规定了向量多长 | 1.4 |
| 测地线 | geodesic | 只能沿流形走时的最短路径（球面上的大圆） | 1.5 |
| 测地距离 | geodesic distance | 测地线的长度；对旋转来说就是相差的角度 | 1.5 |
| 弦长 | chordal distance | 把流形上两点当成外部空间的点直接相减；"穿隧道"的距离 | 1.5 |
| 对数映射 | logarithm map, Log | 流形上两点 → 切空间里的一个向量（方向 + 距离） | 1.6 |
| 指数映射 | exponential map, Exp | Log 的反操作：切空间里的向量 → 流形上的点 | 1.6 |
| 旋转群 | SO(3) | 所有 3D 旋转矩阵组成的 3 维流形 | 2.1 |
| 轴角 | axis-angle, rotation vector, ω | 旋转 = 绕轴 n 转 θ，合成 ω = θn | 2.3 |
| 李群 | Lie group | 既是群（能复合、求逆）又是流形 | 2.4 |
| 李代数 | Lie algebra, so(3), se(3) | 李群在单位元处的切空间 | 2.3 |
| 双不变度量 | bi-invariant metric | 左乘、右乘同一个元素都不改变距离；SO(3) 的标准度量就是 | 2.4 |
| 四元数 | quaternion, q = (x,y,z,w) | 网络实际输出的旋转表示；q 和 −q 是同一个旋转 | 2.5 |
| 双重覆盖 | double cover | 单位四元数 S³ 两个点对应同一个旋转 | 2.5 |
| 刚体变换群 | SE(3) | 旋转 + 平移，4×4 矩阵 | 2.6 |
| 相对位姿 | relative pose, $T_i^{-1}T_{i+1}$ | 从第 i 帧到第 i+1 帧怎么动的 | 2.6 |
| 体坐标系 | body frame | 用相机自己的坐标系描述运动（"往前、往右"） | 2.6 |
| 规范不变性 | gauge invariance | 整条轨迹整体搬家时，量不变 | 2.6 |
| 积度量 | product metric | 旋转和平移各算各的再加起来 | 2.6 |
| 体速度 | body velocity, twist, ξ | $\mathrm{Log}(T_i^{-1}T_{i+1})$，6 维：转了多少 + 走了多远 | 3.2 |
| 平行移动 | parallel transport | 把切向量"不转弯地"搬到另一点的切空间 | 3.3 |
| 协变加速度 | covariant acceleration | 弯曲空间里"真正的"加速度；沿测地线运动时为 0 | 3.3 |
| 左平凡化 | left trivialization | 把李群上每一点的切向量都左乘搬回单位元 | 3.3 |
| 李括号 | Lie bracket, $[\xi,\eta]$, $\mathrm{ad}_\xi\eta$ | 衡量两个"转动"不可交换的程度；$[\xi,\xi]=0$ | 3.3 |
| Euler–Arnold 方程 | Euler–Arnold equation | 李群上协变加速度的左平凡化公式 | 3.3 |
| 割迹 | cut locus | 测地线不再唯一的地方；对旋转是转角 = 180° | 4.4 |
| Sobolev 度量 | Sobolev metric (H¹, H²) | 比较曲线时同时考虑位置、速度（、加速度） | 3.6 |
| 弹性度量 / SRVF | elastic metric, square-root velocity function | 对"曲线走得快慢"不敏感的曲线距离 | 3.6、6.4 |
| ATE | absolute trajectory error | 0 阶轨迹误差 | 6.3 |
| RPE | relative pose error | 1 阶轨迹误差 | 6.3 |
| 测地样条 | geodesic spline | 弯曲空间里的"光滑插值 / 拟合曲线" | 7.3 R2 |
| Gauss–Newton（李群上） | Gauss–Newton on Lie groups | 每步在切空间解线性最小二乘，再用 Exp 更新 | 7.3 R2 |
| 相似变换群 | Sim(3) | 旋转 + 平移 + 均匀缩放 | 7.3 R4 |
| Umeyama 对齐 | Umeyama alignment | 两组对应点之间的最优相似变换，一次 3×3 SVD 求出 | 7.3 R4 |
| Karcher 均值 | Karcher / Fréchet mean | 黎曼流形上的"平均"：到所有点测地距离平方和最小的点 | 7.3 R4 |
| Kendall 形状空间 | Kendall shape space | 点集去掉位置、朝向、尺度之后剩下的"纯形状"组成的空间 | 7.4 R6 |
| 商空间 | quotient space | 把"只差一个变换"的东西视为同一个点得到的空间 | 7.4 R6 |
| Plücker 坐标 | Plücker coordinates | 用 (方向, 矩) 6 个数表示 3D 直线 | 7.5 R9 |

[← 目录](README.md)
