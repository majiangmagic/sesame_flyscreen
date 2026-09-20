# flyscreen

**把任意视频播进一只真果蝇的脑子里，然后看它的身体抽搐。**

- **脑** —— [MaleCNS v1.0](https://male-cns.janelia.org/)：成年雄性黑腹果蝇的完整中枢神经系统连接组，
  141,781 个有胞体坐标的神经元、5,536,527 条突触。
- **身体** —— [flybody](https://github.com/TuragaLab/flybody)：Google DeepMind + HHMI Janelia 的
  解剖级 MuJoCo 果蝇，68 刚体 / 103 关节 / 78 执行器。
- **物理** —— 真重力、地面接触、摩擦。果蝇站在影棚地板上被神经驱动抽得满地打滚，
  每一步都是 MuJoCo 真算的。
- **神经动力学** —— Shiu et al. 2024 的全脑 LIF 脉冲网络，活动沿真实连接组传播。

---

## 用法

一个入口，两个常用子命令：

```powershell
python fly.py                   # ① 起 Web 服务（默认命令）
python fly.py render 视频.mp4    # ② 只出视频
```

**① Web 服务** —— 浏览器打开 **http://127.0.0.1:8760/**，把视频文件拖进页面：

- 左边立刻出**实时预览**：脑点云 + 抽搐的果蝇，两块一起
- 右边 20 个参数控件，**改一下立刻生效**，不用重启
- **同时自动出片**：落到 `out/<名字>_flyscreen_<时间>.mp4`，片尾自动收尾 + 混入原音轨

不需要手动点"开始录制"——拖进去就在录。**日志自动写到 `data/logs/webui.log`**
（控制台和文件同时输出），不用管重定向那一套。

**② 只出视频** —— 不开浏览器，直接渲染：

```powershell
python fly.py render D:\clips\bad_apple.mp4               # 输出自动命名到 out/
python fly.py render D:\clips\bad_apple.mp4 out\a.mp4     # 指定输出
python fly.py render D:\clips\bad_apple.mp4 --seconds 10  # 先试 10 秒
```

默认混入源视频音轨（源没音轨就自动跳过），`--no-audio` 关掉。

### 另外两个子命令

```powershell
python fly.py preview 视频.mp4   # pygame 预览窗（ESC 退出，V 换投影轴，G 开关辉光）
python fly.py check              # 自检：依赖装没装、数据齐不齐，缺什么就告诉你跑哪条命令
```

### 面板档位

不用记那串 `--canvas-size` / `--body-size`，三档搞定：

| `--size` | 面板 | 输出 | 点云密度 | 速度 | 用途 |
|---|---|---|---|---|---|
| `small` | 600×490 + 460×490 | 1102×518 | 0.482 点/像素（重叠） | 最快 | 快速试片 |
| `mid` | 800×660 + 580×660 | 1422×688 | 0.268 | 中 | |
| **`big`（默认）** | 1000×820 + 700×820 | **1742×848** | **0.173（点分离）** | 慢 2.8 倍 | 出片 |

```powershell
python fly.py --size small              # 起服务，小画布（预览流畅）
python fly.py render a.mp4 --size big   # 出片，大画布（默认）
```

⚠️ **面板尺寸只在启动时读一次**（`NeuronCanvas` / `PointCloudRenderer` / `FlyBody`
都在构造时取），而且**不在那 20 个参数面板里**——想改必须重启。

### 后台常驻

`python fly.py` 是前台进程，关掉终端就停了。要常驻：

```powershell
$p = Start-Process python -ArgumentList 'fly.py' `
  -WorkingDirectory 'D:\fly-brain\flyscreen' -NoNewWindow -PassThru
Write-Host "PID = $($p.Id)"
```

**不需要重定向、不需要 `-u`** —— `fly.py` 自己把输出同时写控制台和
`data/logs/webui.log`（行缓冲，实时落盘）。

看日志 / 停服务：

```powershell
Get-Content data\logs\webui.log -Wait                 # 跟日志
Stop-Process -Id 61604 -Force                          # 换成上面打出来的 PID
Get-NetTCPConnection -LocalPort 8760 -State Listen |   # 或者按端口找
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### 中文路径的片源

`Start-Process -ArgumentList` 会把中文路径拆坏，所以 `fly.py` 也认环境变量：

```powershell
$env:FLYSCREEN_VIDEO = 'D:\clips\【4K】Bad apple！！！.mp4'
python fly.py
```

直接 `python fly.py render '带中文的名字.mp4'` 没问题——PowerShell 里直接跑
不经过 `Start-Process`，路径不会被拆。

### 要细调参数时

`webui.py` 和 `run.py` 还在，参数更全（`--stream-scale`、`--brain-dt`、`--max-rate`、
`--body-travel`、`--glow` 等几十个）。日常用 `fly.py` 就够，要抠参数再直接调它们：

```powershell
python -u webui.py --port 8760 --stream-scale 0.75 --quality 70
python run.py --video a.mp4 --out b.mp4 --body-travel 0.3 --glow 0
```

（这两个还是老样子：日志要自己重定向，所以得加 `-u`。）

---

## 快速开始

```powershell
cd D:\fly-brain\flyscreen
pip install -r requirements.txt

# 1) 下载连接组（约 1.06 GB，来自 Janelia 的公开云盘）
python -m flyscreen.data.fetch

# 2) 建神经元表缓存（几秒）
python -m flyscreen.data.prepare

# 3) 建连接图缓存（约 4 分钟）
python -m flyscreen.data.prepare_graph

# 4) 下载果蝇身体（约 144 MB，来自 mujoco_menagerie）
python scripts/fetch_flybody.py
python scripts/repair_flybody.py     # 按大小校验并按需补下

# 5) 自检 —— 缺什么它会直接告诉你跑哪条命令
python fly.py check

# 6) 跑
python fly.py                        # 起 Web 服务
python fly.py render 视频.mp4         # 或者只出视频
```

没有素材可以先造一段：

```powershell
python scripts/make_test_video.py out\_test.mp4
python fly.py render out\_test.mp4 --seconds 12
```

---

## 原理：一条管线

```
任意视频
   │  ffmpeg 子进程（任意容器/编码）
   ▼
灰度帧
   │  NeuronCanvas：按胞体坐标投影到 2D + 双线性采样 + 密度均衡
   ▼
drive[141,781]                     每个神经元一个 0~1 的驱动值
   │  LIF：被驱动的神经元按泊松率发放，脉冲沿连接组传播
   ▼
脉冲 + 拖尾                   ┌──────────────┴──────────────┐
   │                          ▼                             ▼
   │                   点云渲染（三层分色）          8 个运动神经元群的活动
   │                          │                             │
   │                          │                    BodyDriver：逐路归一化
   │                          │                             ▼
   │                          │                    MuJoCo 真重力 + 地面接触
   │                          └──────────┬──────────────────┘
   ▼                                     ▼
合成一帧 RGB ──┬──► JPEG → MJPEG 流 → 浏览器 <img>
               └──► ffmpeg 管道 → out/*.mp4
```

**预览和出片是同一份像素**，录出来的一定和你在浏览器里看到的一样。

---

## 数据

只用了 MaleCNS v1.0 里最小的三个表（总共 1.1 GB，而整个数据集是几十 TB 级）：

| 文件 | 大小 | 内容 |
|---|---|---|
| `body-annotations` | 14 MB | 每个神经元一行：类型、左右、神经毡、**胞体 3D 坐标**、出脑神经 |
| `body-neurotransmitters` | 43 MB | 每个神经元一行：**递质预测**（决定它是"油门"还是"刹车"） |
| `connectome-weights` | 1.05 GB | 每条连线一行：`body_pre, body_post, weight`（突触数），共 1.52 亿条 |

加工成两个缓存：

| 缓存 | 大小 | 内容 |
|---|---|---|
| `cache/neurons.npz` | 1.7 MB | 141,781 个神经元的坐标 / 类型 / 兴奋抑制 / **8 个身体部位的归属掩码** |
| `cache/graph.npz` | 44.9 MB | CSR 稀疏图：`indptr(141782) + indices(5536527) + wsyn(5536527)`，权重带符号 |

### 从 1.52 亿条边到 553 万条

`connectome-weights` 用的是**原始分割 ID**，里面混着大量没校对过的小碎片
（`body_post` 有 8,758 万个唯一值，远多于 21 万个神经元）。过滤规则：

```
保留 = 两端都在注释神经元表里  AND  突触数 >= 5
1.52 亿 ────────────────────────────────────► 553 万（丢掉 96%）
```

### 递质怎么用

表 2 的 `consensus_nt` 只做一件事：**给每个神经元的出边定正负号**。

```
乙酰胆碱 / 多巴胺 / 章鱼胺 / 血清素 / unclear  →  兴奋 (+)
GABA / 谷氨酸 / 组胺                          →  抑制 (−)
```

（谷氨酸在成体果蝇身上是抑制性的，和人脑相反——果蝇用的是 GluCl 受体。）

符号挂在**发出方**（`wsyn = weight * sign[body_pre]`，Dale 法则），
所以结果是 90,723 个兴奋 / 51,058 个抑制神经元。

---

## 神经动力学（LIF）

照 Shiu et al. 2024（*Nature* 634:210）的参考实现：

```
tau_m dv/dt = (v_rest - v) + g          tau_m = 20 ms
tau_s dg/dt = -g                        tau_s = 5 ms
脉冲经 1.8 ms 延迟后：g_i += sign_j * N_ji * 0.275 mV * gain
v > -45 mV → 发放；v = -52 mV，g = 0，不应期 2.2 ms
```

积分用精确线性解（每步只算一次标量指数）。**没有自发活动**——静止的脑永远静止，
实测无输入 200 步 = 0 脉冲。

### 视频驱动用泊松，不用注入电流

一开始用注入电流，撞上了阈值悬崖：稳态 `g` 一越过 7 mV，被点亮的神经元就以
133 Hz 连续发放，几千个一起灌，整个网络瞬间饱和，视频图案完全消失。改成
**泊松脉冲发生器**（参考实现就是这么做的）之后平滑可控。

### 两个偏离参考模型的修正

**① E/I 平衡校正。** 原始连接组的总突触权重里，**兴奋比抑制强 1.526 倍**，
网络天生超临界。把抑制侧整体放大到总权重相等后，活动量降到约 1/2.4。

**② 默认增益 0.30，不是参考值 0.65。** 参考实现的 0.65 是在"只驱动少量特定
感觉神经元"的前提下标定的；本项目用视频帧**广域驱动**几千个散布全脑的神经元，
0.65 会让 25,000 个神经元一起烧、与驱动图案的空间相关归零。

实测（30 秒）：

| gain | 脉冲/步 | 参与神经元 | 与驱动图案相关 |
|---|---|---|---|
| 0.65（参考值） | 2,405 | 24,983 (18%) | ≈ 0 |
| **0.30（默认）** | ~600 | ~7,000 (5%) | 保留结构 |

**这两处是偏离，不是"更正确"**，理由和实测数据都写在代码注释里。

---

## 身体与地面

**默认地面模式**：真重力（模型自带 −981，单位是厘米）、地面碰撞、摩擦。
加 `--float` 回到"悬空只有腿翅动"的旧行为。

模型本来就是按"站在地面上"建的：原版 `scene.xml` 的地面在 z = −0.132，
而实测六只跗爪（脚）的最低点在 z = −0.1319，正好吻合。

启动时跑 8000 步（0.8 秒）让物理自己把果蝇放到地面上。

### 抽搐的本质是变化，不是水平

**第一版是错的**：直接拿运动神经元群的**平均活动**当关节力矩，结果果蝇摆一个
固定姿势不动，而且**画面亮面越大越像标本**——翅膀那两路只有 33、34 个细胞，
被灌满后平均活动恒为 2.1，钳死在最大张开角。

实测（Bad Apple 90 帧）：

| | 旧：用活动水平 | 新：用归一化偏离 |
|---|---|---|
| 翅膀左 | 均值 0.994，**98.9% 饱和** | 全程 [−1, 1] 摆动 |
| 翅膀右 | 均值 1.000，**100% 饱和** | 全程 [−1, 1] 摆动 |
| 8 路整体标准差 | 0.305 | **0.533** |
| 每帧在动的自由度 | — | **109 个里的 100 个** |

修法见 [`flyscreen/body/driver.py`](flyscreen/body/driver.py)：每路对自己的**慢基线**
做 z-score，只用"偏离基线的部分"当驱动，再混一个活动变化率分量。

### 驱动幅度决定它是"踉跄"还是"翻车"

| `--body-travel` | 30 秒内倒下 | 自己起身 | 漂移 | 观感 |
|---|---|---|---|---|
| 0.10 ~ 0.30 | 0 | — | 0.05 cm | 站着微动 |
| 0.38 | 0 | — | 0.12 cm | 站着踉跄 |
| 0.50 | 1 | 0 | 2.9 cm | 倒下 → 蹬腿 → **自己翻回来** |
| **1.00（默认）** | **8** | 0 | 2.8 cm | 疯狂抽搐、满地打滚 |

全部 30 秒无 NaN、无发散。

---

## 渲染：三层分色

| 层 | 颜色 | 含义 |
|---|---|---|
| 结构层 | 暗蓝 | 未点亮的神经元 —— 保留果蝇脑的形状 |
| 驱动层 | 冷色（青白） | 视频直接点亮的 —— **你给它的** |
| 脉冲层 | 暖色（琥珀） | 连接组传播出来的 —— **它自己的反应** |

分色不是为了好看，是为了诚实：一眼能看出哪些是输入、哪些是网络的回应。

---

## 性能

单帧各环节实测（600×490 画布 + 460×490 身体，16 线程 CPU，1080p 源）：

| 环节 | 优化前 | 优化后 |
|---|---|---|
| ffmpeg 解帧 | ~0 | ~0 |
| 帧 → 神经元驱动 | 13.5 ms | 13.5 ms |
| LIF × 33 tick | 35 ms | 35 ms |
| **点云渲染** | **106 ms** | **37 ms** |
| 身体物理 + 渲染 | 12 ms | 12 ms |
| 合成 | 21 ms | ~5 ms |
| JPEG 编码（WebUI 独有） | 4.5 ms | 4.5 ms |
| | **≈3.9 fps** | **≈6.0 fps** |

LIF 单步（numba）：**1.26 ms**（纯 numpy 回退 8.40 ms，快 6.7 倍且脉冲数逐位相同）。

优化点：结构层静态化、跳过 `np.power(v,1.0)`、**颜色平面预分配**
（原来每层新建两个 3.5 MB 的 `(h,w,3)` 数组，全新内存页的缺页中断把
`_layer` 拖到 12 ms）、面板尺寸与渲染输出对齐走 memcpy。

**再快就调小画布**（开销随像素数线性），或者面板里把「辉光半径」调到 0。

---

## 踩过的坑

按发现顺序，全部有实测数据支撑：

| # | 坑 | 现象 |
|---|---|---|
| 1 | `_layer` 的 `lo + (hi-lo)*v` 在 v=0 时返回 `lo` 而不是黑 | 驱动层 (8,44,70) + 脉冲层 (60,18,0) = **(68,62,70)**，本该全黑的底被刷成灰色，冷暖分色全糊掉 |
| 2 | JPEG 默认 4:2:0 色度二次采样 | 点云是高频噪点，色度被整个丢掉 → 整个脑变灰。必须 `subsampling=0` |
| 3 | 用**活动水平**当关节力矩 | 果蝇摆固定姿势，画面越亮越像标本 |
| 4 | 换片源时没重建帧生成器 | 旧生成器在已 `close()` 的进程上读帧 → 崩溃 |
| 5 | `VideoSource` 默认 `loops=True` | 生成器内部自己转圈永远不结束，录了 581 帧还在录（片源只有 360 帧） |
| 6 | 源视频没音轨时 `-map 1:a:0` | ffmpeg 整体报错。要写 `-map 1:a:0?` 或先 `probe()` |
| 7 | 地板只有 6×6 厘米 | 果蝇被抽得 10 秒滑出 6 cm，**跑到地板外面悬空** → 场景里没有地板 |
| 8 | 灯光固定在世界坐标 | 果蝇跑远了脚下地板照不到 → 全黑 |
| 9 | 相机竖直方向跟着果蝇 | 它往上飞（root z 能到 0.38），相机跟着抬，**地板被挤出画面** |
| 10 | 地板格子色差只有 0.085 | 光照一衰减就分不出来，读起来是块暗底 |
| 11 | `mark="edge"` 画细网格线 | texrepeat=400 时线宽不到一个像素，被 mipmap 滤没了 |
| 12 | `Start-Process -ArgumentList` 传中文路径 | 被拆坏。改用 `FLYSCREEN_VIDEO` 环境变量 |
| 13 | MJPEG 流永远不结束 | `Invoke-WebRequest` 会一直挂着。用 `curl --max-time 3` 测 |

---

## 诚实说明：哪些是真的，哪些是手搓的

**来自真实数据（涌现的）**

- 141,781 个神经元的胞体坐标 —— 电子显微镜重建，8nm 体素
- 5,536,527 条突触连接与突触数
- 8 个部位的归属 —— 连接组 `subclass` 里的 `fl/ml/hl/wm`（前/中/后腿 + 翅运动神经元）
  加 `somaSide` 左右，全部是解剖标注
- 左侧神经元驱动左腿、右侧驱动右腿
- LIF 脉冲沿真实连接组的传播

**人工定义的（手搓的）**

- **视频像素 → 神经元的对应关系。** 生物果蝇没有"把画面铺在脑子上"这回事，
  这是艺术处理。默认按胞体坐标投影铺满，映射本身是虚构的。
- **神经元活动 → 关节力矩。** 连接组里没有"哪根神经管哪块肌肉"的对照表
  （扫描只扫了神经，没扫肌肉），所以直接用运动神经元群的**归一化活动**当力矩。
  增益、慢基线时间常数、平滑时间常数全是调的。
- **E/I 平衡倍率 1.526 与默认增益 0.30** —— 见上文，参考实现用的是 1.0 和 0.65。
- **灯光的数量、位置、颜色**，地面的颜色和大小。
- **相机角度与跟踪策略。**

**已知的"不真"**

- 神经元模型是**均匀 LIF**：所有神经元共用同一套参数，没有形态、离子通道差异，
  权重冻结、不会学习。
- 连接组只保留 ≥5 突触的连接（占全部连接的 5%、突触总量的 31%）。
- 递质预测方法的文献准确率是 **88.6%** —— 即约 11% 的神经元符号可能标反，
  而每个标错的神经元**全部**出边都反。
- 果蝇身体的肌肉没有逐块建模，8 个部位 = 8 组执行器。

**尚未实现**

- 复眼视网膜编码（把视频投到 877+892 个 medulla 柱上，而不是铺满整个脑）
- 果蝇色觉：R1-R6 宽带 / R7 紫外 / R8 绿 —— **果蝇看不见红蓝**
- 突触级渲染（需要 `syn-partners.feather`，6.8 GB）

---

## 不同片源的实测结果

同一管线，片源性质不同结果完全不同——这不是 bug，是"把视频铺在脑子上"的必然。

| 片源 | 驱动神经元 | 结果 |
|---|---|---|
| **高对比剪影**（Bad Apple、影绘 PV） | 几千 | ✅ **最佳**。剪影清晰地嵌在脑点云里，同时脑解剖结构完整保留 |
| 测试图案（纯黑底 + 亮块） | ~3,000 | ✅ 清楚 |
| 实拍视频（有黑背景） | 几万 | 🟡 整片脑亮起来，能看出亮度分布，认不出"画面" |
| 实拍视频（中间调多） | **13.8 万（97%）** | 🟡 整个脑都亮。副产品是**脑的解剖纹理全出来了**；可用 `--black-point 0.25` 压中间调 |

---

## 目录结构

```
flyscreen/
├── fly.py                **统一入口**（web / render / preview / check）
├── webui.py              Web 界面实现（约 410 行，参数更全）
├── run.py                命令行出片实现（约 410 行，参数更全）
├── requirements.txt
├── README.md
├── flyscreen/
│   ├── config.py         全部可调参数（dataclass）
│   ├── video.py          任意视频 → 灰度帧（ffmpeg 子进程）
│   ├── writer.py         帧 → mp4 / 混音（ffmpeg 子进程）
│   ├── canvas.py         视频帧 → 神经元驱动（投影 + 密度均衡 + 拖尾）
│   ├── render.py         点云渲染（三层分色 + 辉光）
│   ├── compose.py        画面布局
│   ├── ui.py             pygame 预览窗
│   ├── data/
│   │   ├── fetch.py          下载原始数据
│   │   ├── prepare.py        注释 → 神经元表缓存
│   │   ├── prepare_graph.py  连接表 → CSR 图缓存
│   │   └── dataset.py        加载缓存
│   ├── brain/lif.py      全脑 LIF（numba JIT + numpy 回退）
│   ├── body/
│   │   ├── fly.py        MuJoCo 果蝇身体（地面/悬空两种模式）
│   │   └── driver.py     8 路驱动信号整形
│   └── web/server.py     WebUI（纯标准库 HTTP + MJPEG + 参数面板）
├── scripts/
│   ├── inspect_data.py   把三个原始表格的真实内容打印出来（学习用）
│   ├── make_test_video.py 造测试视频
│   ├── fetch_flybody.py   下载果蝇身体模型
│   ├── repair_flybody.py  按大小校验并补下
│   └── verify_data.py     数据完整性自检
├── data/
│   ├── raw/              1.06 GB  三个原始 feather
│   ├── cache/             44 MB   neurons.npz + graph.npz
│   ├── assets/flybody/   137 MB   92 个文件（MJCF + 网格）
│   ├── uploads/                   Web 页面拖进来的视频
│   └── logs/                      WebUI 的 stdout/stderr（用 `-u` 启动才实时落盘）
└── out/                          成片输出
```

代码总计约 3,500 行。依赖只有 `numpy / pyarrow / pygame / mujoco / pillow / numba`，
视频读写走系统 `ffmpeg`。**WebUI 是纯标准库实现**，不需要 Flask/FastAPI/websockets。

查看原始数据长什么样：

```powershell
python scripts/inspect_data.py
```

---

## 数据来源与许可

**连接组（脑）** —— [MaleCNS v1.0](https://male-cns.janelia.org/download/)，**CC BY 4.0**

> Berg S, Beckett IR, Costa M, Schlegel P, Januszewski M, Marin EC, Nern A, Preibisch S,
> Qiu W, Takemura S, et al. **Sexual dimorphism in the complete Drosophila male central
> nervous system connectome.** *Cell* 189(18):5504–5526.e15 (2026).
> https://doi.org/10.1016/j.cell.2026.08.015

由 HHMI Janelia FlyEM、University of Cambridge、MRC Laboratory of Molecular Biology
和 Google Research 合作完成。

**动力学模型** —— Shiu et al., *Nature* 634:210（2024）

> Shiu PK, Sterne GR, Spiller N, et al. **A Drosophila computational brain model reveals
> sensorimotor processing.** https://doi.org/10.1038/s41586-024-07763-9

**身体** —— [flybody](https://github.com/TuragaLab/flybody)，**Apache-2.0**，
经 [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) 分发

> Vaxenburg R, Siwanowicz I, Merel J, Robie AA, Morrow C, Novati G, Stefanidi Z,
> Card GM, Reiser MB, Botvinick MM, Branson KM, Tassa Y, Turaga SC.
> **Whole-body simulation of realistic fruit fly locomotion with deep reinforcement
> learning.** bioRxiv (2024). https://doi.org/10.1101/2024.03.11.584515

**本项目代码**：MIT。

本项目与 HHMI、Janelia、Google、DeepMind、Cambridge、MRC 均无隶属关系，也未获其背书。
