"""把 raw 三个表格的真实内容打出来，看清它们长什么样、怎么串起来。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.feather as feather

RAW = Path(r"D:\fly-brain\flyscreen\data\raw")


def line(c="-", n=96):
    print(c * n)


# ============================================================ 表 1
line("=")
print("表 1  body-annotations.feather   神经元名册")
line("=")
t = feather.read_table(RAW / "body-annotations.feather")
print(f"形状：{t.num_rows:,} 行 × {len(t.column_names)} 列")
print("完整列名：")
for i, c in enumerate(t.column_names):
    print(f"   {i:2d}. {c}")
print()

cols = ["bodyId", "type", "class", "subclass", "superclass",
        "somaSide", "somaNeuromere", "exitNerve", "somaLocation", "status"]
print("挑 6 个神经元（每列 = 一个属性，每行 = 一个神经元）：")
print()
hdr = f"{'bodyId':>8} {'type':<12} {'superclass':<20} {'subclass':<10} " \
      f"{'side':<5} {'neuromere':<10} {'somaLocation':<22} {'status':<10}"
print(hdr)
line()
import random
random.seed(7)
idx = sorted(random.sample(range(t.num_rows), 6))
for i in idx:
    loc = t["somaLocation"][i].as_py()
    locs = "None" if loc is None else str(list(loc))
    print(f"{t['bodyId'][i].as_py():>8} {str(t['type'][i].as_py())[:11]:<12} "
          f"{str(t['superclass'][i].as_py())[:19]:<20} "
          f"{str(t['subclass'][i].as_py())[:9]:<10} "
          f"{str(t['somaSide'][i].as_py())[:4]:<5} "
          f"{str(t['somaNeuromere'][i].as_py())[:9]:<10} "
          f"{locs[:21]:<22} {str(t['status'][i].as_py())[:9]:<10}")

print()
print("注意 somaLocation 是 [x, y, z] 三个整数，单位是 8 纳米：")
for i in idx[:3]:
    loc = t["somaLocation"][i].as_py()
    if loc:
        print(f"   bodyId {t['bodyId'][i].as_py()}: 体素 {list(loc)}  "
              f"-> 微米 {[round(v*0.008,1) for v in loc]}")

# ============================================================ 表 2
print()
line("=")
print("表 2  body-neurotransmitters.feather   神经递质预测")
line("=")
nt = feather.read_table(RAW / "body-neurotransmitters.feather")
print(f"形状：{nt.num_rows:,} 行 × {len(nt.column_names)} 列")
print(f"列名：{nt.column_names}")
print()
print("同样是每行一个神经元，但只有 6 列有用：")
print()
print(f"{'body':>8} {'cell_type':<12} {'predicted_nt':<16} {'consensus_nt':<16} "
      f"{'pred_conf':>9}")
line()
body2row = {}
b = nt["body"].to_numpy(zero_copy_only=False)
for i in range(nt.num_rows):
    body2row[int(b[i])] = i

for i in idx[:6]:
    bid = t["bodyId"][i].as_py()
    if bid in body2row:
        j = body2row[bid]
        print(f"{bid:>8} {str(nt['cell_type'][j].as_py())[:11]:<12} "
              f"{str(nt['predicted_nt'][j].as_py())[:15]:<16} "
              f"{str(nt['consensus_nt'][j].as_py())[:15]:<16} "
              f"{nt['predicted_nt_confidence'][j].as_py():>9.3f}")
    else:
        print(f"{bid:>8} (这个 bodyId 不在递质表里)")

print()
print("递质种类统计（决定兴奋 / 抑制）：")
u, c = np.unique(np.array([str(x) for x in nt["consensus_nt"].to_numpy(zero_copy_only=False)]),
                 return_counts=True)
for i in np.argsort(-c)[:10]:
    sign = "抑制" if u[i] in ("gaba", "glutamate", "histamine") else "兴奋"
    print(f"   {u[i]:<18} {c[i]:>9,d}  -> {sign}")

# ============================================================ 表 3
print()
line("=")
print("表 3  connectome-weights.feather   接线表（谁连谁、几个突触）")
line("=")
d = ds.dataset(str(RAW / "connectome-weights.feather"), format="feather")
print("只有 3 列：body_pre（发出方）、body_post（接收方）、weight（突触数）")
n_rows = 0
sample = []
for bt in d.scanner(columns=["body_pre", "body_post", "weight"], batch_size=64).to_batches():
    if not sample:
        sample = [(bt.column("body_pre")[k].as_py(),
                   bt.column("body_post")[k].as_py(),
                   bt.column("weight")[k].as_py()) for k in range(8)]
    break
print()
print(f"{'body_pre':>12} {'body_post':>14} {'weight':>8}")
line()
for p, q, w in sample:
    print(f"{p:>12,} {q:>14,} {w:>8}")

print()
print("读法：第 1 行表示 —— 神经元 10001 向神经元 X 伸了 N 个突触。")
print("整个文件就是这种行，一共 151,856,684 行。")
print()
print("把 weight 理解成『这条连线有多粗』：")
print("   1 个突触   = 很细的线，可能传不过去")
print("   50 个突触  = 粗线，刺激一下就够兴奋对方")
print("   2591 个突触 = 最粗的线（整个数据集最大值）")

# 追一个神经元
print()
line("=")
print("三个表怎么串起来：追一个具体神经元")
line("=")
TARGET = 10001
i = None
ids = t["bodyId"].to_numpy(zero_copy_only=False)
hit = np.flatnonzero(ids == TARGET)
print(f"\n① 在表 1 里查 bodyId = {TARGET}:")
if hit.size:
    i = int(hit[0])
    j = body2row.get(TARGET)
    print(f"     type       = {t['type'][i].as_py()}")
    print(f"     superclass = {t['superclass'][i].as_py()}")
    print(f"     subclass   = {t['subclass'][i].as_py()}")
    print(f"     胞体坐标    = {t['somaLocation'][i].as_py()}")
    print(f"\n② 拿同一个 bodyId 去表 2 查它的递质：")
    if j is not None:
        nm = str(nt["consensus_nt"][j].as_py())
        sign = "抑制(-)" if nm in ("gaba", "glutamate", "histamine") else "兴奋(+)"
        print(f"     consensus_nt = {nm}  ->  {sign}")

print(f"\n③ 拿同一个 bodyId 去表 3 查它的连线（前 10 条）：")
cnt = 0
out_sum = 0
for bt in d.scanner(columns=["body_pre", "body_post", "weight"], batch_size=2_000_000).to_batches():
    p = bt.column("body_pre").to_numpy(zero_copy_only=False)
    m = p == TARGET
    if m.any():
        q = bt.column("body_post").to_numpy(zero_copy_only=False)[m]
        w = bt.column("weight").to_numpy(zero_copy_only=False)[m]
        out_sum += int(w.sum())
        for k in range(min(10 - cnt, len(q))):
            print(f"     -> bodyId {q[k]:>10,}   突触 {w[k]}")
        cnt += int(m.sum())
    if cnt >= 10:
        break
print(f"     （这个神经元总出边数不只 10 条，总突触权重 {out_sum:,}）")

line("=")
print("一句话总结这三个表的关系：")
line("=")
print("""
   表1 (名册)   每行 = 一个神经元   写着 名字/类型/位置/左右
   表2 (体检)   每行 = 一个神经元   写着 它是兴奋的还是抑制的
   表3 (接线)   每行 = 一条连接     写着 谁连谁、连得多粗

   三张表的公共钥匙就是 bodyId（神经元编号）。
   表1、表2 是"每个神经元一行"，表3 是"每条连线一行"。
""")
