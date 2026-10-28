# π0.5 四任务持续学习：协议 v2

本实验坚持使用 pi05_base 机器人预训练权重。四个 LIBERO-10 任务作为 pilot，
先比较不回放与回放 20%，不宣称复现论文的 π0 / 完整十任务结果。
旧版固定 T1 归一化的结果与本协议不混用，旧 100 步诊断模型也不作为正式 T1 起点。

## 两组主实验

| 配置 | sequential | replay |
|---|---|---|
| 初始权重 | 相同 pi05_base | 相同 pi05_base |
| 后续阶段 | 继承上一阶段全部权重及 LoRA | 同左 |
| 旧数据 | 无 | 每个旧任务训练样本起点的 20%，向下取整 |
| 每个 batch | 全部当前任务 | 一半当前任务，一半旧任务 |
| 旧任务之间 | 不适用 | 等概率选任务，再在该任务缓冲区内均匀采样 |
| 归一化 | 每任务独立统计且固定 | 同左；回放使用样本所属任务的统计量 |
| 视觉 | 默认全量微调 | 相同 |
| 语言/动作 Transformer | 原始权重冻结，训练同一组 LoRA | 相同 |
| 更新预算 | 每阶段 10000 次，batch=8 | 相同 |

20% 是缓冲区保存比例，50% 是每批训练采样比例，两者不同。
与论文主设置固定 1000 起点/旧任务相比，本实验遵循用户要求，按各任务训练集大小取 20%。
训练种子同时控制模型随机性与回放抽样；每个种子内缓冲区一旦选定，后续阶段不重新抽样。
每个旧任务的采样机会相同，不因任务数据量较大而占更多训练机会。

| 任务 | 训练轨迹 | 验证轨迹 | 训练起点 | 20% 回放起点 |
|---|---:|---:|---:|---:|
| T1 书放入收纳盒 | 32 | 9 | 5947 | 1189 |
| T2 开炉并放摩卡壶 | 32 | 9 | 8520 | 1704 |
| T3 杯子放微波炉并关门 | 27 | 7 | 7931 | 1586 |
| T4 碗放抽屉并关抽屉 | 28 | 7 | 6847 | 1369 |

T4 训练时只回放 T1、T2、T3，共 4479 个起点。T4 的 1369 是将来继续学习时的预算。
一个起点包含图像/指令及后续 10 步动作块，不能称为仅保存一个标量动作。
随机 20% 起点可能覆盖明显超过 20% 的独立动作标签，manifest 同时记录实际标签覆盖率。
底层完整数据仍留在数据盘，训练通过固定索引限制访问；这是回放协议模拟，不是磁盘数据销毁方案。

## 归一化与数据隔离

作者公开代码按任务分别统计、测试时按任务选择统计量。本实现采用同样的任务隔离方式。
每个任务仅用 cl/splits.json 中的训练轨迹计算；验证轨迹不进入训练、统计或回放。
动作统计覆盖与训练一致的十步动作块及轨迹末尾重复填充；图像不参与数值统计。
统计文件只生成一次，拒绝覆盖；保存划分、原始数值数据、统计文件的 SHA256 与统计规则。
各训练阶段仅要求已学任务的统计文件，未来任务文件可尚不存在。
检查点携带所有已学任务的统计量；跨阶段与评测前核对哈希，拒绝静默替换。

π0.5 默认保留 OpenPI 的 q01/q99 分位数缩放；不裁剪超范围动作。
作者 π0 的常规配置采用均值/标准差缩放，因此数值公式并非完全一致。
若比较该差异，两组都显式传 --normalization zscore，并使用另一组输出目录从头训练。
不要将不同归一化公式的结果混合，也不要把新的统计量直接套到旧模型。

## 模型与预算

- action_horizon=10，执行前 5 步后重新规划，流匹配采样步数=10。
- 沿用 π0.5 LIBERO 的 discrete_state_input=False，不宣称使用了本体状态条件。
- 语言 LoRA rank/alpha=16/16；动作 LoRA=32/32；训练动作投影与时间 MLP。
- AdamW：b1=0.9、b2=0.95、eps=1e-8、weight_decay=1e-10、梯度裁剪=1；EMA 关闭。
- LR 峰值 2.5e-5，500 步 warmup，余弦降到 2.5e-6。
- 每阶段重置优化器与调度器，只继承模型；同阶段 --resume 同时恢复优化器。
- 恢复训练不保证恢复数据迭代器的精确位置，不宣称逐位可复现。
- 默认视觉全量微调、batch=8 对齐论文经验，但尚未验证当前 GPU 空闲量足够。
- 小显存工程测试可在两组同时使用 --freeze-vision --batch-size 2；这是独立实验设置。
- 相同更新次数下，回放组当前任务曝光量约减半；不把这称为相同当前任务数据曝光。
- 正式比较使用 42、43、44 三个训练种子；先完成 seed=42 的一对实验验证流程。
- 每 1000 步保存，但每阶段仅保留最新一份；完整四阶段保留四份阶段末模型。
- 10000 次更新对应检查点 9999；主结果用固定阶段末模型，不能按测试表现挑选。

## 数据盘与准备

所有命令从仓库根目录执行，训练使用项目 .venv/bin/python。
缓存、临时文件、日志、检查点、统计量、视频都放入以下运行根目录。
启动训练前默认要求至少 100 GiB 可用；保存前至少 20 GiB，且不得位于系统文件系统。

```bash
PY=.venv/bin/python
DATA=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/libero_data
BASE=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cache_test/pi05_base_params/params
RUNTIME=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime
NORM="$RUNTIME/reports/per_task_norm_v1"
run_disk() { "$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- "$@"; }
# 当前服务器已生成 T1--T4；不重复运行下面的生成命令。
# 新部署按任务生成也可加 --tasks T1，之后再生成 T2 等。
JAX_PLATFORMS=cpu run_disk "$PY" -B -m experiments.continual.normalization --dataset-root "$DATA" --output "$NORM"
```

## 两组训练

将 METHOD 设为 sequential 或 replay；其他参数相同。
默认 replay-fraction=0.2。保留 --replay-budget 1000 作为固定预算对照，主实验不用它。
每组、每个种子使用独立 OUT；不要中途改变批量、视觉策略或归一化。

```bash
METHOD=sequential   # 另一组改成 replay
SEED=42
OUT="$RUNTIME/checkpoints/v2_${METHOD}_seed${SEED}"
COMMON=(--method "$METHOD" --seed "$SEED" --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" --output "$OUT")
run_disk "$PY" -B -m experiments.continual.run plan --stage 1 "${COMMON[@]}"
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run train --stage 1 "${COMMON[@]}"
# 先完成 T1 评测，确认已经学会，再继续。
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run train --stage 2 --previous "$OUT/cl_pi05_continual_v2/T1/9999" "${COMMON[@]}"
# T3/T4 同理，分别指向紧邻前一阶段的阶段末检查点。
```

默认目录名在不传 --output 时自动包含方法、回放比例、种子、视觉策略、归一化与 batch。
工程 smoke 使用独立 OUT 和 --steps 5 --warmup 2，不将其结果接入正式阶段。
不自动抢占显卡或启动四阶段长训练，先确认资源及 T1 rollout。

## 评测

服务器已建立独立 Python 3.10 LIBERO 环境，examples/libero/.venv 指向数据盘。
四任务 CPU 渲染/步进检查通过；重建方式见 SIMULATION.md，不把仿真依赖装入训练 .venv。
新入口自动读取训练 manifest，逐个启动已学任务的服务，选择该任务的归一化，然后评测。
同一阶段所有任务使用同一份模型；结束任务后释放服务与模拟器。

```bash
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.evaluate \
  --checkpoint "$OUT/cl_pi05_continual_v2/T2/9999" --norm-assets "$NORM" \
  --sim-python examples/libero/.venv/bin/python
```

对 T1/T2/T3/T4 分别运行一次，自动产生 1+2+3+4=10 个单元格的结果文件。
默认每任务 40 回合，初始状态编号 10..49，环境 seed=7；两个方法完全相同。
0..9 仅用于调参。这样的保留测试集是本 pilot 的设计，不声称复刻作者 50 回合设置。
环境初始状态相同不代表策略噪声逐回合完全匹配；服务重启时采用 OpenPI 默认策略 RNG。
每个任务默认只保留前 3 个视频，--videos-per-task 0 可关闭；所有回合都保留结构化结果。
任务、阶段、检查点、统计量必须匹配，否则在 rollout 前拒绝运行。
环境/服务错误直接失败，部分结果不能混入完整测评；修复后添加 --run-tag retry1 重跑，汇总时只选择该次完整结果。

## 汇总标准

```bash
run_disk "$PY" -B -m experiments.continual.analyze metrics \
  --results "$RUNTIME"/reports/v2_sequential_seed42_T*_offset10_n40/T*.jsonl \
  --output "$RUNTIME/reports/sequential_seed42_metrics.json"
```

同一份汇总只包含一个方法、一个训练种子、同一评测设置。
拒绝重复回合、不完整矩阵、不匹配的初始状态编号、跨任务统计量及阶段内混用模型。
输出完整成功率矩阵、每格 Wilson 95% 区间、对角线成功率、最终平均成功率和最终遗忘。
还输出每个旧任务跨所有后续阶段平均的 NBT，并保留负值表示改善。
论文公式的最后任务没有后续阶段，存在空均值；为避免隐含约定，明确同时报告：
1. 对前三个任务平均的 NBT；2. 最后任务记零后对四任务平均的 NBT。
归一化 NBT 对初始成功率为零的任务返回 null，不用微小分母伪造可比数值。
不能只凭低遗忘认定有效：必须同时检查任务最初是否学会、最后成功率是否足够高。
跨种子报告均值与样本标准差；一个种子只是 pilot，不作为稳定结论。

## 审计与验证

```bash
JAX_PLATFORMS=cpu run_disk "$PY" -B -m experiments.continual.analyze audit --dataset-root "$DATA" --norm-assets "$NORM" --output "$RUNTIME/reports/new_norm_audit.json"
JAX_PLATFORMS=cpu run_disk "$PY" -B -m experiments.continual.verify_data --dataset-root "$DATA" --norm-assets "$NORM"
JAX_PLATFORMS=cpu run_disk "$PY" -B -m pytest -p no:cacheprovider experiments/continual scripts/train_test.py -q
```

审计按任务输出训练数量、原始范围、归一化范围及超范围比例。
quantile 模式超范围阈值为 abs(z)>1.01；zscore 模式为 abs(z)>3。
统计过程和验证命令不调用 GPU；完整策略 rollout 与真实 π0.5 GPU 训练需要另行验收。

参考：
- https://arxiv.org/html/2603.03818v1
- https://github.com/Continual-VLAs/continual-openpi/blob/main/scripts/compute_norm_stats.py
- https://github.com/Continual-VLAs/continual-openpi/blob/main/src/openpi/policies/policy_config.py
- https://github.com/Continual-VLAs/continual-openpi/blob/main/examples/libero/correct_eval/eval_continual_per_task.py

两组及多种子的配对比较：

```bash
run_disk "$PY" -B -m experiments.continual.compare \
  --sequential "$RUNTIME"/reports/sequential_seed*_metrics.json \
  --replay "$RUNTIME"/reports/replay_seed*_metrics.json \
  --output "$RUNTIME/reports/sequential_vs_replay20.json"
```

配对比较还核对初始状态编号与统计量哈希；只有匹配的训练种子才能配对。
成功率差值为正代表回放更好，遗忘/NBT 差值为负代表回放更好。
单种子标准差返回 null，不据此声称统计显著性。
