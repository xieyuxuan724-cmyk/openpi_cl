# π0.5 四任务持续学习实验协议

本目录将四任务试验定义为独立的 pilot，不宣称复现完整 LIBERO benchmark。
旧的 cl_pi05_t1_lora_smoke 仅用于工程调试；正式训练使用本目录入口。
不复用旧的 100 步诊断模型作为 T1 正式起点。

## 固定协议

- T1–T4 的任务和按轨迹 train/val 划分读取 cl/splits.json，禁止把验证轨迹加入训练或回放。
- 初始模型为 pi05_base 完整机器人预训练参数；所有阶段沿用同一组 LoRA。
- 语言 LoRA rank/alpha=16/16，动作 LoRA=32/32，位置为注意力和 FFN。
- 默认全程冻结视觉与两个 Transformer 的原始参数；训练 LoRA、动作投影和时间 MLP。
- 沿用官方 LIBERO 的 discrete_state_input=False：本实验不使用本体状态条件。不要误称状态已输入网络。
- action_horizon=10，执行前 5 步后重新规划，流匹配推理步数=10。
- 使用项目 .venv/bin/python，单主机 JAX，默认 batch=2、seed=42、每阶段 10000 次更新。
- AdamW 沿用官方默认，梯度裁剪=1；EMA 关闭。峰值 LR=2.5e-5，500 步 warmup，至阶段末降到 2.5e-6。
- 所有阶段固定 T1 训练集归一化；不得使用未来任务重新估计训练统计量。数据审计只输出诊断，不改变训练归一化。
- 每阶段重置优化器和调度器；同阶段中断使用 --resume。恢复模型/优化器不保证恢复数据迭代器的精确位置。
- 每 1000 步保存一次用于恢复；max_to_keep=1、keep_period=None，仅保留该阶段最新检查点，旧快照由检查点管理器清理。每阶段独立目录，完整四阶段最终留下四份模型。
- 主结果使用阶段末检查点。当前 trainer 从零编号，因此 10000 次更新对应目录 9999。
- 参数文件由官方检查点保存完整模型，不单独导出 LoRA。
- 不自动启动长训练。先计划检查，再运行训练。

## 两个首轮实验

sequential：只训练当前任务，衡量自然遗忘。
replay：每个旧任务固定 100 个随机 transition 起点，不重复采样入库，后续阶段保持同一缓冲区。
采样概率为当前任务总计 50%、旧任务总计 50%；这是期望比例，小 batch 不保证逐 batch 精确一半。
等容量旧任务缓冲区使旧任务之间均匀。缓冲区随已学任务数线性增长，并非全程固定总内存。
transition 起点需要对应最多 10 帧动作标签；报告起点数和标签帧上界，不能把它称为只存储 100 个标量动作。
当前实现通过索引限制训练访问，底层仍保留完整数据文件；不是物理删除历史数据的存储系统。
两方法固定更新数和 batch，回放组当前任务曝光量约减半；报告这一差异。
后续才增加视觉全量微调消融，全序列固定策略，不中途解冻。

## 执行示例

从仓库根目录运行；变量为服务器已有目录。
每组实验使用独立 OUT，正式多种子实验分别使用 42、43、44。

```bash
PY=.venv/bin/python
DATA=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/libero_data
BASE=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cache_test/pi05_base_params/params
NORM=assets/cl_t1_norm_stats
RUNTIME=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime
OUT="$RUNTIME/checkpoints/sequential_seed42_frozen_vision"
run_disk() { "$PY" -B -m experiments.continual.storage --storage-root "$RUNTIME" -- "$@"; }
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run plan --stage 1 --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" --output "$OUT"
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run train --stage 1 --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" --output "$OUT"
# T2: repeat all protocol options; T3/T4 similarly use the immediately previous stage.
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run train --stage 2 --previous "$OUT/cl_pi05_t1_lora_smoke/T1/9999" --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" --output "$OUT"
```

replay 组全阶段添加 --method replay；视觉消融全阶段添加 --unfreeze-vision。
训练前可用独立 OUT 做 --steps 5 --warmup 2 的工程测试；禁止把该检查点混入正式阶段。
正式运行保存 manifest.json（配置身份、commit、解释器、统计量/划分哈希、回放索引）。
修改代码后先提交再跑正式实验；保存环境版本及启动命令。
首轮先测 T1 任务成功率；若几乎为零，在验证环境调参数，确认学会后冻结方案再跑完整序列。
10000 步是候选预算，不是已经验证收敛的结论。

## 数据量审计

```bash
run_disk "$PY" -B -m experiments.continual.analyze audit --dataset-root "$DATA" --norm-file "$NORM/physical-intelligence/libero/norm_stats.json" --output "$RUNTIME/reports/data_audit.json"
```

输出轨迹数、帧数、动作范围、固定归一化后的范围及超过 1.01 的逐维比例。
1.01 容差用于避免分位数插值使离散夹爪动作略超 1 的假警报；不裁剪动作。
有效 epoch 为更新次数×batch/训练帧数，只是曝光量参考，动作 chunk 会重叠。
正式还需记录 wall time、GPU 型号和峰值显存；损失下降不能替代 rollout。

## 评测与统计

模拟器使用独立 examples/libero/.venv，按 examples/libero/README.md 的依赖说明创建；不要把旧版模拟器依赖直接装入训练环境。当前训练 .venv 不含 libero，完整 rollout 尚未验收。

用 run serve 指定同一组全部配置和 --checkpoint；T2–T4 同样提供 --previous。
serve 固定流匹配步数=10，检查 manifest，避免误用官方已微调的 pi05_libero。

```bash
CUDA_VISIBLE_DEVICES=0 run_disk "$PY" -B -m experiments.continual.run serve --stage 1 --dataset-root "$DATA" --base-params "$BASE" --norm-assets "$NORM" --output "$OUT" --checkpoint "$OUT/cl_pi05_t1_lora_smoke/T1/9999"
# 另一个终端；本组四任务属于 libero_10，脚本会验证任务文本确实存在。
run_disk examples/libero/.venv/bin/python examples/libero/main.py --task-suite-name libero_10 --task-keys T1 --num-trials-per-task 40 --initial-state-offset 10 --stage 1 --checkpoint-label "$OUT/cl_pi05_t1_lora_smoke/T1/9999" --result-path "$RUNTIME/reports/eval_T1.jsonl" --video-out-path "$RUNTIME/videos/T1"
```

T2 后传 --task-keys T1 T2，依此类推；阶段末必须构成 10 个有效单元格。
固定环境 seed=7，首轮测试使用初始状态编号 10..49（40 回合）；调参仅用 0..9，工程测试 5 回合。脚本支持 offset，强制检查索引范围。
推理服务使用自身 RNG；环境 seed 不等于逐回合扩散噪声匹配，禁止宣称完全相同推理噪声。
校准/调参用编号 0..9 的初始状态集合；编号 10..49 的测试回合只用于报告，禁止按测试结果选检查点。
环境异常直接中止，不能当作策略失败。异常产生的部分结果必须隔离后重跑到新路径。
各训练种子单独汇总，再报告跨种子均值/标准差；40 回合只有 2.5 个百分点分辨率，勿将小差异当作确定提升。

```bash
run_disk "$PY" -B -m experiments.continual.analyze metrics --results "$RUNTIME/reports/eval_T1.jsonl" "$RUNTIME/reports/eval_T2.jsonl" "$RUNTIME/reports/eval_T3.jsonl" "$RUNTIME/reports/eval_T4.jsonl" --output "$RUNTIME/reports/metrics.json"
```

R[s,t] = 学完阶段 s 后在任务 t 的成功率。
报告完整矩阵、对角线平均成功率、T4 后四任务平均成功率、
最终遗忘 mean(t=1..3)(R[t,t]-R[4,t])。负值保留，代表改善；这不是论文跨所有后续阶段平均的 NBT。
低遗忘必须与新任务成功率一起解释。可额外评估训练前模型，但不要为缺失零样本单元格填零。
本轮未自动运行 GPU 训练、环境 rollout 或保存重载动作一致性测试，这些是启动正式实验前的验收项。

参考：https://arxiv.org/abs/2603.03818

## 数据盘存储规范

默认根目录：/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime。
可通过 --storage-root 或 OPENPI_CL_STORAGE_ROOT 显式指定其他非系统文件系统。
数据集、预训练权重和只读归一化文件仍从原路径读取，不重复搬运。

| 子目录 | 内容 |
|---|---|
| tmp/ | Python、数据加载和第三方库临时文件 |
| cache/ | JAX、Hugging Face、OpenPI、PyTorch、Triton、CUDA 等缓存 |
| checkpoints/ | 按方法、种子、视觉策略分组的阶段检查点 |
| logs/ | 每次启动独立 stdout/stderr 日志与 JSON 运行记录；时间戳为 UTC |
| reports/ | 数据审计、逐回合 JSONL 和指标 JSON |
| videos/ | 分实验、阶段保存的回放视频 |

推荐统一用上面的 run_disk 包装器，参数设置在子进程启动前生效；原生库 stderr 同样写入日志。
新终端需重新定义 PY、RUNTIME、OUT 和 run_disk；不修改用户全局 shell 设置。
直接运行 continual.run 也会在导入 ML 库前配置存储，但不会自动捕获完整 stdout/stderr；因此正式运行使用包装器。
未传 --output 时按 checkpoints/{method}_seed{seed}_{frozen_vision|vision_ft} 自动生成。
显式 --output 必须位于运行根目录的 checkpoints/ 下，旧示例中的仓库内/系统盘输出不再接受。

启动前检查：挂载文件系统与系统盘不同、解析符号链接后不越出管理目录、写权限、可用空间。
训练/启动包装器默认至少需要 100 GiB；--minimum-free-gib 可显式调整，数据审计入口最低 1 GiB。
每次保存检查点前额外检查至少剩余 20 GiB，空间不足即中止，不删除旧检查点腾空间。
这些是当时的预检，并不预留磁盘空间，其他用户仍可能消耗空间。
完整检查点约数 GiB。每组四阶段保留四份阶段末检查点；保存期间新旧快照会短暂共存，运行前按实验组数和种子数核算容量。
只复制已缓存的公开 tokenizer 小文件到新 OpenPI 缓存，避免无谓重新下载；不复制凭据，不迁移或清理旧缓存。
中断后临时文件保留供诊断，运行记录包含实际目录、命令和退出码。测试验收通过后清理专用临时目录与测试日志，保留测试源代码和简短验证记录；不要清空仍在运行的任务目录。
外部 shell 的重定向由 shell 决定；不要额外使用 > ~/... 或仓库内日志路径。

CPU 回归测试也应通过包装器：

```bash
run_disk env JAX_PLATFORMS=cpu "$PY" -B -m pytest -p no:cacheprovider scripts/train_test.py -q
```

直接运行 scripts/train.py 的 cl_* 配置也有保护：未通过存储入口设置环境，或检查点仍位于仓库内时会报错。通用非 CL 配置保持原有默认行为；使用包装器后同样遵循数据盘缓存设置。
