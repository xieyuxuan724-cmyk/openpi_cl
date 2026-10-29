# 从共同 T1 基线继续四任务对照实验

T1 已完成 10,000 步训练与 40 次评估，成功率为 39/40。两组共用同一个 T1 模型与测量结果，从 T2 起分成两条独立训练链：

| GPU | 方法 | 后续流程 |
|---|---|---|
| 0 | sequential，无回放 | T2 → 评估 T1/T2 → T3 → 评估 T1/T2/T3 → T4 → 评估全部任务 |
| 2 | replay，每个旧任务保留 20% 训练起始帧 | 同上，每批一半新任务、一半旧任务，旧任务均匀采样 |

这属于两组实验并行，每组在一张卡上运行。组内任务必须按顺序训练，不能将 T2、T3 同时独立训练再视为持续学习。

## 启动与日志

在服务器仓库根目录执行：

```bash
cd /home/xieyuxuan/denghaoxin_project/openpi_cl
bash experiments/continual/continue_two_gpus.sh check  # 只校验，不启动训练
bash experiments/continual/continue_two_gpus.sh        # 内部 nohup，同时启动两组
```

检查命令可省略，启动时也会执行。脚本默认 GPU 0、2，端口 8000、8002；可用 SEQUENTIAL_GPU、REPLAY_GPU、SEQUENTIAL_PORT、REPLAY_PORT 环境变量更改。

```bash
RUNTIME=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime
tail -n 50 -f "$RUNTIME/logs/continual_sequential_seed42.latest.log"
tail -n 50 -f "$RUNTIME/logs/continual_replay20_seed42.latest.log"
# 两组是否完成、汇总文件位置：
tail -n 50 -f "$RUNTIME/logs/continual_pair_seed42.latest.log"
```

每次启动使用独立日志，保存 PID 和结束退出码。Ctrl+C 退出 tail 不会停止实验；SSH 断开后后台继续。不要反复启动，锁会阻止同一实验重复运行。

## 保持一致的实验设置

- 训练种子 42；每个后续任务 10,000 步，warmup 500，batch 8。
- 使用 π0.5 + LoRA，视觉参与训练；继承上一任务全部模型参数，包括已有 LoRA。新阶段重置优化器。
- 每个任务只使用自己的训练数据计算固定 quantile 归一化；两组使用相同统计量。
- replay 保留每个旧任务 floor(20% × 训练起始帧数)，并跨后续阶段保持固定。T2 为 T1 的 1,189 个；T3 加入 T2 的 1,704 个；T4 加入 T3 的 1,586 个。
- 每次评估所有已见任务，每任务 40 次，使用初始状态 10–49、环境 seed 7；每个任务保存前 3 次视频。
- 两组各需继续训练 3 个阶段、执行 360 次评估；共用已经完成的 40 次 T1 基线。

## 共同基线与结果溯源

replay T2 显式使用 `--shared-t1`，仅允许从设置完全一致的 sequential T1 初始化。T3/T4 仍严格要求相同方法和协议衔接。不会复制或重写 T1 权重、manifest、原始评估记录。

汇总 replay 指标时，读取共同 T1 原始结果，验证 replay T2 的实际来源，再在内存中统一分组标签。汇总文件保留共同基线来源、原始训练协议以及所有输入文件的哈希；不会将 T1 伪装成重新训练或重新评估的结果。

## 完成与恢复

完整保存的阶段及其完整评估会跳过。有已提交的中间检查点时，重新执行启动命令会恢复该阶段；数据迭代顺序会重新初始化，因此不承诺与不中断运行逐步完全相同。若失败阶段没有任何有效检查点，脚本会停下并提示先检查，不自动删数据或覆盖。

未完成的评估记录保留；重试使用新目录，只有完整且元数据匹配的结果才进入汇总。一组失败不会终止另一组；两组都成功后才生成对比结果。

输出位于数据盘运行根目录：

- 检查点：`checkpoints/v2_sequential_seed42/`、`checkpoints/v2_replay20_seed42/`。
- 各阶段结果和视频：`reports/`、`videos/` 下对应分组和阶段目录。
- 指标：`reports/v2_sequential_seed42_metrics.json`、`reports/v2_replay20_seed42_metrics.json`。
- 两组比较：`reports/paired_seed42_4tasks_<本次运行标识>.json`。

指标包括各阶段成功率矩阵、最终平均成功率、最终遗忘和 NBT。此次是单训练种子的四任务 pilot，不能据此报告跨训练种子的显著性。
