# 协议 v2 验证记录（2026-10-28）

当前方案：pi05_base，四任务，独立任务归一化；不回放 vs 每个旧任务 20% 样本起点回放。
默认视觉全量微调、语言/动作 LoRA、batch=8、每阶段 10000 步。
每批新旧各半，旧任务等概率；默认 π0.5 分位数缩放，另提供 zscore。

已验证：
- 四份统计量生成成功：训练起点 5947 / 8520 / 7931 / 6847；未使用验证轨迹。
- 20% 缓冲区：1189 / 1704 / 1586 / 1369；T4 阶段旧任务共 4479 起点。
- seed=42 时 T1/T2/T3 独立动作标签覆盖率约 88.33% / 87.97% / 87.61%。
  因每个起点带十步动作块，不能把本设置称为仅访问 20% 的独立动作标签。
- 真实 LeRobot 数据：四任务共 12 个起点（含中间和轨迹末尾），验证轨迹顺序、
  动作块补齐、任务归一化、反归一化与回放 Subset 路由，全部通过。
- CPU 训练/保存/恢复及当时协议测试：19 passed，167.71 秒。
- 后续数据加载与协议测试：25 passed、1 deselected，23.30 秒。
  跳过的是需额外下载 ALOHA 数据的测试；实际 LIBERO 数据另行验证。
- 最终协议测试：21 passed，6.05 秒。
- Ruff check/format 与 git diff --check 通过。
- 训练计划入口验证成功；没有启动正式 π0.5 GPU 训练。

数据盘产物位于 openpi_cl_runtime/reports/：
- per_task_norm_v1/T1..T4/{norm_stats.json,metadata.json}
- per_task_norm_audit_v2.json
- replay20_buffer_audit_v2.json

完整 LIBERO rollout 尚未验证：训练 .venv 不包含模拟器，需使用独立模拟器环境。
CPU debug 回归不代表真实 π0.5 的显存、收敛或机器人成功率已通过。
本轮开始时 scripts/train.py 已有更全面的有限值检查修改；保留并参与 CPU 回归，
没有将这份原有未提交修改纳入本轮提交。

---
以下为旧版协议的历史验证记录，不代表 v2 当前默认设置。

# Validation record — 2026-10-28

- Protocol unit tests: 4 passed (fixed replay buffers, no-replay isolation, budget validation, metric/incomplete-run rejection).
- Existing CPU training + checkpoint resume regression: 1 passed in 143.17 s. Dependency deprecation warnings remain.
- Real T2 replay dataset: 8,520 current starts + 100 T1 starts = 8,620; first/last samples of both partitions read correctly, action shape (10, 7).
- Existing stage checkpoint metadata: 20 LoRA tensor keys found using the same Orbax metadata API used by the stage guard.
- Real data audit: results/data_audit.json, all training action arrays finite.
- Stage-one plan command validated against actual dataset, normalization and split paths.
- Ruff checks and formatting passed; compileall and git diff --check passed.

The first CPU regression attempt failed because the system filesystem ran out of space. Rerun used a temporary directory on the data disk and passed. Existing user files were not deleted.

Not validated: full-size GPU training under the new schedule, numerical parameter-delta checks, policy action equality after restore, simulator rollouts, or empirical convergence/forgetting. The training virtualenv does not contain LIBERO; use a separate simulator environment as described in the protocol. Do not interpret these engineering checks as evidence of policy success.

## Data-disk storage update — 2026-10-28

- Storage + protocol tests: 9 passed, including system-filesystem rejection, resolved symlink escape rejection, checkpoint output containment, low-space failure, inherited subprocess TMPDIR and direct CL trainer misuse guards.
- Actual storage launcher probe: tempfile, JAX, OpenPI and HF paths all resolved beneath /mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime; stdout/stderr captured together on that disk.
- Stage-one plan: default checkpoint output resolved to checkpoints/sequential_seed42_frozen_vision beneath runtime root; recorded startup free capacity about 6020 GiB.
- Existing CPU train/save/resume regression through the new launcher: passed in 148.30 s. Test temporary files and JAX cache used the data disk. Dependency warnings are unchanged.
- Ruff and diff whitespace checks passed. No full-size GPU training launched by this update.

## Post-validation cleanup

Removed only this session's pytest temporary checkpoints, test launch logs and dummy-model JAX cache entries (6.02 GiB). Validation summaries and test source remain; original training checkpoints and logs were preserved. New runs retain only the latest checkpoint per stage (keep_period=None; existing manager max_to_keep=1), leaving four stage-end checkpoints per completed sequence.
