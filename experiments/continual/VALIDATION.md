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
