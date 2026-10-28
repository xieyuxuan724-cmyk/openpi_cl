# 独立 LIBERO 仿真环境

训练环境继续使用仓库 .venv（Python 3.11）；仿真使用独立 Python 3.10。
当前测评入口含 Python 3.10 类型语法，不能直接使用旧示例的 Python 3.8。
环境与安装缓存、临时文件、LIBERO 配置均位于数据盘，不安装到训练环境或系统 Python。

- 仿真解释器：`/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime/envs/libero-py310/bin/python`
- 兼容入口：`examples/libero/.venv` 是指向上述环境的符号链接。
- 核心版本：robosuite 1.4.1、MuJoCo 3.2.3、NumPy 1.22.4、Numba 0.56.4。
- 仅为 Python 3.10 兼容性将旧锁文件的 Numba 0.53.1 升到 0.56.4；MuJoCo/robosuite 保持仓库版本。
- PyTorch 1.11.0+cpu 用于读取初始状态；模型推理仍在训练环境的独立服务中执行。
- `sim_requirements.txt` 固定直接依赖；安装后完整依赖记录在数据盘 reports/libero_sim_freeze.txt。
- 使用已初始化的 third_party/libero 源码与场景资产，不下载另一份训练数据。

## 建立或恢复

从仓库根目录运行：

```bash
bash experiments/continual/setup_sim.sh
# 国内网络可选：
SIM_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash experiments/continual/setup_sim.sh
```

脚本只操作专用仿真环境。检查数据盘空间，安装依赖、检查依赖冲突，生成 LIBERO 配置和兼容链接。
当前服务器继承的代理曾返回错误，脚本仅在自身进程内取消代理，直接使用官方 PyPI / PyTorch 下载源。
不要对训练环境运行仿真依赖的 pip sync/install。

## 无模型的四任务检查

```bash
RUNTIME=/mnt/disk20t/xieyuxuan/denghaoxindatadisk/openpi_cl_runtime
"$RUNTIME/envs/libero-py310/bin/python" -B -m experiments.continual.sim_smoke
```

默认 OSMesa CPU 渲染，不占 GPU。依次加载四任务、读取第 10 个初始状态、执行 12 步、
检查主相机和腕部相机的形状与非空图像，并导入真实 WebSocket 测评客户端。
结果和四张场景截图写入 reports/libero_sim_smoke/；再次运行需要 --output-name 指定新目录。
这是环境连通性检查，不是机器人任务成功率测评。

## 正式测评

训练服务与仿真客户端使用各自解释器，通过 WebSocket 通信。
使用 README.md 中的 experiments.continual.evaluate 入口，默认仿真也采用 OSMesa。
等 GPU 资源允许时可使用 EGL 加速渲染：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -B -m experiments.continual.storage -- \
.venv/bin/python -B -m experiments.continual.evaluate \
  --checkpoint /数据盘/某组/cl_pi05_continual_v2/T1/9999 \
  --norm-assets "$RUNTIME/reports/per_task_norm_v1"
```

EGL 路径需要单独验收；OSMesa 验证成功不等于 EGL 已验证。
旧 100 步诊断检查点属于旧协议，不能绕过 v2 的 manifest/归一化检查直接作为正式测评结果。

## 本机验收（2026-10-28）

- pip check：No broken requirements found。
- 四个任务全部通过：场景加载、reset、初始状态读取、12 步动作执行。
- 各任务有 50 个初始状态；主相机和腕部相机均为 256×256×3 非空图像。
- 验收采用 OSMesa CPU 渲染，不加载 π0.5，不分配训练 GPU。
- 报告：openpi_cl_runtime/reports/libero_sim_smoke/report.json；同目录有 T1..T4.png。
- LIBERO 源码固定在 f78abd68ee283de9f9be3c8f7e2a9ad60246e95c。
- 共享存储工具已兼容 Python 3.10；真实测评模块与 WebSocket 客户端导入成功。
- 尚未完成真实模型服务联调、EGL 验证或成功率测评。

参考：仓库 examples/libero/requirements.txt、third_party/libero/requirements.txt；
[LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)；
[Numba 0.56.4 的 Python 3.10 支持](https://pypi.org/project/numba/0.56.4/)。
