## 1) 仓库简介：MUSt3R 是什么？解决什么问题？

MUSt3R（CVPR 2025）是 NAVER LABS Europe 提出的一个**从多视图（multi-view）直接预测相机位姿与稠密 3D（pointmaps）**的网络。它从 DUSt3R 的“成对图像（pairwise）预测”出发，主要做了两类关键改造：

1. 把 DUSt3R 的结构改成更适合 N-view 的**对称（symmetric）结构**，让多视图能在**同一个坐标系**下直接输出 3D 结构；
2. 引入**多层 memory 机制**，让新来的帧/图片只需要“看历史 memory”，从而把多视图推理从“二次方级别的两两配对”压力中解放出来，支持**离线 SfM（无序图片集）**与**在线 VO/SLAM（视频流）**两种场景。 [2](https://ar5iv.org/abs/2503.01661)

从应用角度，仓库提供了两条最直接的“可复现路径”：

- **离线重建 Demo（Gradio + viser 可视化）**：适合你丢一堆图片进去做重建；如果图片无序，需要配合 retrieval 模型走“unordered: retrieval”模式。 [1](https://github.com/naver/must3r)
- **在线 Visual Odometry/SLAM Demo（open3d）**：适合从 webcam / 视频 / 帧目录实时跑轨迹与稠密点云式的重建。 [1](https://github.com/naver/must3r)

------

## 2) 仓库结构（你克隆下来会看到什么）

仓库根目录能看到这些关键内容：

- `dust3r/`：以 **git submodule** 方式引入 DUSt3R（这也是为什么 README 强烈建议 `--recursive`）。 [1](https://github.com/naver/must3r)
- `must3r/`：MUSt3R 主包代码。 [1](https://github.com/naver/must3r)
- `demo.py`：离线 Gradio Demo 入口（安装后也会提供 `must3r_demo` 可执行入口）。 [1](https://github.com/naver/must3r)
- `slam.py`：在线 VO/SLAM Demo 入口（安装后也会提供 `must3r_slam` 可执行入口）。 [1](https://github.com/naver/must3r)
- `train.py / eval.py`：训练与评估脚本（README 明确提到训练脚本不内置 validation，需要手动跑 `eval.py`）。 [1](https://github.com/naver/must3r)
- `assets/`：包含示例图、评测链接等资源。 [1](https://github.com/naver/must3r)

------

## 3) 许可证/合规提醒（非常重要）

MUSt3R 代码是 **Non-Commercial（非商用）**许可；另外 README 也特别强调其训练数据集许可（例如 mapfree）可能非常严格，你在下载/使用官方 checkpoint 前应自行确认这些数据集许可条款是否与你的使用场景兼容。 [1](https://github.com/naver/must3r)

------

# 4) 复现步骤（Linux + NVIDIA GPU）

下面以 README 给出的“官方推荐环境”为准：Python 3.11、PyTorch 2.7.0、xFormers 0.0.30，并示例 CUDA 12.6 的 wheel 源。你可以照抄，只需要把 CUDA wheel 源换成你机器匹配的版本。 [1](https://github.com/naver/must3r)

------

## Step 0：准备条件（硬件/系统）

- 推荐：Linux + NVIDIA GPU（CUDA 可用）
- 你当然也可以尝试 CPU，但推理速度和体验会差很多；而且 README 中很多参数默认就是 `--device cuda`。 [1](https://github.com/naver/must3r)

## A. AutoDL 上租用服务器

### A1) 创建实例（租用新实例）

在 AutoDL 控制台进入“我的实例”，点击“租用新实例”，然后依次选择 **计费方式、地区、GPU 型号、GPU 数量、空闲主机、镜像** 创建即可。官方快速开始文档就是这个流程。

> 计费提醒：实例状态变成“运行中”就开始计费，不用时记得关机。[1](https://www.autodl.com/docs/quick_start/)
> 数据保留提醒：关机数据保留，但**连续关机 15 天实例会被释放并清空**。[1](https://www.autodl.com/docs/quick_start/)

### A2) GPU / 规格怎么选（建议）

MUSt3R 离线重建/可视化比较吃显存；为了减少 OOM（爆显存）和调参时间，建议：

- **推荐起步：单卡 24GB 显存档**（4090/3090/A5000/A40 等同级别）
- **预算紧：16GB 也能跑**，但更建议用 `224` 模型或减少图片数量，并在 Demo 里把 “Maximum batch size” 设为 1（MUSt3R README 明确强调这一点）。[](https://github.com/naver/must3r)

AutoDL 的 CPU/内存一般按“每卡倍增”分配（例如标注 8核/GPU、32GB/GPU，那么 1 卡就是 8 核 32GB）。[3](https://www.autodl.com/docs/env/)

### A3) 镜像怎么选（尽量省事）

MUSt3R README 推荐的组合是 **Python 3.11 + PyTorch 2.7.0 + xFormers 0.0.30**。[2](https://github.com/naver/must3r)

AutoDL 平台提供很多内置 PyTorch 镜像版本，其中包含 **PyTorch 2.7.0（Python 3.12 / CUDA 12.8）**。你可以直接选这个镜像，省掉大量装框架的时间。[4](https://www.autodl.com/docs/base_config/)

------

## B. 连接实例：SSH + VSCode Remote-SSH（推荐工作流）

### B1) 先确认你能 SSH 登陆

准备工作可选择无卡模式开机（省钱），实例开机后，在控制台里复制 SSH 登录指令，形如：

```
ssh -p 10309 root@connect.nmb1.seetacloud.com
```

然后在本地终端执行，输入密码即可。[5](https://www.autodl.com/docs/ssh/)

> 长时间跑任务（下载权重/跑重建）务必用 `screen/tmux` 或 JupyterLab 终端，避免 SSH 断开导致进程停掉。[5](https://www.autodl.com/docs/ssh/)

### B2) VSCode Remote-SSH 配置（一步步照做）

AutoDL 官方给了 Remote-SSH 配置流程，核心步骤是：安装 VSCode 扩展 Remote-SSH → 添加 SSH 主机 → 选择 Linux → 输入密码登录 → 打开远程目录。[6](https://api.autodl.com/docs/vscode/)

**具体操作要点（按 AutoDL 文档）：**

1. 本地 VSCode 安装扩展 **Remote-SSH**。[6](https://api.autodl.com/docs/vscode/)
2. 选择 “Add New SSH Host”，粘贴 AutoDL 提供的 ssh 命令（注意末尾不要多空格）。[6](https://api.autodl.com/docs/vscode/)
3. 远程系统选择 Linux，输入密码，连接成功后打开你的工作目录。[6](https://api.autodl.com/docs/vscode/)

**如果 VSCode 远程连接失败**，AutoDL 也有专门的排查建议：在 Remote-SSH 扩展设置里把 “Config File” 指到报错里提示的 config 路径。[7](https://api.autodl.com/docs/qa5/)

### B3) （可选）配置 SSH 免密登录

AutoDL 支持你在控制台配置公钥，实现免密码登录；公钥生成用 `ssh-keygen -t rsa`。

------

## Step 1：创建环境并安装 PyTorch + xFormers（最关键一步）

### 方式 A：按 README 用 micromamba（最贴近官方）

```
conda create -n must3r python=3.11 -y
conda init bash
source /root/.bashrc
conda activate must3r

pip install --upgrade pip

# CUDA 12.8 示例（AutoDL PyTorch 2.7.0 镜像常见搭配）
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128

# （推荐）安装 xformers（节省显存/更快 attention）
pip install -U xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
```

------

## Step 2：安装 must3r 代码

MUSt3R 仓库依赖 `dust3r` 子模块，README 推荐 `--recursive`。

```
cd /root
git clone --recursive https://github.com/naver/must3r.git
cd must3r

#安装依赖
pip install -r dust3r/requirements.txt
pip install -r dust3r/requirements_optional.txt
pip install -r requirements.txt
```

------

## Step 3：安装 retrieval 依赖：ASMK + FAISS（做“无序图片检索/配对”会用到）

README 的开发安装流程里明确包含：安装 `faiss-cpu`，然后编译安装 `asmk`。这块通常是很多人复现卡住的点之一。 [1](https://github.com/naver/must3r)

```
pip install faiss-cpu

mkdir -p build
cd build
git clone https://github.com/jenicek/asmk.git
cd asmk/cython/
pip install cython
cythonize *.pyx
cd ..
pip install .
cd ../..
```

> 说明：
>
> - **如果你的图片是“无序集合”**（例如随手拍了一堆照片，没有视频顺序），README 说你“必须”选 `unordered: retrieval` 模式（并且只有在你提供 `--retrieval` 的情况下才有）。而这条链路就依赖检索（retrieval）组件。 [1](https://github.com/naver/must3r)

⚠报错

```
(must3r) root@autodl-container-1b2041a5c3-bb864b45:~/must3r/build/asmk# pip install .
Looking in indexes: http://mirrors.aliyun.com/pypi/simple
Processing /root/must3r/build/asmk
Installing build dependencies ... done
Getting requirements to build wheel ... done
Preparing metadata (pyproject.toml) ... done
Requirement already satisfied: numpy in /root/miniconda3/envs/must3r/lib/python3.11/site-packages (from asmk==0.1) (2.2.6)
Requirement already satisfied: pyaml in /root/miniconda3/envs/must3r/lib/python3.11/site-packages (from asmk==0.1) (25.7.0)
Requirement already satisfied: PyYAML in /root/miniconda3/envs/must3r/lib/python3.11/site-packages (from pyaml->asmk==0.1) (6.0.3)
Building wheels for collected packages: asmk
Building wheel for asmk (pyproject.toml) ... error
error: subprocess-exited-with-error

× Building wheel for asmk (pyproject.toml) did not run successfully.
│ exit code: 1
╰─> [17 lines of output]
running bdist_wheel
running build
running build_py
copying asmk/init.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/asmk_method.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/codebook.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/functional.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/index.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/inverted_file.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/io_helpers.py -> build/lib.linux-x86_64-cpython-311/asmk
copying asmk/kernel.py -> build/lib.linux-x86_64-cpython-311/asmk
running build_ext
installing to build/bdist.linux-x86_64/wheel
running install

text

  ERROR: faiss package not installed (install either faiss-cpu or faiss-gpu before installing this package.).
  
  [end of output]
note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building wheel for asmk
Failed to build asmk
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> asmk
```

你这个报错的关键点在这里：

> ```
> ERROR: faiss package not installed ... before installing this package
> ```

它**不一定**表示你当前 `(must3r)` 环境里真的没装 `faiss`，而是很常见的一种情况：**pip 用 PEP517 的“隔离构建环境（build isolation）”去编译/打 wheel**，在那个临时构建环境里 `import faiss` 失败，于是 asmk 的安装脚本直接中止。解决思路就是：**确保 faiss 能在当前环境 import，然后用 `--no-build-isolation` 安装 asmk**。 [1](https://stackoverflow.com/questions/78096854/avoiding-package-reinstalls-when-developing-binary-module)

下面给你一套“按顺序照抄就能好”的完整修复步骤。

------

## 1) 先确认：你现在这个环境里 faiss 到底能不能 import

在 AutoDL 里，先确认你用的就是 `(must3r)` 的 python/pip：

```
which python
which pip
python -V
```

然后测试 faiss：

```
python -c "import faiss; print('faiss ok, version=', getattr(faiss,'__version__','unknown'))"
pip list | grep -i faiss || true
```

- 如果这一步 **能 import**，那几乎可以断定：你遇到的是 **build isolation** 问题（下一步直接用 `--no-build-isolation` 安装 asmk）。
- 如果这一步 **不能 import**，那就先装好 faiss（看第 2 步）。

------

## 2) 安装 faiss（建议优先用官方 PyPI 或 conda-forge，别被镜像坑）

你现在 pip 用的是阿里云镜像 `http://mirrors.aliyun.com/pypi/simple`，有时会遇到 wheel 同步不全/解析奇怪的问题。建议临时切到官方源安装 faiss：

### 方案 A（推荐，简单）：pip 安装 faiss-cpu

```
pip install -U "faiss-cpu" -i https://pypi.org/simple
```

说明：PyPI 上的包名就是 `faiss-cpu`。 [2](https://pypi.org/project/rf-faiss-gpu/)

装完再测一次：

```
python -c "import faiss; print('faiss import ok')"
```

### 方案 B（更稳，尤其是你 conda 环境）：conda-forge 安装

```
conda install -y -c conda-forge faiss-cpu
python -c "import faiss; print('faiss import ok')"
```

------

## 3) 关键修复：安装 asmk 时关闭 build isolation

进入 asmk 仓库根目录（你现在就在 `/root/must3r/build/asmk`，对的），然后这样装：

```
cd /root/must3r/build/asmk

# 建议把构建工具也先补齐一下
pip install -U pip setuptools wheel cython

# 关键：关闭隔离构建，否则安装脚本在临时环境里 import faiss 会失败
pip install --no-build-isolation .
```

为什么这能解决：`--no-build-isolation` 会让 pip 在**当前环境**里构建/安装，而不是建一个临时隔离环境；你当前环境里已经装好的 faiss 就能被 asmk 的安装过程检测到。 [1](https://stackoverflow.com/questions/78096854/avoiding-package-reinstalls-when-developing-binary-module)

------

## 4) 验证 asmk 是否安装成功

```
python -c "import asmk; import faiss; print('asmk ok')"
```

如果这句过了，一般就说明 must3r 里“unordered: retrieval”相关依赖链条基本 OK。

⚠又报错

```
(must3r) root@autodl-container-1b2041a5c3-bb864b45:~/must3r/build/asmk# python -c "import asmk; import faiss; print('asmk ok')"
Traceback (most recent call last):
File "<string>", line 1, in <module>
File "/root/must3r/build/asmk/asmk/init.py", line 6, in <module>
from .asmk_method import ASMKMethod
File "/root/must3r/build/asmk/asmk/asmk_method.py", line 7, in <module>
from . import kernel as kern_pkg, codebook as cdb_pkg, index as idx_pkg, inverted_file as ivf_pkg
File "/root/must3r/build/asmk/asmk/kernel.py", line 5, in <module>
from . import functional, hamming, io_helpers
ImportError: cannot import name 'hamming' from partially initialized module 'asmk' (most likely due to a circular import) (/root/must3r/build/asmk/asmk/init.py)
```

你现在这个报错的“本质原因”不是 MUSt3R 的问题，而是 **ASMK 这个包需要一个 Cython 编译出来的扩展模块 `asmk.hamming`**。你在 `~/must3r/build/asmk` 目录里直接 `import asmk` 时，Python 优先从“当前目录”加载源码包 `asmk/`，但这里 **没有编译好的 `hamming\*.so`**，于是 `kernel.py` 里 `from . import ... hamming ...` 找不到扩展模块，最后以“partially initialized module / circular import”这种典型形式爆出来。这个现象也符合 Python 对“循环导入/同名遮蔽”导致的 partially-initialized 报错特征。[1](https://bobbyhadz.com/blog/python-attributeerror-partially-initialized-module-has-no-attribute)

下面给你一套**从清理到重新编译安装、再验证**的完整修复步骤（按顺序照抄即可）。

------

## 1) 先不要在 asmk 源码目录里做 import（立刻规避“源码遮蔽”）

先切出去再测（否则即使你装好了，仍可能被当前目录的源码包遮蔽）：

```
cd /root
python -c "import asmk, faiss; print('asmk file:', asmk.__file__); print('ok')"
```

- 如果此时还能报同样错误，说明 **asmk 还没正确安装/扩展没编译出来**，继续做第 2 步。
- 如果此时成功，说明你之前只是被“当前目录源码包遮蔽”，之后使用时避免在 `~/must3r/build/asmk` 目录下跑即可。[1](https://bobbyhadz.com/blog/python-attributeerror-partially-initialized-module-has-no-attribute)

------

## 2) 完整“清理 + 重新编译安装” ASMK（推荐这样做，最干净）

### 2.1 卸载旧的 asmk（如果装过）

```
conda activate must3r
pip uninstall -y asmk || true
```

### 2.2 确保编译工具齐全（AutoDL 镜像有时缺 gcc）

```
apt-get update
apt-get install -y build-essential
```

### 2.3 确保 faiss 在当前环境可 import（这一步是 ASMK 安装脚本的硬前置）

**建议临时用官方 PyPI 源**（避免阿里镜像缺轮子/解析异常）：

```
pip install -U faiss-cpu -i https://pypi.org/simple
python -c "import faiss; print('faiss ok')"
```

> 你之前遇到的 “faiss package not installed” 很典型就是：要么 faiss 真没装，要么 pip 的 PEP517 隔离构建环境里 import 不到。后面我们会用 `--no-build-isolation` 彻底规避。

### 2.4 安装 Cython 等构建依赖

```
pip install -U pip setuptools wheel cython
```

### 2.5 重新生成/编译 Cython 扩展，并关闭 build isolation 安装

在 asmk 仓库根目录执行（你当前 asmk 路径是 `/root/must3r/build/asmk`）：

```
cd /root/must3r/build/asmk

# 先把 cython/ 里所有 pyx 生成 C 文件（有些项目需要这一步）
cd cython
cythonize *.pyx
cd ..

# 关键：关闭 PEP517 build isolation，让构建时能用到你环境里的 faiss
pip install -v --no-build-isolation .
```

------

## 3) 验证：确认 `hamming` 扩展模块真的装出来了

### 3.1 确认 asmk 来自 site-packages（而不是源码目录）

```
cd /root
python -c "import asmk; print(asmk.__file__)"
```

输出路径应类似：

- `/root/miniconda3/envs/must3r/lib/python3.11/site-packages/asmk/__init__.py`

而不应是：

- `/root/must3r/build/asmk/asmk/__init__.py`

### 3.2 确认 `asmk.hamming` 能找到（这是你当前报错的核心）

```
python -c "import importlib.util; print(importlib.util.find_spec('asmk.hamming'))"
python -c "import asmk.hamming as h; print('hamming ok', h)"
python -c "import asmk, faiss; print('asmk ok')"
```

------

## 4) 如果你“必须在 asmk 源码目录里跑”（不推荐，但可以救急）

如果你就是在 `/root/must3r/build/asmk` 里做实验，至少这样跑，避免当前目录遮蔽：

```
cd /root/must3r/build/asmk
PYTHONPATH="" python -c "import asmk; print(asmk.__file__)"
```

------

## 5) 仍失败时的两个高频分支（直接给你处理法）

### 分支 A：编译时报 numpy/C API 相关错误

有些老项目对 numpy 2.x 不够友好。你现在环境里是 `numpy 2.2.6`。如果编译阶段出现 numpy 相关报错，可以尝试降到 numpy 1.x 后重装：

```
pip install -U "numpy<2"
pip uninstall -y asmk || true
cd /root/must3r/build/asmk
pip install -v --no-build-isolation .
```

### 分支 B：仍然提示 faiss 未安装

再次确认你不是在隔离构建环境里装（必须带 `--no-build-isolation`），并确认：

```
python -c "import faiss; print('faiss ok')"
pip show faiss-cpu
```

------

按上面做完，`asmk.hamming` 这个扩展模块会正常出现，你的 `import asmk; import faiss` 就会通过，然后 MUSt3R 的 **unordered: retrieval** 链路才能稳定工作。

------

## Step 4（可选但推荐）：编译 curope（更快的 RoPE CUDA kernel）

```
cd dust3r/croco/models/curope/
pip install .
cd ../../../../
```

------

⚠报错

```
(must3r) root@autodl-container-1b2041a5c3-bb864b45:~/must3r/dust3r/croco/models/curope# pip install .
Looking in indexes: http://mirrors.aliyun.com/pypi/simple
Processing /root/must3r/dust3r/croco/models/curope
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [20 lines of output]
      Traceback (most recent call last):
        File "/root/miniconda3/envs/must3r/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/root/miniconda3/envs/must3r/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/root/miniconda3/envs/must3r/lib/python3.11/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 143, in get_requires_for_build_wheel
          return hook(config_settings)
                 ^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-4d4ekt5d/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 331, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "/tmp/pip-build-env-4d4ekt5d/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "/tmp/pip-build-env-4d4ekt5d/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 512, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/tmp/pip-build-env-4d4ekt5d/overlay/lib/python3.11/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 5, in <module>
      ModuleNotFoundError: No module named 'torch'
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'file:///root/must3r/dust3r/croco/models/curope' when getting requirements to build wheel
```

# 5) 下载官方 checkpoints（必须）

仓库提供了多种 checkpoint：

- 主模型：`MUSt3R_224_cvpr.pth`、`MUSt3R_512_cvpr.pth`、`MUSt3R_512.pth`
- retrieval 用：`*_retrieval_trainingfree.pth` + `*_retrieval_codebook.pkl`
  并给了 MD5 校验列表。 [1](https://github.com/naver/must3r)

## 5.1 建议的下载目录结构

```
mkdir -p checkpoints
cd checkpoints
```

## 5.2 直接 wget 下载很慢（按 checksums.txt 推断文件在同一目录下）

我已成功打开官方 `checksums.txt`，其中列出的文件名为 `./MUSt3R_512.pth` 等，因此下载 URL 通常就是“目录 + 文件名”。 [3](https://download.europe.naverlabs.com/ComputerVision/MUSt3R/checksums.txt)

```
# 主模型（建议优先用 MUSt3R_512.pth）
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth
# 或 CVPR 提交版本
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_cvpr.pth
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_cvpr.pth

# retrieval（无序图片集强烈建议准备）
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_codebook.pkl

wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_retrieval_trainingfree.pth
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_224_retrieval_codebook.pkl
```

## 本地下载→ 使用文件传输软件（WinSCP）上传服务器

该方案避开海外服务器直连，利用本地宽带下载（可借助国内浏览器下载加速），再通过WinSCP远程传输功能上传，步骤如下：

### 步骤 1：本地电脑下载完整模型文件

1. 打开电脑浏览器，复制以下官方模型链接，逐个粘贴到浏览器地址栏，开始本地下载：
   - 核心模型：https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512.pth
   - 检索模型：https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_trainingfree.pth
   - 检索码本（可选）：https://download.europe.naverlabs.com/ComputerVision/MUSt3R/MUSt3R_512_retrieval_codebook.pkl
2. 等待本地下载完成，确认文件无损坏（查看文件大小，核心模型约数 GB，若仅几十 KB 说明下载失败，重新下载）。

## 5.3 校验 MD5（强烈建议做）

```
wget -c https://download.europe.naverlabs.com/ComputerVision/MUSt3R/checksums.txt

# Linux 下你可以这样校验（示例：校验 MUSt3R_512.pth）
md5sum MUSt3R_512.pth
# 对照 checksums.txt 里对应的 8854f948a8674fb1740258c1872f80dc
```

------

# 6) 运行离线重建 Demo（Gradio + viser，可视化最友好）

README 给的标准启动方式如下（512 模型 + retrieval + viser）：

```
# 在仓库目录（或你能调用到 demo.py 的位置）运行
python demo.py \
  --weights checkpoints/MUSt3R_512.pth \
  --retrieval checkpoints/MUSt3R_512_retrieval_trainingfree.pth \
  --image_size 512 \
  --viser \
  --embed_viser
```

### 常用参数建议（按 README 的“经验提示”）

- 显存紧张/图片多：把 UI 里的 “Maximum batch size” 设为 1（README 强调 IMPORTANT）。 [1](https://github.com/naver/must3r)
- 图片无序：必须选 `unordered: retrieval`（只有提供 `--retrieval` 才能用）。 [1](https://github.com/naver/must3r)
- 想要回环更稳：把 “Number of refinement iterations” 调到 1~2。 [1](https://github.com/naver/must3r)

**viser 3D 可视化工具的参数控制面板**，用于调整 3D 重建结果的显示效果，各参数的功能、当前设置及使用建议如下：

### 一、参数功能与当前状态

| 参数名           | 当前设置 | 功能说明                                                     |
| ---------------- | -------- | ------------------------------------------------------------ |
| Camera Near/far  | 0.01     | 控制 3D 视图的近 / 远裁剪范围，值越小可显示更近的物体，值越大可显示更远的物体 |
| Point size       | 0.012    | 3D 点云的点尺寸，值越大点越醒目，便于观察稀疏点云的分布      |
| Camera size      | 0.35     | 3D 视图中相机模型的显示尺寸，调整后可更清晰区分不同相机的姿态位置 |
| Confidence       | 1        | 3D 点的置信度阈值，仅显示置信度≥该值的点，值越低显示的点越多（含噪声越多） |
| Max Points       | 19200    | 最大显示的 3D 点数量（上限 25000），限制数量以避免界面卡顿   |
| Local pointmaps  | 已勾选   | 启用局部点图渲染，提升 3D 点云的细节显示效果                 |
| Follow Cam       | 未勾选   | 若勾选，3D 视图会跟随相机的姿态自动移动                      |
| Keyframes Only   | 已勾选   | 仅显示关键帧对应的相机和点云，减少非关键数据的干扰           |
| Hide Images      | 未勾选   | 不隐藏原始图像（3D 视图中会显示图像对应的平面）              |
| Hide Predictions | 未勾选   | 不隐藏预测生成的 3D 点云                                     |

### 二、使用建议

1. 若 3D 视图中点云过少：降低`Confidence`阈值（如调至 0.5），或提高`Max Points`至接近 25000；

2. 若 3D 界面卡顿：降低`Max Points`数值，或调小`Point size`；

3. 若想观察所有帧的完整数据：取消`Keyframes Only`的勾选；

4. 若想聚焦相机姿态：增大`Camera size`，更清晰区分不同相机的位置。

   

**MUSt3R 项目的 Gradio 上传与参数配置界面**，用于设置 3D 重建的处理参数，当前已上传单张图像，各参数的功能、当前设置及核心问题 / 建议如下：

### 一、参数功能与当前状态

| 参数名                          | 当前设置                 | 功能说明                                                     |
| ------------------------------- | ------------------------ | ------------------------------------------------------------ |
| Number of refinement iterations | 0                        | 3D 重建的优化迭代次数，次数越高结果精度越高，但处理速度越慢  |
| Maximum batch size              | 1                        | 单次批量处理的图像数量，当前为单张图像处理模式               |
| Mode                            | sequence: slam keyframes | 处理模式（当前为 “SLAM 关键帧序列模式”），适用于连续拍摄的图像序列（如视频帧） |
| Local context size              | 0                        | 特征匹配的局部上下文窗口大小，影响多视图特征的关联范围       |
| subsample                       | 2                        | 图像下采样倍数，降低分辨率以减少计算资源占用                 |
| min conf keyframe               | 1.5                      | 关键帧的最小置信度阈值，用于过滤低置信度的无效关键帧         |
| keyframe overlap thr            | 0.05                     | 关键帧的重叠度阈值，控制相邻关键帧的内容重叠比例             |
| overlap percentile              | 85                       | 重叠度的百分位数筛选条件，仅保留重叠度处于前 85% 的帧        |

### 二、核心问题与使用建议

1. **模式与输入不匹配（当前主要问题）**

   你当前选择的`Mode: sequence: slam keyframes`是 “SLAM 关键帧序列模式”，**需要上传多张连续的图像序列（如视频帧、连续拍摄的照片）**，但当前仅上传了 1 张图像，该模式无法对单张图像进行处理。

   - 若只有单张图像：将`Mode`下拉框切换为`unordered: retrieval`（无序图像检索模式），适配单张 / 零散图像的 3D 重建。
   - 若有图像序列：补充上传多张连续图像（同一物体 / 场景的不同视角），再执行处理。

2. **参数优化建议**

   - 若追求重建精度：将`Number of refinement iterations`调至 10-20（避免过高导致卡顿）；
   - 若减少处理时间：保持`subsample`为 2，或适当增大下采样倍数；
   - 若为序列图像：可将`Maximum batch size`调至 2-4（需 GPU 显存足够），提升批量处理效率。

------

⚠

```
Warning, cannot find cuda-compiled version of RoPE2D
```

RoPE2D CUDA 加速模块未编译，使用纯 PyTorch 版本.不影响功能，仅降低运行速度，可后续优化.

# 7) 运行在线 Visual Odometry/SLAM Demo（open3d）

README 给了 3 类输入：webcam、帧目录、视频文件；以及有/无 GUI 两种模式。 [1](https://github.com/naver/must3r)

## 7.1 webcam 示例（512 模型）

```
python slam.py \
  --chkpt checkpoints/MUSt3R_512.pth \
  --res 512 \
  --subsamp 4 \
  --gui \
  --input cam:0
```

## 7.2 帧目录示例（224 模型）

```
python slam.py \
  --chkpt checkpoints/MUSt3R_224_cvpr.pth \
  --res 224 \
  --subsamp 2 \
  --keyframe_overlap_thr 0.05 \
  --min_conf_keyframe 1.5 \
  --overlap_percentile 85 \
  --input "/path_to/TUM_RGBD/rgbd_dataset_freiburg1_xyz/rgb" \
  --gui
```

## 7.3 无 GUI（输出轨迹与 memory 状态）

README 明确说：无 GUI 会写出 `memory.pkl` 和 `all_poses.npz`，并且可选 `--rerender`。

```
python slam.py \
  --chkpt checkpoints/MUSt3R_512.pth \
  --res 512 \
  --subsamp 4 \
  --input /path/to/video.mp4 \
  --output /path/to/export
```

------

# 8) （可选）训练复现：怎么把训练跑起来？

MUSt3R 仓库 README 给了一个“训练超参示例”，核心点包括：

- 用 `torchrun`（示例是 8 卡）
- `CausalMUSt3R` 是为了训练加速的版本；推理时用 `MUSt3R` 类
- `train.py` 没有内置 validation，需要自己跑 `eval.py`。 [1](https://github.com/naver/must3r)

README 示例（原样保留结构，你需要把数据集部分替换成你自己的实现/路径）：

```
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
torchrun --nnodes 1 --nproc_per_node 8 --master_port $((RANDOM%500+29000)) train.py \
  --decoder CausalMUSt3R(img_size=(512, 512), feedback_type='single_mlp', memory_mode="kv", mem_dropout=0.1, dropout_mode='temporary', use_xformers_mask=True, use_mem_mask=True) \
  --epochs 100 --warmup_epochs 10 \
  --memory_num_views 20 --memory_batch_views 5 --min_memory_num_views 2 \
  --causal --loss_in_log --amp bf16 --use_memory_efficient_attention \
  --batch_size 1 --accum_iter 4 \
  --dataset "XXX @YOUR_DATASET(ROOT='/pathto/your/dataset', resolution=[(512, 384),(512, 336),(512, 288),(512, 256),(512, 160)], aug_crop=64, num_views=20, min_memory_num_views=2, max_memory_num_views=20, transform=ColorJitter) + ..." \
  --chkpt "/path/to/base_checkpoint" \
  --output_dir "/path/to/output_dir"
```

> 训练复现的现实提醒：
> 论文里提到他们训练时会混合大量数据集（十多个数据源），这部分一般不是“普通个人电脑可完整复现”的路线；但你可以用它的训练脚本框架在自己的数据上 fine-tune 或做小规模验证。 [2](https://ar5iv.org/abs/2503.01661)

------

## 9) 常见复现坑（按 README 暗示的高频问题整理）

1. **忘记 `--recursive` / submodule 没拉下来**
   会导致 `dust3r/...` 依赖缺失。解决：`git submodule update --init --recursive`。 [1](https://github.com/naver/must3r)
2. **无序图片集却没启用 retrieval**
   README 写得很死：无序必须选 `unordered: retrieval`，而这要求启动 demo 时传 `--retrieval`。 [1](https://github.com/naver/must3r)
3. **显存爆炸**
   README 明确提示：图片多或 GPU 小时，把 “Maximum batch size” 调 1 来限 VRAM。 [1](https://github.com/naver/must3r)
4. **CUDA / PyTorch / xformers 版本不匹配**
   README 是按 Torch 2.7.0 + xformers 0.0.30（示例 cu126）给的；如果你 CUDA 不同，请同步换 wheel 源和版本组合。