GPU setup and TensorFlow compatibility
====================================

Goal
----
Provide safe steps to enable GPU training for this TF1-style codebase when the system has a CUDA runtime that does not match the old TF1 GPU wheels in the repo. Two practical options:

- Install a TF2.x GPU wheel that matches your CUDA and run the repository code under TF1 compatibility using `tf.compat.v1` (recommended).
- Or create a conda environment with a TF1 GPU build that exactly matches your CUDA (only if you have that exact CUDA version).

What I changed
--------------
- `dcrnn_train.py` and `run_demo.py` now detect TF version and, if TF2.x is installed, enable `tf.compat.v1` + `tf.disable_v2_behavior()` so the existing TF1-style code runs unchanged.

Quick detection commands (run on your machine)
---------------------------------------------
- Check NVIDIA driver and CUDA runtime (will show CUDA version used by driver):
  ```bash
  nvidia-smi
  ```
- Check installed nvcc (if CUDA toolkit is installed):
  ```bash
  nvcc --version
  ```
- Check which tensorflow is installed and its CUDA compatibility (run inside your conda env):
  ```bash
  python -c "import tensorflow as tf; print(tf.__version__)"
  python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda(), tf.test.is_gpu_available())"
  ```

Recommended (modern, robust) path — use TF2 GPU wheel + compat.v1
-----------------------------------------------------------------
1. Detect your CUDA version as described above.
2. Create a conda env and install a TF2 GPU wheel whose CUDA/CuDNN requirement matches your system. Common choices:

   - CUDA 11.8 → `pip install tensorflow==2.11.0` (or a TF2 wheel built for 11.8)
   - CUDA 11.2 → `pip install tensorflow==2.10.0`
   - CUDA 10.1/10.0 → older TF2 or TF1 builds (less recommended)

   Example using pip in a conda env:
   ```bash
   conda create -n dcrnn_gpu python=3.8 -y
   conda activate dcrnn_gpu
   pip install --upgrade pip
   pip install numpy scipy pandas pyyaml
   pip install tensorflow==2.10.0  # choose the version that matches your CUDA
   pip install -r requirements.txt
   ```

3. Run training or demo as before (the code will switch to `tf.compat.v1` automatically):
   ```bash
   python dcrnn_train.py --config_filename data/student_nodes/config/dcrnn_students_best_nestedcv.yaml --use_cpu_only False
   ```

If that fails because the wheel still doesn't match your CUDA, either:

- Try a different TF2 version that matches your CUDA (consult TensorFlow's release matrix), or
- Use the TF1 GPU wheel only if your system's CUDA exactly matches (see the `env.gpu.yml` in the repo — it expects CUDA 8 / cuDNN6 and `tensorflow-gpu==1.4`, which is very old).

Alternative: create a TF1 GPU conda env (only if your system CUDA == required CUDA)
--------------------------------------------------------------------------------
The repository provides `env.gpu.yml` that targets very old toolkits (CUDA 8 / cuDNN6). If your system actually uses that old CUDA, you can create the env:

```bash
conda env create -f env.gpu.yml
conda activate dcrnn
```

But this rarely matches modern systems — prefer the TF2 path.

Next steps I can do for you
--------------------------
- (A) Generate a conda env YAML tuned to your detected CUDA version (I can create example files for CUDA 11.8 / 11.2 / 10.1). I need the output of `nvidia-smi` or `nvcc --version`.
- (B) Try a short dry-run locally in this workspace (import tensorflow; print version) — only possible if you want me to run commands here.
- (C) Create a small Dockerfile with entrypoint and matching CUDA base image for a chosen TF version (useful if you can run Docker).

Tell me which next step you want (A/B/C) or paste your `nvidia-smi` output and I'll pick the matching TF wheel and provide exact install commands.
