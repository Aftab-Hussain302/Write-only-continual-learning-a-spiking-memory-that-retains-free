# Environment

`scripts/run_all.py` writes `ENVIRONMENT.md` inside every run folder with the machine it ran on (GPU, driver, CUDA, Python, torch,
`pip freeze`) and `MANIFEST.json` with a SHA-256 of every source file, so a result folder identifies the code and the machine that
produced it. The paper's Methods quote the GPU, the PyTorch and CUDA versions and the wall-clock time from those files of the final
run, and `determinism.json` holds the largest difference between two runs of the same unit.
