# Glassbox

Glassbox is a small Rust and CUDA deep-learning engine built to make tensor storage, kernels, autograd, and transformer execution inspectable.

## Build and test

```bash
cargo test
python3 -m venv .venv
source .venv/bin/activate
maturin develop
python python/tests/test_smoke.py
```

The current GPT path executes on the CPU autograd tape and supports one attention head. CUDA operations currently copy host data to and from the device for each call. Persistent device buffers and end-to-end GPU model execution are planned for Week 5.

## Train

The character dataset is vendored from the TinyShakespeare corpus at https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt.

```bash
source .venv/bin/activate
maturin develop
python examples/train.py --steps 100
python examples/train.py --steps 200 --overfit
```

The script reports step-0 loss against `ln(vocab_size)`, validation loss, and every parameter gradient norm. It writes the reproducible curve to `artifacts/loss_curve.csv` and model parameters to `artifacts/tinyshakespeare.gbx`.
