# Glassbox

Glassbox is a CPU-tape-based autograd engine with a CUDA kernel library validated against CPU reference implementations.

## Build and test

```bash
cargo test
python3 -m venv .venv
source .venv/bin/activate
maturin develop
python python/tests/test_smoke.py
```

The GPT and optimizer training path runs entirely on CPU tensors backed by shared host memory and currently supports one attention head. The separate forward-only CUDA backend copies host data to the device for each operation and has no CUDA backward kernels. GPU-accelerated training would require persistent device storage plus CUDA backward implementations; it is a scoped-out next step and is not wired into the training path.

At the Rust tape level, detach creates a new leaf that severs the source graph connection while sharing the same CPU storage. Eager Python tensors do not carry a tape connection, so Python detach currently returns a separate shared-storage handle but has no gradient graph to sever.

## Train

The character dataset is vendored from the TinyShakespeare corpus at https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt.

```bash
source .venv/bin/activate
maturin develop
python examples/train.py --steps 100
python examples/train.py --steps 300 --output artifacts/long_run_curve.csv
python examples/train.py --steps 200 --overfit
python examples/train.py --steps 100 --optimizer sgd --learning-rate 1e-2
```

The script reports step-0 loss against `ln(vocab_size)`, validation loss, and every parameter gradient norm. The committed short and 300-step stability curves are documented in `artifacts/README.md`; generated model checkpoints remain ignored.

Choose plain SGD with `--optimizer sgd`. For Adam, `--weight-decay` applies decoupled AdamW weight decay.

After saving the checkpoint, the training example loads it into a fresh model and prints an autoregressive sample. Use `--prompt`, `--generate-tokens`, `--temperature`, `--top-k`, and `--sample-seed` to control generation.

## Continuous integration

The workflow runs formatting, linting, all CPU/autograd tests, and the Python smoke test on pushes and pull requests to main. It uses the cpu-only feature because free hosted runners do not provide a CUDA GPU. CUDA kernel tests remain part of the normal local cargo test run on an NVIDIA machine.

## Visualizer design

Tracing is opt-in and interval-based. Untraced tape operations return before an event or JSON string is built, preserving the normal training path. Traced events enter a bounded channel with `try_send`; a full channel drops the event and increments an exposed counter instead of blocking training.

The browser receives the one-way event stream over Server-Sent Events. SSE matches the server-to-browser data flow, reconnects natively through `EventSource`, and avoids the protocol and dependency cost of WebSockets. A background `std::net::TcpListener` serves both the static page and `/events` from the training process. It listens on `127.0.0.1:8080` by default and accepts a configurable port.

Enable a live trace from the training example with:

```bash
python examples/train.py --steps 100 --visualizer --trace-interval 10
```

The Python `GPT.enable_visualizer(port=8080, trace_interval=50, capacity=8192)` method returns the page URL. `GPT.dropped_events()` reports queue overflow, and `GPT.disable_visualizer()` removes the emitter.

The renderer represents each tensor once and draws one operation-labeled edge for every distinct input-to-output pair. It assigns horizontal layers by longest-path topological depth and spreads nodes vertically within each layer, avoiding a force-directed layout. Batched traces are reduced in the browser to the first sequence through its cross-entropy node; shared parameter leaves remain visible, and the filter can be disabled.

Backward edges point from outputs back to inputs in a separate orange hue. Their width and intensity use a logarithmic scale whose legend shows the actual minimum and maximum gradient norms. Fixture replay advances by event index rather than recorded wall-clock time. The page works from the built-in server, or can be double-clicked and given either JSON fixture through the file picker.

Live mode keeps the five newest complete steps and exposes a selector for older buffered steps. Selecting a different step rebuilds the graph from that step alone because tape tensor IDs are reused. After an SSE reconnect, the first observed step is discarded conservatively so a stream that resumed mid-step cannot be rendered as complete.
