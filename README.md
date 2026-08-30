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
