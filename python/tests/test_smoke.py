import math
import os
import random
import tempfile

import glassbox as gb


left = gb.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
right = gb.Tensor([0.5, 1.5], [2, 1])
assert (left @ right).shape == [2, 1]
assert (left + left).data == [2.0, 4.0, 6.0, 8.0]
assert "shape=[2, 2]" in repr(left)
near = gb.Tensor([1.000001, 1.999999, 3.000001, 3.999999], [2, 2])
assert gb.isclose(near, left)
assert near.isclose(left)
assert not gb.isclose(gb.Tensor([1.1, 2.0, 3.0, 4.0], [2, 2]), left)
detached = left.detach()
assert detached.data == left.data
assert detached is not left

try:
    gb.Tensor([1.0], [2])
except ValueError as error:
    assert "shape requires 2 elements" in str(error)
else:
    raise AssertionError("invalid tensor shape should fail")

linear = gb.nn.Linear(2, 3)
norm = gb.nn.LayerNorm(2)
embedding = gb.nn.Embedding(65, 8)
relu = gb.nn.ReLU()
sigmoid = gb.nn.Sigmoid()
tanh = gb.nn.Tanh()
assert isinstance(linear, gb.nn.Module)
assert isinstance(relu, gb.nn.Module)
assert linear.forward(left).shape == [2, 3]
assert norm.forward(left).shape == [2, 2]
assert embedding.forward([1, 4, 1]).shape == [3, 8]
assert relu.forward(gb.Tensor([-1.0, 0.0, 1.0], [3])).data == [0.0, 0.0, 1.0]
assert gb.isclose(sigmoid.forward(gb.Tensor([0.0], [1])), gb.Tensor([0.5], [1]))
assert gb.isclose(tanh.forward(gb.Tensor([0.0], [1])), gb.Tensor([0.0], [1]))
assert left.relu().shape == left.shape
assert left.sigmoid().shape == left.shape
assert left.tanh().shape == left.shape

random.seed(42)
tokens = [[random.randrange(65) for _ in range(16)] for _ in range(2)]
model = gb.nn.GPT(65, 16, 2, 1, 8)
logits = model.forward(tokens)
assert logits.shape == [2, 16, 65]
assert len(logits.data) == 2 * 16 * 65

dataset = gb.data.CharDataset("data/tinyshakespeare.txt", 4, seed=42)
assert dataset.vocab_size == 65
inputs, targets = dataset.batch("train", 2)
trainer = gb.nn.GPT(65, 4, 1, 1, 8, seed=42, learning_rate=1e-2)
url = trainer.enable_visualizer(port=0, trace_interval=1)
assert url.startswith("http://127.0.0.1:")
initial = trainer.evaluate(inputs, targets)
loss, norms = trainer.train_step(inputs, targets)
assert math.isfinite(loss)
assert abs(loss - initial) < 1e-6
assert len(norms) == len(trainer.parameters())
assert trainer.dropped_events() == 0
trainer.disable_visualizer()
assert all(math.isfinite(norm) for norm in norms)

checkpoint = tempfile.NamedTemporaryFile(suffix=".gbx", delete=False)
checkpoint.close()
trainer.save_checkpoint(checkpoint.name)
expected = trainer.forward(inputs)
restored = gb.nn.GPT(65, 4, 1, 1, 8, seed=7)
restored.load_checkpoint(checkpoint.name)
actual = restored.forward(inputs)
assert actual.data == expected.data
os.remove(checkpoint.name)

try:
    gb.nn.GPT(65, 16, 1, 2, 8)
except ValueError as error:
    assert "supports n_head=1" in str(error)
else:
    raise AssertionError("multi-head construction should fail")

print("glassbox Python smoke test passed")
