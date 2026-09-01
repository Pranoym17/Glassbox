import math
import os
import random
import tempfile

import glassbox as gb
import glassbox.data as data
import glassbox.nn as nn
import glassbox.optim as optim
from glassbox.nn import Linear
from glassbox.optim import SGD

assert SGD is optim.SGD
assert data.CharDataset is gb.data.CharDataset
assert Linear is nn.Linear


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

head = gb.nn.Linear(3, 1, seed=7)
loss_tensor = head.forward(tanh.forward(linear.forward(left)))
loss_tensor.backward()
parameters = linear.parameters() + head.parameters()
assert all(parameter.grad is not None for parameter in parameters)
before = [parameter.data for parameter in parameters]
sgd = gb.optim.SGD(parameters, learning_rate=1e-2)
sgd.step()
after = [parameter.data for parameter in parameters]
assert any(old != new for old, new in zip(before, after))
sgd.zero_grad()
assert all(parameter.grad is None for parameter in parameters)

assert gb.isclose(left.exp().log(), left)
exp_log_loss = head.forward(linear.forward(left).exp().log())
exp_log_loss.backward()
assert all(parameter.grad is not None for parameter in parameters)
sgd.step()
sgd.zero_grad()
assert all(parameter.grad is None for parameter in parameters)

adam_loss = head.forward(relu.forward(linear.forward(left)))
adam_loss.backward()
adam = gb.optim.Adam(parameters, learning_rate=1e-3, weight_decay=1e-2)
adam.step()
adam.zero_grad()
assert all(parameter.grad is None for parameter in parameters)

detached_hidden = linear.forward(left).detach()
detached_loss = head.forward(tanh.forward(detached_hidden))
detached_loss.backward()
assert all(parameter.grad is None for parameter in linear.parameters())
assert all(parameter.grad is not None for parameter in head.parameters())
adam.zero_grad()

random.seed(42)
tokens = [[random.randrange(65) for _ in range(16)] for _ in range(2)]
model = gb.nn.GPT(65, 16, 2, 1, 8)
logits = model.forward(tokens)
assert logits.shape == [2, 16, 65]
assert len(logits.data) == 2 * 16 * 65

dataset = gb.data.CharDataset("data/tinyshakespeare.txt", 4, seed=42)
assert dataset.vocab_size == 65
inputs, targets = dataset.batch("train", 2)

granular = gb.nn.GPT(65, 4, 1, 1, 8, seed=42, learning_rate=1e-2, optimizer="sgd")
assert hasattr(gb.nn.GPT, "backward")
assert hasattr(gb.nn.GPT, "step")
assert hasattr(gb.nn.GPT, "zero_grad")
granular_loss = granular.loss(inputs, targets)
assert math.isfinite(granular_loss.value)
granular_norms = granular_loss.backward()
assert len(granular_norms) == len(granular.parameters())
granular.step()
granular.zero_grad()
second_loss = granular.loss(inputs, targets)
assert len(granular.backward(second_loss)) == len(granular.parameters())
granular.zero_grad()
trainer = gb.nn.GPT(65, 4, 1, 1, 8, seed=42, learning_rate=1e-2, optimizer="sgd")
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

adamw = gb.nn.GPT(65, 4, 1, 1, 8, seed=42, weight_decay=0.01)
adamw_loss, _ = adamw.train_step(inputs, targets)
assert math.isfinite(adamw_loss)

checkpoint = tempfile.NamedTemporaryFile(suffix=".gbx", delete=False)
checkpoint.close()
trainer.save_checkpoint(checkpoint.name)
expected = trainer.forward(inputs)
restored = gb.nn.GPT(65, 4, 1, 1, 8, seed=7)
restored.load_checkpoint(checkpoint.name)
actual = restored.forward(inputs)
assert actual.data == expected.data
prompt = dataset.encode("\n")
sample = restored.generate(prompt, 8, temperature=0.8, top_k=5, seed=9)
assert sample[: len(prompt)] == prompt
assert len(sample) == len(prompt) + 8
assert dataset.decode(sample).startswith("\n")
reloaded = gb.nn.GPT(65, 4, 1, 1, 8, seed=99)
reloaded.load_checkpoint(checkpoint.name)
assert reloaded.generate(prompt, 8, temperature=0.8, top_k=5, seed=9) == sample
os.remove(checkpoint.name)

try:
    gb.nn.GPT(65, 16, 1, 2, 8)
except ValueError as error:
    assert "supports n_head=1" in str(error)
else:
    raise AssertionError("multi-head construction should fail")

print("glassbox Python smoke test passed")
