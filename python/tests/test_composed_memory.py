import gc

import glassbox as gb


def rss_kib():
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS is unavailable")


left = gb.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
linear = gb.nn.Linear(2, 8, seed=11)
head = gb.nn.Linear(8, 1, seed=12)
parameters = linear.parameters() + head.parameters()
optimizer = gb.optim.SGD(parameters, learning_rate=1e-3)

samples = {}
for iteration in range(1, 2001):
    loss = head.forward(linear.forward(left).tanh())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if iteration in (100, 500, 1000, 1500, 2000):
        gc.collect()
        samples[iteration] = rss_kib()

growth = samples[2000] - samples[100]
print(" ".join(f"{iteration}={rss}KiB" for iteration, rss in samples.items()))
print(f"growth_100_to_2000={growth}KiB")
assert growth <= 4096, f"composed tape RSS grew by {growth}KiB"
