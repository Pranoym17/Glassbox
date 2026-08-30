import argparse
import csv
import math
from pathlib import Path

import glassbox as gb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--output", default="artifacts/loss_curve.csv")
    parser.add_argument("--checkpoint", default="artifacts/tinyshakespeare.gbx")
    parser.add_argument("--visualizer", action="store_true")
    parser.add_argument("--visualizer-port", type=int, default=8080)
    parser.add_argument("--trace-interval", type=int, default=50)
    args = parser.parse_args()

    dataset = gb.data.CharDataset(
        "data/tinyshakespeare.txt", args.block_size, args.seed
    )
    model = gb.nn.GPT(
        dataset.vocab_size,
        args.block_size,
        args.n_layer,
        1,
        args.n_embd,
        seed=args.seed,
        learning_rate=args.learning_rate,
    )

    if args.visualizer:
        url = model.enable_visualizer(
            port=args.visualizer_port,
            trace_interval=args.trace_interval,
        )
        print(f"visualizer={url}")
    fixed_batch = (
        dataset.batch("train", args.batch_size) if args.overfit else None
    )
    curve = []
    for step in range(args.steps):
        inputs, targets = (
            fixed_batch
            if fixed_batch is not None
            else dataset.batch("train", args.batch_size)
        )
        loss, norms = model.train_step(inputs, targets, maximum_norm=1.0)
        if not math.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        validation = ""
        if step % args.eval_interval == 0 or step + 1 == args.steps:
            validation_inputs, validation_targets = dataset.batch(
                "validation", args.batch_size
            )
            validation = model.evaluate(validation_inputs, validation_targets)
            print(
                f"step={step:04d} train={loss:.6f} validation={validation:.6f}"
            )
            print(
                "gradient_norms="
                + ",".join(f"{index}:{norm:.3e}" for index, norm in enumerate(norms))
            )
        if step == 0:
            print(
                f"step0={loss:.6f} ln_vocab={math.log(dataset.vocab_size):.6f}"
            )
        curve.append((step, loss, validation))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["step", "train_loss", "validation_loss"])
        writer.writerows(curve)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(str(checkpoint))
    print(f"loss_curve={output}")
    print(f"checkpoint={checkpoint}")

    if args.visualizer:
        print(f"visualizer_dropped_events={model.dropped_events()}")


if __name__ == "__main__":
    main()
