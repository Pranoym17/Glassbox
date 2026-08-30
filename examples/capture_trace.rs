use glassbox::nn::Gpt;
use glassbox::visualizer::event_channel;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

struct Config {
    vocab: usize,
    sequence: usize,
    layers: usize,
    batch: usize,
    embedding: usize,
}

fn capture(path: &str, config: Config) -> Result<(), Box<dyn Error>> {
    let mut model = Gpt::new_with_seed(
        config.vocab,
        config.sequence,
        config.layers,
        1,
        config.embedding,
        1337,
    )?;
    let (emitter, receiver) = event_channel(16_384);
    model.set_event_emitter(Some(emitter.clone()));
    model.set_trace(0, true);
    let inputs: Vec<Vec<_>> = (0..config.batch)
        .map(|row| {
            (0..config.sequence)
                .map(|column| (row * 13 + column * 7) % config.vocab)
                .collect()
        })
        .collect();
    let targets: Vec<Vec<_>> = inputs
        .iter()
        .map(|sequence| {
            sequence
                .iter()
                .map(|token| (token + 1) % config.vocab)
                .collect()
        })
        .collect();

    let started = Instant::now();
    let loss = model.loss(&inputs, &targets)?;
    let _gradients = model.backward(loss)?;
    model.set_trace(0, false);
    let elapsed = started.elapsed();
    let events: Vec<_> = receiver.try_iter().collect();
    if emitter.dropped() != 0 {
        return Err(format!("fixture dropped {} events", emitter.dropped()).into());
    }
    let backward = events
        .iter()
        .position(|event| event.contains("\"phase\":\"backward\""))
        .ok_or("fixture has no backward events")?;
    if events[..backward]
        .iter()
        .any(|event| !event.contains("\"phase\":\"forward\""))
        || events[backward..]
            .iter()
            .any(|event| !event.contains("\"phase\":\"backward\""))
    {
        return Err("fixture phases are not ordered".into());
    }

    let output = format!(
        "[\n{}\n]\n",
        events
            .iter()
            .map(|event| format!("  {event}"))
            .collect::<Vec<_>>()
            .join(",\n")
    );
    let path = Path::new(path);
    std::fs::create_dir_all(path.parent().ok_or("fixture path has no parent")?)?;
    std::fs::write(path, output)?;
    let rate = events.len() as f64 / elapsed.as_secs_f64();
    println!(
        "{} events={} dropped=0 elapsed={:.3}s rate={:.0}/s",
        path.display(),
        events.len(),
        elapsed.as_secs_f64(),
        rate
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    capture(
        "visualizer/fixtures/step.json",
        Config {
            vocab: 65,
            sequence: 32,
            layers: 2,
            batch: 8,
            embedding: 64,
        },
    )?;
    capture(
        "visualizer/fixtures/tiny_step.json",
        Config {
            vocab: 17,
            sequence: 8,
            layers: 1,
            batch: 1,
            embedding: 8,
        },
    )
}
