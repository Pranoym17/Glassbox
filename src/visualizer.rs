use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Forward,
    Backward,
}

impl Phase {
    fn as_str(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Backward => "backward",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    pub step: u64,
    pub phase: Phase,
    pub op: String,
    pub output_id: usize,
    pub input_ids: Vec<usize>,
    pub shape: Vec<usize>,
    pub grad_norm: Option<f32>,
}

impl Event {
    pub fn to_json(&self) -> String {
        let mut json = String::new();
        json.push_str("{\"step\":");
        json.push_str(&self.step.to_string());
        json.push_str(",\"phase\":");
        push_json_string(&mut json, self.phase.as_str());
        json.push_str(",\"op\":");
        push_json_string(&mut json, &self.op);
        json.push_str(",\"output_id\":");
        json.push_str(&self.output_id.to_string());
        json.push_str(",\"input_ids\":");
        push_usize_array(&mut json, &self.input_ids);
        json.push_str(",\"shape\":");
        push_usize_array(&mut json, &self.shape);
        if let Some(norm) = self.grad_norm {
            json.push_str(",\"grad_norm\":");
            json.push_str(&norm.to_string());
        }
        json.push('}');
        json
    }
}

#[derive(Clone)]
pub struct EventEmitter {
    sender: SyncSender<String>,
    dropped: Arc<AtomicU64>,
}

impl EventEmitter {
    pub fn emit(&self, event: Event) {
        match self.sender.try_send(event.to_json()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

pub fn event_channel(capacity: usize) -> (EventEmitter, Receiver<String>) {
    let (sender, receiver) = sync_channel(capacity.max(1));
    (
        EventEmitter {
            sender,
            dropped: Arc::new(AtomicU64::new(0)),
        },
        receiver,
    )
}

fn push_usize_array(output: &mut String, values: &[usize]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&value.to_string());
    }
    output.push(']');
}

fn push_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", value as u32));
            }
            value => output.push(value),
        }
    }
    output.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(op: &str) -> Event {
        Event {
            step: 7,
            phase: Phase::Forward,
            op: op.to_string(),
            output_id: 3,
            input_ids: vec![1, 2],
            shape: vec![2, 4],
            grad_norm: None,
        }
    }

    #[test]
    fn json_is_compact_and_escapes_strings() {
        assert_eq!(
            event("add\n\"quoted\"").to_json(),
            "{\"step\":7,\"phase\":\"forward\",\"op\":\"add\\n\\\"quoted\\\"\",\"output_id\":3,\"input_ids\":[1,2],\"shape\":[2,4]}"
        );
    }

    #[test]
    fn a_full_channel_drops_without_blocking() {
        let (emitter, _receiver) = event_channel(1);
        emitter.emit(event("first"));
        emitter.emit(event("second"));
        assert_eq!(emitter.dropped(), 1);
    }
}
