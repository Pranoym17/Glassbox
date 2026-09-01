use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::thread;
use std::time::Duration;

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

pub const DEFAULT_PORT: u16 = 8080;
const INDEX_HTML: &str = include_str!("../visualizer/index.html");
const APP_JS: &str = include_str!("../visualizer/app.js");
const D3_JS: &str = include_str!("../visualizer/vendor/d3.v7.9.0.min.js");
const STEP_JSON: &str = include_str!("../visualizer/fixtures/step.json");
const TINY_STEP_JSON: &str = include_str!("../visualizer/fixtures/tiny_step.json");

pub struct VisualizerServer {
    emitter: EventEmitter,
    address: SocketAddr,
    running: Arc<AtomicBool>,
}

impl VisualizerServer {
    pub fn start(port: u16, capacity: usize) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        listener.set_nonblocking(true)?;
        let address = listener.local_addr()?;
        let (emitter, receiver) = event_channel(capacity);
        let clients = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicBool::new(true));

        let broadcast_clients = clients.clone();
        let broadcast_running = running.clone();
        thread::spawn(move || broadcast(receiver, broadcast_clients, broadcast_running));

        let accept_running = running.clone();
        thread::spawn(move || accept_connections(listener, clients, accept_running));

        Ok(Self {
            emitter,
            address,
            running,
        })
    }

    pub fn start_default(capacity: usize) -> std::io::Result<Self> {
        Self::start(DEFAULT_PORT, capacity)
    }

    pub fn emitter(&self) -> EventEmitter {
        self.emitter.clone()
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn url(&self) -> String {
        format!("http://{}", self.address)
    }

    pub fn dropped(&self) -> u64 {
        self.emitter.dropped()
    }
}

impl Drop for VisualizerServer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

fn broadcast(
    receiver: Receiver<String>,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        match receiver.recv_timeout(Duration::from_millis(50)) {
            Ok(event) => {
                let message = format!("data: {event}\n\n");
                clients
                    .lock()
                    .expect("visualizer clients lock poisoned")
                    .retain_mut(|client| client.write_all(message.as_bytes()).is_ok());
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn accept_connections(
    listener: TcpListener,
    clients: Arc<Mutex<Vec<TcpStream>>>,
    running: Arc<AtomicBool>,
) {
    while running.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => handle_connection(stream, &clients),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(10));
            }
            Err(_) => break,
        }
    }
}

fn handle_connection(mut stream: TcpStream, clients: &Arc<Mutex<Vec<TcpStream>>>) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(250)));
    let mut buffer = [0_u8; 8192];
    let Ok(length) = stream.read(&mut buffer) else {
        return;
    };
    let request = String::from_utf8_lossy(&buffer[..length]);
    let path = request.split_whitespace().nth(1).unwrap_or("/");
    if path == "/events" {
        let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n";
        let _ = stream.set_write_timeout(Some(Duration::from_millis(100)));
        let mut clients = clients.lock().expect("visualizer clients lock poisoned");
        if stream.write_all(headers.as_bytes()).is_ok() {
            clients.push(stream);
        }
        return;
    }
    let asset = match path {
        "/" | "/index.html" => Some((INDEX_HTML, "text/html; charset=utf-8")),
        "/app.js" => Some((APP_JS, "text/javascript; charset=utf-8")),
        "/vendor/d3.v7.9.0.min.js" => Some((D3_JS, "text/javascript; charset=utf-8")),
        "/fixtures/step.json" => Some((STEP_JSON, "application/json")),
        "/fixtures/tiny_step.json" => Some((TINY_STEP_JSON, "application/json")),
        _ => None,
    };
    if let Some((body, content_type)) = asset {
        serve_asset(&mut stream, body, content_type);
        return;
    }
    let body = "not found";
    let response = format!(
        "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
}
fn serve_asset(stream: &mut TcpStream, body: &str, content_type: &str) {
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(headers.as_bytes());
    let _ = stream.write_all(body.as_bytes());
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

    fn read_until(stream: &mut TcpStream, needle: &str) -> String {
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let mut output = String::new();
        let mut buffer = [0_u8; 4096];
        while !output.contains(needle) && std::time::Instant::now() < deadline {
            match stream.read(&mut buffer) {
                Ok(0) => break,
                Ok(length) => output.push_str(&String::from_utf8_lossy(&buffer[..length])),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) => {}
                Err(error) => panic!("read failed: {error}"),
            }
        }
        output
    }

    #[test]
    fn server_serves_page_and_ordered_sse_data_lines() {
        let server = VisualizerServer::start(0, 16).unwrap();

        let mut page = TcpStream::connect(server.address()).unwrap();
        page.write_all(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        page.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut response = String::new();
        page.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("<script type=\"module\" src=\"app.js\">"));
        assert!(response.contains("<script src=\"vendor/d3.v7.9.0.min.js\">"));
        assert!(response.contains("<span id=\"connection\">connecting</span>"));
        assert!(!response.contains("fixture mode"));

        let mut d3 = TcpStream::connect(server.address()).unwrap();
        d3.write_all(b"GET /vendor/d3.v7.9.0.min.js HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        d3.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut d3_response = String::new();
        d3.read_to_string(&mut d3_response).unwrap();
        assert!(d3_response.contains("Content-Type: text/javascript"));
        assert!(d3_response.contains("d3js.org v7.9.0"));

        let mut fixture = TcpStream::connect(server.address()).unwrap();
        fixture
            .write_all(b"GET /fixtures/tiny_step.json HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        fixture
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut fixture_response = String::new();
        fixture.read_to_string(&mut fixture_response).unwrap();
        assert!(fixture_response.contains("Content-Type: application/json"));
        assert!(fixture_response.contains("\"phase\":\"forward\""));

        let mut events = TcpStream::connect(server.address()).unwrap();
        events
            .write_all(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        events
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        let headers = read_until(&mut events, "\r\n\r\n");
        assert!(headers.starts_with("HTTP/1.1 200 OK"));
        assert!(headers.contains("Content-Type: text/event-stream"));

        server.emitter().emit(event("first"));
        server.emitter().emit(event("second"));
        let body = read_until(&mut events, "\"op\":\"second\"");
        let lines: Vec<_> = body
            .lines()
            .filter(|line| line.starts_with("data: "))
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"op\":\"first\""));
        assert!(lines[1].contains("\"op\":\"second\""));
    }
}
