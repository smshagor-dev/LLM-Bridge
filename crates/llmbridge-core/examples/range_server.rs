//! Minimal deterministic HTTP/1.1 Range file server for local integration testing
//! of the `remote` staging pipeline (std only). Serves exactly one file with correct
//! `206 Partial Content` + `Content-Range`/`Content-Length` semantics and a strong
//! ETag. Not a production server: loopback only, one file, no write methods.
//!
//! Usage: cargo run --example range_server --features remote -- <port> <file>

use std::fs;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::TcpListener;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: range_server <port> <file>");
        std::process::exit(2);
    }
    let port: u16 = args[1].parse().expect("port");
    let path = args[2].clone();
    let total = fs::metadata(&path).expect("stat file").len();
    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind");
    println!("RANGE_SERVER_READY port={port} size={total}");
    let _ = std::io::stdout().flush();

    for conn in listener.incoming() {
        let mut stream = match conn {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut reader = BufReader::new(match stream.try_clone() {
            Ok(s) => s,
            Err(_) => continue,
        });
        let mut head = String::new();
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let blank = line == "\r\n" || line == "\n";
                    head.push_str(&line);
                    if blank {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        let (start, end) = match parse_range(&head) {
            Some((s, e)) if e >= s && e < total => (s, e),
            _ => {
                let _ = stream.write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                continue;
            }
        };
        let len = (end - start + 1) as usize;
        let mut file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => continue,
        };
        if file.seek(SeekFrom::Start(start)).is_err() {
            continue;
        }
        let mut body = vec![0u8; len];
        if file.read_exact(&mut body).is_err() {
            continue;
        }
        let header = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{total}\r\nContent-Length: {len}\r\nETag: \"range-server-v1\"\r\nAccept-Ranges: bytes\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(header.as_bytes()).is_err() {
            continue;
        }
        let _ = stream.write_all(&body);
        let _ = stream.flush();
    }
}

fn parse_range(head: &str) -> Option<(u64, u64)> {
    for line in head.lines() {
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range:") {
            let spec = v.trim().strip_prefix("bytes=")?;
            let (a, b) = spec.split_once('-')?;
            return Some((a.trim().parse().ok()?, b.trim().parse().ok()?));
        }
    }
    None
}
