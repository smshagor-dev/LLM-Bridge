//! Vendor-neutral remote model storage: a bounded HTTP/1.1 Range transport and a
//! secure, resumable local staging pipeline that publishes a fully SHA-256-verified
//! snapshot for authenticated (Mode B) local inference.
//!
//! Scope and non-goals. This is a *staging* downloader: the complete remote object
//! is transferred to local disk in bounded chunks, verified against a trusted
//! whole-file SHA-256, and atomically published; per-token weight reads are never
//! streamed over the network. Remote storage is a transport for bytes, not remote
//! computation, and model-size support is not unbounded (the model must fit on the
//! staging filesystem). Transport object-version tokens (ETag/Last-Modified/
//! VersionId) are treated as change detectors, explicitly distinct from the trusted
//! content digest, which is the only authority for publication.
//!
//! Dependencies: std only. TLS/HTTPS-to-public-internet and S3 are intentionally
//! deferred (see `docs/REMOTE_RANGE_READ_DESIGN.md`); only the `http` scheme is
//! accepted, which the deterministic local-server tests and the localhost 7B
//! staging experiment exercise end to end.

use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Typed transport/staging error taxonomy. Only `Transient`/`Io`-class errors are
/// retried; integrity, version and policy errors fail closed.
#[derive(Debug)]
pub enum RemoteError {
    InvalidUrl(String),
    UnsupportedScheme(String),
    DisallowedEndpoint(String),
    Io(String),
    Protocol(String),
    NotFound,
    PermissionDenied,
    PreconditionFailed,
    Transient(String),
    RangeUnsatisfiable,
    RangeOverflow,
    ShortRead,
    OversizedBody,
    UnsupportedEncoding(String),
    TooManyRedirects,
    ObjectChanged,
    IntegrityMismatch,
    Cancelled,
    DeadlineExpired,
    RetryExhausted(String),
    Staging(String),
}

impl fmt::Display for RemoteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RemoteError::InvalidUrl(m) => write!(f, "invalid url: {m}"),
            RemoteError::UnsupportedScheme(m) => write!(f, "unsupported scheme: {m}"),
            RemoteError::DisallowedEndpoint(m) => write!(f, "disallowed endpoint: {m}"),
            RemoteError::Io(m) => write!(f, "io error: {m}"),
            RemoteError::Protocol(m) => write!(f, "protocol error: {m}"),
            RemoteError::NotFound => write!(f, "object not found"),
            RemoteError::PermissionDenied => write!(f, "permission denied"),
            RemoteError::PreconditionFailed => {
                write!(f, "precondition failed (object version changed)")
            }
            RemoteError::Transient(m) => write!(f, "transient error: {m}"),
            RemoteError::RangeUnsatisfiable => write!(f, "range not satisfiable"),
            RemoteError::RangeOverflow => write!(f, "range arithmetic overflow"),
            RemoteError::ShortRead => write!(f, "short read: fewer bytes than requested"),
            RemoteError::OversizedBody => write!(f, "response body exceeds the requested range"),
            RemoteError::UnsupportedEncoding(m) => write!(f, "unsupported content encoding: {m}"),
            RemoteError::TooManyRedirects => write!(f, "too many redirects"),
            RemoteError::ObjectChanged => {
                write!(f, "remote object changed; refusing mixed-version bytes")
            }
            RemoteError::IntegrityMismatch => {
                write!(f, "whole-file SHA-256 mismatch; refusing publication")
            }
            RemoteError::Cancelled => write!(f, "cancelled"),
            RemoteError::DeadlineExpired => write!(f, "deadline expired"),
            RemoteError::RetryExhausted(m) => write!(f, "retries exhausted: {m}"),
            RemoteError::Staging(m) => write!(f, "staging error: {m}"),
        }
    }
}
impl std::error::Error for RemoteError {}
impl From<io::Error> for RemoteError {
    fn from(e: io::Error) -> Self {
        RemoteError::Io(e.to_string())
    }
}
impl RemoteError {
    fn is_retryable(&self) -> bool {
        matches!(
            self,
            RemoteError::Transient(_) | RemoteError::Io(_) | RemoteError::ShortRead
        )
    }
}

/// Provider version tokens. A change detector, NOT a cryptographic content proof.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ObjectValidator {
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub version_id: Option<String>,
}
impl ObjectValidator {
    fn is_empty(&self) -> bool {
        self.etag.is_none() && self.last_modified.is_none() && self.version_id.is_none()
    }
}

/// Stable identity pinned at open time: uri, length, and transport validator.
#[derive(Clone, Debug)]
pub struct ObjectIdentity {
    pub uri: String,
    pub size: u64,
    pub validator: ObjectValidator,
}

/// Vendor-neutral bounded range source. Implementors deliver `[offset, offset+len)`
/// into the caller's bounded buffer; they never allocate model-sized buffers.
pub trait RemoteSource {
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn identity(&self) -> &ObjectIdentity;
    fn read_range(
        &mut self,
        offset: u64,
        len: usize,
        buf: &mut [u8],
        cancelled: &mut dyn FnMut() -> bool,
        deadline: Option<Instant>,
    ) -> Result<(), RemoteError>;
}

fn check_active(
    cancelled: &mut dyn FnMut() -> bool,
    deadline: Option<Instant>,
) -> Result<(), RemoteError> {
    if cancelled() {
        return Err(RemoteError::Cancelled);
    }
    if let Some(d) = deadline {
        if Instant::now() >= d {
            return Err(RemoteError::DeadlineExpired);
        }
    }
    Ok(())
}

// ----------------------------- URL + endpoint policy -----------------------------

#[derive(Clone, Debug)]
struct Url {
    host: String,
    port: u16,
    path: String,
}

fn parse_http_url(raw: &str) -> Result<Url, RemoteError> {
    let rest = raw.strip_prefix("http://").ok_or_else(|| {
        if raw.starts_with("https://") {
            RemoteError::UnsupportedScheme(
                "https/TLS is deferred; use the http scheme or a TLS-terminating proxy".into(),
            )
        } else {
            RemoteError::InvalidUrl(format!("only http:// is supported: {raw}"))
        }
    })?;
    if rest.is_empty() {
        return Err(RemoteError::InvalidUrl("empty authority".into()));
    }
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if authority.contains('@') {
        return Err(RemoteError::InvalidUrl(
            "userinfo in authority is not allowed".into(),
        ));
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) => (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| RemoteError::InvalidUrl("bad port".into()))?,
        ),
        None => (authority.to_string(), 80u16),
    };
    if host.is_empty() {
        return Err(RemoteError::InvalidUrl("empty host".into()));
    }
    // Reject control characters / header-injection vectors in the request target.
    if path.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0)
        || host.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0)
    {
        return Err(RemoteError::InvalidUrl("control character in url".into()));
    }
    Ok(Url {
        host,
        port,
        path: path.to_string(),
    })
}

fn ip_is_private_or_local(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.octets()[0] == 0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                // Unique-local fc00::/7 and link-local fe80::/10 (checked manually,
                // since the std helpers are unstable).
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Resolve the authority and enforce the SSRF endpoint policy, returning the
/// permitted socket address. Private/loopback/link-local targets are rejected
/// unless `allow_private_endpoints` explicitly opts into a trusted-endpoint policy
/// (e.g. localhost tests or an S3-compatible service on a private network).
fn resolve_allowed(url: &Url, allow_private_endpoints: bool) -> Result<SocketAddr, RemoteError> {
    let addrs = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|e| RemoteError::DisallowedEndpoint(format!("resolve failed: {e}")))?;
    let mut last: Option<SocketAddr> = None;
    for addr in addrs {
        if ip_is_private_or_local(&addr.ip()) && !allow_private_endpoints {
            return Err(RemoteError::DisallowedEndpoint(format!(
                "{} resolves to a private/loopback address; set allow_private_endpoints for a trusted endpoint",
                url.host
            )));
        }
        last = Some(addr);
    }
    last.ok_or_else(|| RemoteError::DisallowedEndpoint("no address for host".into()))
}

// ----------------------------- HTTP Range adapter -----------------------------

#[derive(Clone, Debug)]
pub struct HttpRangeConfig {
    pub max_redirects: u32,
    pub max_header_bytes: usize,
    pub connect_timeout: Duration,
    pub io_timeout: Duration,
    pub max_retries: u32,
    pub backoff_base: Duration,
    pub allow_private_endpoints: bool,
    /// Optional authorization header value. Sent only to the origin host and
    /// dropped on cross-host redirects. Never logged or persisted.
    pub authorization: Option<String>,
}
impl Default for HttpRangeConfig {
    fn default() -> Self {
        Self {
            max_redirects: 5,
            max_header_bytes: 64 * 1024,
            connect_timeout: Duration::from_secs(15),
            io_timeout: Duration::from_secs(30),
            max_retries: 4,
            backoff_base: Duration::from_millis(200),
            allow_private_endpoints: false,
            authorization: None,
        }
    }
}

struct RangeResponse {
    start: u64,
    end: u64,
    total: u64,
    body_len: usize,
    validator: ObjectValidator,
}

#[derive(Debug)]
pub struct HttpRangeSource {
    url: Url,
    cfg: HttpRangeConfig,
    identity: ObjectIdentity,
}

impl HttpRangeSource {
    /// Open the source: resolve + SSRF-check the endpoint, probe the object with a
    /// one-byte range to learn its length and validator, and pin the identity.
    pub fn connect(
        raw_url: &str,
        expected_size: Option<u64>,
        cfg: HttpRangeConfig,
    ) -> Result<Self, RemoteError> {
        let url = parse_http_url(raw_url)?;
        resolve_allowed(&url, cfg.allow_private_endpoints)?;
        let mut probe = [0u8; 1];
        let mut never = || false;
        let resp = fetch_range_with_retry(&url, &cfg, 0, 1, &mut probe, &mut never, None)?;
        if let Some(want) = expected_size {
            if want != resp.total {
                return Err(RemoteError::ObjectChanged);
            }
        }
        let identity = ObjectIdentity {
            uri: raw_url.to_string(),
            size: resp.total,
            validator: resp.validator,
        };
        Ok(Self { url, cfg, identity })
    }
}

impl RemoteSource for HttpRangeSource {
    fn len(&self) -> u64 {
        self.identity.size
    }
    fn identity(&self) -> &ObjectIdentity {
        &self.identity
    }
    fn read_range(
        &mut self,
        offset: u64,
        len: usize,
        buf: &mut [u8],
        cancelled: &mut dyn FnMut() -> bool,
        deadline: Option<Instant>,
    ) -> Result<(), RemoteError> {
        if len == 0 {
            return Ok(());
        }
        if buf.len() < len {
            return Err(RemoteError::Staging(
                "read buffer smaller than requested len".into(),
            ));
        }
        let end = offset
            .checked_add(len as u64)
            .ok_or(RemoteError::RangeOverflow)?;
        if end > self.identity.size {
            return Err(RemoteError::RangeOverflow);
        }
        let resp = fetch_range_with_retry(
            &self.url,
            &self.cfg,
            offset,
            len,
            &mut buf[..len],
            cancelled,
            deadline,
        )?;
        // Fail closed on any version drift or length inconsistency across calls.
        if resp.total != self.identity.size {
            return Err(RemoteError::ObjectChanged);
        }
        if !self.identity.validator.is_empty()
            && !resp.validator.is_empty()
            && resp.validator != self.identity.validator
        {
            return Err(RemoteError::ObjectChanged);
        }
        if resp.start != offset || resp.end != offset + len as u64 - 1 || resp.body_len != len {
            return Err(RemoteError::Protocol(
                "content-range/length mismatch".into(),
            ));
        }
        Ok(())
    }
}

fn fetch_range_with_retry(
    url: &Url,
    cfg: &HttpRangeConfig,
    offset: u64,
    len: usize,
    buf: &mut [u8],
    cancelled: &mut dyn FnMut() -> bool,
    deadline: Option<Instant>,
) -> Result<RangeResponse, RemoteError> {
    let mut attempt = 0u32;
    loop {
        check_active(cancelled, deadline)?;
        match fetch_range_once(url, cfg, offset, len, buf, cancelled, deadline) {
            Ok(r) => return Ok(r),
            Err(e) => {
                if !e.is_retryable() || attempt >= cfg.max_retries {
                    if e.is_retryable() {
                        return Err(RemoteError::RetryExhausted(e.to_string()));
                    }
                    return Err(e);
                }
                attempt += 1;
                let backoff = cfg
                    .backoff_base
                    .checked_mul(1u32 << (attempt - 1).min(16))
                    .unwrap_or(cfg.backoff_base);
                let step = Duration::from_millis(50);
                let mut slept = Duration::ZERO;
                while slept < backoff {
                    check_active(cancelled, deadline)?;
                    std::thread::sleep(step.min(backoff - slept));
                    slept += step;
                }
            }
        }
    }
}

fn fetch_range_once(
    start_url: &Url,
    cfg: &HttpRangeConfig,
    offset: u64,
    len: usize,
    buf: &mut [u8],
    cancelled: &mut dyn FnMut() -> bool,
    deadline: Option<Instant>,
) -> Result<RangeResponse, RemoteError> {
    let mut url = start_url.clone();
    let mut carry_auth = true;
    let last = offset
        .checked_add(len as u64)
        .and_then(|e| e.checked_sub(1))
        .ok_or(RemoteError::RangeOverflow)?;

    for _redirect in 0..=cfg.max_redirects {
        check_active(cancelled, deadline)?;
        let addr = resolve_allowed(&url, cfg.allow_private_endpoints)?;
        let mut stream = TcpStream::connect_timeout(&addr, cfg.connect_timeout)
            .map_err(|e| RemoteError::Transient(format!("connect: {e}")))?;
        stream.set_read_timeout(Some(cfg.io_timeout)).ok();
        stream.set_write_timeout(Some(cfg.io_timeout)).ok();
        stream.set_nodelay(true).ok();

        let mut req = String::new();
        req.push_str(&format!("GET {} HTTP/1.1\r\n", url.path));
        req.push_str(&format!("Host: {}\r\n", host_header(&url)));
        req.push_str("User-Agent: llmbridge-range/1\r\n");
        req.push_str("Accept: */*\r\n");
        req.push_str("Accept-Encoding: identity\r\n");
        req.push_str(&format!("Range: bytes={offset}-{last}\r\n"));
        if carry_auth {
            if let Some(auth) = &cfg.authorization {
                req.push_str(&format!("Authorization: {auth}\r\n"));
            }
        }
        req.push_str("Connection: close\r\n\r\n");
        stream
            .write_all(req.as_bytes())
            .map_err(|e| RemoteError::Transient(format!("write: {e}")))?;

        let mut reader = BufReader::new(stream);
        let (status, headers) = read_status_and_headers(&mut reader, cfg.max_header_bytes)?;

        match status {
            206 => {
                let cr = headers
                    .get("content-range")
                    .ok_or_else(|| RemoteError::Protocol("206 without content-range".into()))?;
                let (c_start, c_end, c_total) = parse_content_range(cr)?;
                if let Some(enc) = headers.get("content-encoding") {
                    if !enc.eq_ignore_ascii_case("identity") {
                        return Err(RemoteError::UnsupportedEncoding(enc.clone()));
                    }
                }
                if headers.contains_key("transfer-encoding") {
                    return Err(RemoteError::Protocol(
                        "unexpected transfer-encoding on ranged response".into(),
                    ));
                }
                let clen: usize = headers
                    .get("content-length")
                    .ok_or_else(|| RemoteError::Protocol("206 without content-length".into()))?
                    .parse()
                    .map_err(|_| RemoteError::Protocol("bad content-length".into()))?;
                let expected = (c_end - c_start + 1) as usize;
                if clen != expected || expected != len {
                    return Err(RemoteError::Protocol(format!(
                        "length mismatch: content-length={clen} span={expected} requested={len}"
                    )));
                }
                if c_start != offset || c_end != last {
                    return Err(RemoteError::Protocol(
                        "content-range does not match request".into(),
                    ));
                }
                if clen > buf.len() {
                    return Err(RemoteError::OversizedBody);
                }
                check_active(cancelled, deadline)?;
                reader
                    .read_exact(&mut buf[..clen])
                    .map_err(|_| RemoteError::ShortRead)?;
                return Ok(RangeResponse {
                    start: c_start,
                    end: c_end,
                    total: c_total,
                    body_len: clen,
                    validator: validator_from(&headers),
                });
            }
            200 => {
                return Err(RemoteError::Protocol(
                    "unexpected 200 OK for a ranged request".into(),
                ))
            }
            301 | 302 | 303 | 307 | 308 => {
                let loc = headers
                    .get("location")
                    .ok_or_else(|| RemoteError::Protocol("redirect without location".into()))?;
                let next = parse_http_url(loc)?;
                if !host_eq(&url, &next) {
                    carry_auth = false;
                }
                url = next;
                continue;
            }
            404 => return Err(RemoteError::NotFound),
            401 | 403 => return Err(RemoteError::PermissionDenied),
            412 => return Err(RemoteError::PreconditionFailed),
            416 => return Err(RemoteError::RangeUnsatisfiable),
            429 | 500 | 502 | 503 | 504 => {
                return Err(RemoteError::Transient(format!("status {status}")))
            }
            other => return Err(RemoteError::Protocol(format!("unexpected status {other}"))),
        }
    }
    Err(RemoteError::TooManyRedirects)
}

fn host_header(url: &Url) -> String {
    if url.port == 80 {
        url.host.clone()
    } else {
        format!("{}:{}", url.host, url.port)
    }
}
fn host_eq(a: &Url, b: &Url) -> bool {
    a.host.eq_ignore_ascii_case(&b.host) && a.port == b.port
}

type Headers = std::collections::BTreeMap<String, String>;

fn read_status_and_headers<R: BufRead>(
    reader: &mut R,
    max_header_bytes: usize,
) -> Result<(u16, Headers), RemoteError> {
    let mut consumed = 0usize;
    let mut line = String::new();
    let n = reader
        .read_line(&mut line)
        .map_err(|e| RemoteError::Transient(format!("read status: {e}")))?;
    if n == 0 {
        return Err(RemoteError::Transient("empty response".into()));
    }
    consumed += n;
    let status = parse_status_line(&line)?;
    let mut headers = Headers::new();
    loop {
        if consumed > max_header_bytes {
            return Err(RemoteError::Protocol(
                "response headers exceed bound".into(),
            ));
        }
        let mut h = String::new();
        let n = reader
            .read_line(&mut h)
            .map_err(|e| RemoteError::Transient(format!("read header: {e}")))?;
        if n == 0 {
            return Err(RemoteError::Transient("truncated headers".into()));
        }
        consumed += n;
        let trimmed = h.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    Ok((status, headers))
}

fn parse_status_line(line: &str) -> Result<u16, RemoteError> {
    let mut parts = line.split_whitespace();
    let proto = parts.next().unwrap_or("");
    if !proto.starts_with("HTTP/") {
        return Err(RemoteError::Protocol("not an HTTP response".into()));
    }
    parts
        .next()
        .and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| RemoteError::Protocol("missing status code".into()))
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64), RemoteError> {
    // "bytes START-END/TOTAL"
    let rest = value
        .trim()
        .strip_prefix("bytes ")
        .ok_or_else(|| RemoteError::Protocol("content-range not in bytes".into()))?;
    let (range, total) = rest
        .split_once('/')
        .ok_or_else(|| RemoteError::Protocol("content-range missing total".into()))?;
    let (s, e) = range
        .split_once('-')
        .ok_or_else(|| RemoteError::Protocol("content-range missing end".into()))?;
    let start: u64 = s
        .trim()
        .parse()
        .map_err(|_| RemoteError::Protocol("bad range start".into()))?;
    let end: u64 = e
        .trim()
        .parse()
        .map_err(|_| RemoteError::Protocol("bad range end".into()))?;
    let total: u64 = total
        .trim()
        .parse()
        .map_err(|_| RemoteError::Protocol("bad total".into()))?;
    if end < start || end >= total {
        return Err(RemoteError::Protocol("inconsistent content-range".into()));
    }
    Ok((start, end, total))
}

fn validator_from(headers: &Headers) -> ObjectValidator {
    ObjectValidator {
        etag: headers.get("etag").cloned(),
        last_modified: headers.get("last-modified").cloned(),
        version_id: headers.get("x-amz-version-id").cloned(),
    }
}

// ----------------------------- Secure local staging -----------------------------

#[derive(Clone, Debug)]
pub struct StagingConfig {
    pub expected_size: u64,
    pub expected_sha256: String,
    pub chunk_bytes: usize,
    pub staging_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct PublishedSnapshot {
    pub path: PathBuf,
    pub size: u64,
    pub sha256: String,
    pub identity: ObjectIdentity,
    pub bytes_transferred: u64,
    pub chunks_transferred: u64,
    pub chunks_resumed: u64,
}

fn staging_err(m: impl Into<String>) -> RemoteError {
    RemoteError::Staging(m.into())
}

fn is_valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

#[allow(clippy::unnecessary_cast)] // statvfs field widths are platform-dependent
fn available_disk_bytes(_dir: &Path) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(_dir.as_os_str().as_bytes()).ok()?;
        // SAFETY: zeroed statvfs is a valid out-param for a path we own.
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        Some(st.f_bavail as u64 * st.f_frsize as u64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn open_private_part(path: &Path) -> Result<File, RemoteError> {
    let mut opts = OpenOptions::new();
    opts.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
        opts.custom_flags(libc::O_NOFOLLOW);
    }
    opts.open(path)
        .map_err(|e| staging_err(format!("open staging file: {e}")))
}

fn pwrite_all(file: &File, offset: u64, data: &[u8]) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(data, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut written = 0usize;
        while written < data.len() {
            let n = file.seek_write(&data[written..], offset + written as u64)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "seek_write wrote zero",
                ));
            }
            written += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, offset, data);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "pwrite unsupported",
        ))
    }
}

fn state_path(part: &Path) -> PathBuf {
    let mut s = part.as_os_str().to_owned();
    s.push(".state");
    PathBuf::from(s)
}

fn load_resume_bitmap(state: &Path, cfg: &StagingConfig, chunks: usize) -> Option<Vec<bool>> {
    let data = fs::read(state).ok()?;
    // Header: "LBPART1\n<sha>\n<size>\n<chunk_bytes>\n" then raw 0/1 bytes.
    let text_end = data
        .iter()
        .enumerate()
        .filter(|(_, b)| **b == b'\n')
        .nth(3)
        .map(|(i, _)| i + 1)?;
    let header = std::str::from_utf8(&data[..text_end]).ok()?;
    let mut lines = header.lines();
    if lines.next()? != "LBPART1" {
        return None;
    }
    if lines.next()? != cfg.expected_sha256 {
        return None;
    }
    if lines.next()?.parse::<u64>().ok()? != cfg.expected_size {
        return None;
    }
    if lines.next()?.parse::<usize>().ok()? != cfg.chunk_bytes {
        return None;
    }
    let bits = &data[text_end..];
    if bits.len() != chunks {
        return None;
    }
    Some(bits.iter().map(|b| *b == 1).collect())
}

fn save_resume_bitmap(state: &Path, cfg: &StagingConfig, bitmap: &[bool]) -> io::Result<()> {
    let mut out = Vec::with_capacity(64 + bitmap.len());
    out.extend_from_slice(
        format!(
            "LBPART1\n{}\n{}\n{}\n",
            cfg.expected_sha256, cfg.expected_size, cfg.chunk_bytes
        )
        .as_bytes(),
    );
    out.extend(bitmap.iter().map(|b| if *b { 1u8 } else { 0u8 }));
    let tmp = {
        let mut s = state.as_os_str().to_owned();
        s.push(".tmp");
        PathBuf::from(s)
    };
    fs::write(&tmp, &out)?;
    fs::rename(&tmp, state)
}

fn hash_file(path: &Path, buf_bytes: usize) -> Result<String, RemoteError> {
    let mut f = File::open(path).map_err(|e| staging_err(format!("reopen staging: {e}")))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; buf_bytes.max(64 * 1024)];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| staging_err(format!("hash read: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn fsync_dir(dir: &Path) {
    #[cfg(unix)]
    {
        if let Ok(f) = File::open(dir) {
            let _ = f.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
}

/// Download `source` fully into a private staging file, verify it against the
/// trusted whole-file SHA-256, and atomically publish it at `output`. Fail closed:
/// a partial or mismatched file is never published, user originals are never
/// touched, and only a whole-file digest match authorises publication.
pub fn stage_to_file(
    source: &mut dyn RemoteSource,
    output: &Path,
    cfg: &StagingConfig,
    cancelled: &mut dyn FnMut() -> bool,
    deadline: Option<Instant>,
) -> Result<PublishedSnapshot, RemoteError> {
    if !is_valid_sha256(&cfg.expected_sha256) {
        return Err(staging_err("expected SHA-256 must be 64 hex digits"));
    }
    if cfg.chunk_bytes == 0 || cfg.chunk_bytes > 64 * 1024 * 1024 {
        return Err(staging_err("chunk size out of bounds (1..=64MiB)"));
    }
    if cfg.expected_size != source.len() {
        return Err(RemoteError::ObjectChanged);
    }
    // Path-traversal and symlink guards on the publication target.
    if output
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(staging_err("output path must not contain '..'"));
    }
    if let Ok(meta) = fs::symlink_metadata(output) {
        if meta.file_type().is_symlink() {
            return Err(staging_err("output path is a symlink"));
        }
        return Err(staging_err("output already exists; refusing to overwrite"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(staging_err("output parent directory does not exist"));
    }
    if let Some(avail) = available_disk_bytes(&cfg.staging_dir) {
        if avail < cfg.expected_size {
            return Err(staging_err(format!(
                "insufficient disk: need {} bytes, available {}",
                cfg.expected_size, avail
            )));
        }
    }

    let file_name = output
        .file_name()
        .ok_or_else(|| staging_err("output has no file name"))?;
    let mut part_name = file_name.to_owned();
    part_name.push(".part");
    let part = cfg.staging_dir.join(&part_name);
    let state = state_path(&part);

    let chunks = cfg.expected_size.div_ceil(cfg.chunk_bytes as u64) as usize;
    let mut bitmap = load_resume_bitmap(&state, cfg, chunks).unwrap_or_else(|| vec![false; chunks]);
    let resumed_chunks = bitmap.iter().filter(|b| **b).count() as u64;

    let file = open_private_part(&part)?;
    file.set_len(cfg.expected_size)
        .map_err(|e| staging_err(format!("set_len: {e}")))?;

    let mut buf = vec![0u8; cfg.chunk_bytes];
    let mut bytes_transferred = 0u64;
    let mut chunks_transferred = 0u64;
    for i in 0..chunks {
        if bitmap[i] {
            continue;
        }
        check_active(cancelled, deadline)?;
        let offset = (i as u64) * cfg.chunk_bytes as u64;
        let len = cfg.chunk_bytes.min((cfg.expected_size - offset) as usize);
        source.read_range(offset, len, &mut buf, cancelled, deadline)?;
        pwrite_all(&file, offset, &buf[..len]).map_err(|e| staging_err(format!("pwrite: {e}")))?;
        bitmap[i] = true;
        bytes_transferred += len as u64;
        chunks_transferred += 1;
        // Persist resume state after every completed chunk so an interruption can
        // always resume; fsync the data periodically to bound sync overhead.
        save_resume_bitmap(&state, cfg, &bitmap).map_err(|e| staging_err(format!("state: {e}")))?;
        if chunks_transferred % 16 == 0 {
            file.sync_data().ok();
        }
    }
    file.sync_all()
        .map_err(|e| staging_err(format!("fsync part: {e}")))?;
    save_resume_bitmap(&state, cfg, &bitmap).ok();

    // Whole-file verification gates publication. A mismatch keeps the .part for a
    // later retry and never publishes.
    let actual = hash_file(&part, cfg.chunk_bytes)?;
    if !actual.eq_ignore_ascii_case(&cfg.expected_sha256) {
        return Err(RemoteError::IntegrityMismatch);
    }

    // Atomic publication: rename within the same directory, then fsync the dir.
    fs::rename(&part, output).map_err(|e| staging_err(format!("publish rename: {e}")))?;
    fsync_dir(parent);
    let _ = fs::remove_file(&state);

    Ok(PublishedSnapshot {
        path: output.to_path_buf(),
        size: cfg.expected_size,
        sha256: actual,
        identity: source.identity().clone(),
        bytes_transferred,
        chunks_transferred,
        chunks_resumed: resumed_chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{SocketAddr, TcpListener};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread::JoinHandle;

    enum Action {
        Respond(Vec<u8>),
    }

    struct TestServer {
        addr: SocketAddr,
        shutdown: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }
    impl TestServer {
        fn spawn<F>(handler: F) -> Self
        where
            F: Fn(usize, &str) -> Action + Send + Sync + 'static,
        {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let shutdown = Arc::new(AtomicBool::new(false));
            let stop = Arc::clone(&shutdown);
            let handler = Arc::new(handler);
            let handle = std::thread::spawn(move || {
                let mut conn = 0usize;
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                            let mut reader = BufReader::new(stream.try_clone().unwrap());
                            let mut head = String::new();
                            loop {
                                let mut line = String::new();
                                match reader.read_line(&mut line) {
                                    Ok(0) => break,
                                    Ok(_) => {
                                        let done = line == "\r\n" || line == "\n";
                                        head.push_str(&line);
                                        if done {
                                            break;
                                        }
                                    }
                                    Err(_) => break,
                                }
                            }
                            let idx = conn;
                            conn += 1;
                            match handler(idx, &head) {
                                Action::Respond(bytes) => {
                                    let _ = stream.write_all(&bytes);
                                    let _ = stream.flush();
                                }
                            }
                        }
                        Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(3));
                        }
                        Err(_) => break,
                    }
                }
            });
            TestServer {
                addr,
                shutdown,
                handle: Some(handle),
            }
        }
        fn url(&self) -> String {
            format!("http://{}/model", self.addr)
        }
        fn cfg() -> HttpRangeConfig {
            HttpRangeConfig {
                allow_private_endpoints: true,
                max_retries: 2,
                backoff_base: Duration::from_millis(1),
                ..Default::default()
            }
        }
    }
    impl Drop for TestServer {
        fn drop(&mut self) {
            self.shutdown.store(true, Ordering::Relaxed);
            // Unblock accept and let the thread observe shutdown.
            let _ = TcpStream::connect(self.addr);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn parse_req_range(head: &str) -> Option<(u64, u64)> {
        for line in head.lines() {
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("range:") {
                let v = v.trim();
                let spec = v.strip_prefix("bytes=")?;
                let (a, b) = spec.split_once('-')?;
                return Some((a.trim().parse().ok()?, b.trim().parse().ok()?));
            }
        }
        None
    }
    fn has_auth(head: &str) -> bool {
        head.lines()
            .any(|l| l.to_ascii_lowercase().starts_with("authorization:"))
    }

    fn serve_206(content: &[u8], head: &str) -> Action {
        let (a, b) = parse_req_range(head).unwrap();
        let (a, b) = (a as usize, b as usize);
        let body = &content[a..=b.min(content.len() - 1)];
        let mut resp = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\nETag: \"v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
            a,
            a + body.len() - 1,
            content.len(),
            body.len()
        )
        .into_bytes();
        resp.extend_from_slice(body);
        Action::Respond(resp)
    }

    fn fixture(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + 13) as u8).collect()
    }

    // ------------------------------- transport -------------------------------

    #[test]
    fn valid_range_download_roundtrip_and_identity() {
        let content = fixture(1000);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let mut src =
            HttpRangeSource::connect(&server.url(), Some(1000), TestServer::cfg()).unwrap();
        assert_eq!(src.len(), 1000);
        assert_eq!(src.identity().validator.etag.as_deref(), Some("\"v1\""));
        let mut buf = vec![0u8; 256];
        src.read_range(100, 200, &mut buf, &mut || false, None)
            .unwrap();
        assert_eq!(&buf[..200], &content[100..300]);
    }

    #[test]
    fn unexpected_200_is_rejected() {
        let server = TestServer::spawn(|_, _| {
            Action::Respond(
                b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc".to_vec(),
            )
        });
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(matches!(err, RemoteError::Protocol(_)), "{err}");
    }

    #[test]
    fn wrong_content_range_total_is_object_changed() {
        let content = fixture(500);
        let c2 = content.clone();
        // Probe returns total 500; a later call returns a different total.
        let server = TestServer::spawn(move |idx, head| {
            if idx == 0 {
                serve_206(&c2, head)
            } else {
                let (a, b) = parse_req_range(head).unwrap();
                let body = &c2[a as usize..=(b as usize)];
                let mut resp = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/999\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    a, b, body.len()
                ).into_bytes();
                resp.extend_from_slice(body);
                Action::Respond(resp)
            }
        });
        let mut src = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap();
        let mut buf = vec![0u8; 16];
        let err = src
            .read_range(0, 16, &mut buf, &mut || false, None)
            .unwrap_err();
        assert!(matches!(err, RemoteError::ObjectChanged), "{err}");
    }

    #[test]
    fn content_length_mismatch_rejected() {
        let server = TestServer::spawn(|_, head| {
            let (a, b) = parse_req_range(head).unwrap();
            // Claim one more byte than the range spans.
            let span = b - a + 1;
            let mut resp = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/1000\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                a, b, span + 1
            ).into_bytes();
            resp.extend(std::iter::repeat(0u8).take((span + 1) as usize));
            Action::Respond(resp)
        });
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(matches!(err, RemoteError::Protocol(_)), "{err}");
    }

    #[test]
    fn gzip_encoding_rejected() {
        let server = TestServer::spawn(|_, head| {
            let (a, b) = parse_req_range(head).unwrap();
            let span = (b - a + 1) as usize;
            let mut resp = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/1000\r\nContent-Length: {}\r\nContent-Encoding: gzip\r\nConnection: close\r\n\r\n",
                a, b, span
            ).into_bytes();
            resp.extend(std::iter::repeat(0u8).take(span));
            Action::Respond(resp)
        });
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(matches!(err, RemoteError::UnsupportedEncoding(_)), "{err}");
    }

    #[test]
    fn redirect_loop_exhausts() {
        let server = TestServer::spawn(|_, _| {
            Action::Respond(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/model\r\nConnection: close\r\n\r\n".to_vec())
        });
        // Allow private so the redirect target policy is not the thing under test;
        // the loop/limit is.
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(
            matches!(
                err,
                RemoteError::TooManyRedirects
                    | RemoteError::Transient(_)
                    | RemoteError::RetryExhausted(_)
            ),
            "{err}"
        );
    }

    #[test]
    fn transient_500_then_success_retries() {
        let content = fixture(64);
        let c2 = content.clone();
        let server = TestServer::spawn(move |idx, head| {
            if idx == 0 {
                Action::Respond(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec())
            } else {
                serve_206(&c2, head)
            }
        });
        let src = HttpRangeSource::connect(&server.url(), Some(64), TestServer::cfg());
        assert!(src.is_ok(), "{:?}", src.err());
    }

    #[test]
    fn persistent_503_exhausts_retries() {
        let server = TestServer::spawn(|_, _| {
            Action::Respond(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec())
        });
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(matches!(err, RemoteError::RetryExhausted(_)), "{err}");
    }

    #[test]
    fn truncated_body_is_short_read_then_retry_exhausted() {
        let server = TestServer::spawn(|_, head| {
            let (a, b) = parse_req_range(head).unwrap();
            let span = (b - a + 1) as usize;
            // Advertise span bytes but send only half, then close.
            let mut resp = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/1000\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                a, b, span
            ).into_bytes();
            resp.extend(std::iter::repeat(1u8).take(span / 2));
            Action::Respond(resp)
        });
        // Probe is 1 byte (span=1, half=0) -> read_exact of 1 from empty body fails.
        let err = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap_err();
        assert!(matches!(err, RemoteError::RetryExhausted(_)), "{err}");
    }

    #[test]
    fn private_endpoint_rejected_without_opt_in() {
        let content = fixture(32);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let cfg = HttpRangeConfig::default(); // allow_private_endpoints = false
        let err = HttpRangeSource::connect(&server.url(), None, cfg).unwrap_err();
        assert!(matches!(err, RemoteError::DisallowedEndpoint(_)), "{err}");
    }

    #[test]
    fn https_scheme_is_unsupported() {
        let err = HttpRangeSource::connect(
            "https://example.com/model",
            None,
            HttpRangeConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(err, RemoteError::UnsupportedScheme(_)), "{err}");
    }

    #[test]
    fn range_overflow_rejected() {
        let content = fixture(100);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let mut src = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap();
        let mut buf = vec![0u8; 64];
        let err = src
            .read_range(90, 64, &mut buf, &mut || false, None)
            .unwrap_err();
        assert!(matches!(err, RemoteError::RangeOverflow), "{err}");
    }

    #[test]
    fn cancellation_during_fetch_is_reported() {
        let content = fixture(100);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let mut src = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap();
        let mut buf = vec![0u8; 16];
        let err = src
            .read_range(0, 16, &mut buf, &mut || true, None)
            .unwrap_err();
        assert!(matches!(err, RemoteError::Cancelled), "{err}");
    }

    #[test]
    fn credentials_are_dropped_on_cross_host_redirect() {
        let content = fixture(64);
        let c2 = content.clone();
        let saw_auth = Arc::new(AtomicBool::new(false));
        let saw = Arc::clone(&saw_auth);
        // Target server records whether Authorization reached it.
        let target = TestServer::spawn(move |_, head| {
            if has_auth(head) {
                saw.store(true, Ordering::Relaxed);
            }
            serve_206(&c2, head)
        });
        let target_url = target.url();
        // Origin redirects (different port => different host) to the target.
        let origin = TestServer::spawn(move |_, _| {
            Action::Respond(
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: {target_url}\r\nConnection: close\r\n\r\n"
                )
                .into_bytes(),
            )
        });
        let cfg = HttpRangeConfig {
            allow_private_endpoints: true,
            authorization: Some("Bearer secret-token".into()),
            max_retries: 1,
            backoff_base: Duration::from_millis(1),
            ..Default::default()
        };
        let src = HttpRangeSource::connect(&origin.url(), Some(64), cfg);
        assert!(src.is_ok(), "{:?}", src.err());
        assert!(
            !saw_auth.load(Ordering::Relaxed),
            "authorization leaked across host redirect"
        );
    }

    // ------------------------------- staging -------------------------------

    fn staging_dir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "llmbridge-stage-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&d).unwrap();
        d
    }
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn sha_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn full_download_verifies_and_publishes_atomically() {
        let content = fixture(5000);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        let mut src =
            HttpRangeSource::connect(&server.url(), Some(5000), TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 5000,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 512,
            staging_dir: dir.clone(),
        };
        let pub_ = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap();
        assert_eq!(pub_.sha256, sha_hex(&content));
        assert_eq!(fs::read(&out).unwrap(), content);
        assert!(!dir.join("model.gguf.part").exists());
        assert!(!dir.join("model.gguf.part.state").exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_sha_fails_closed_and_does_not_publish() {
        let content = fixture(2048);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        let mut src =
            HttpRangeSource::connect(&server.url(), Some(2048), TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 2048,
            expected_sha256: "0".repeat(64),
            chunk_bytes: 512,
            staging_dir: dir.clone(),
        };
        let err = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::IntegrityMismatch), "{err}");
        assert!(!out.exists(), "partial must never be published");
        assert!(
            dir.join("model.gguf.part").exists(),
            "part retained for retry"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn wrong_expected_size_is_object_changed() {
        let content = fixture(1000);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        let mut src = HttpRangeSource::connect(&server.url(), None, TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 999, // != actual 1000
            expected_sha256: sha_hex(&content),
            chunk_bytes: 256,
            staging_dir: dir.clone(),
        };
        let err = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::ObjectChanged), "{err}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resume_completes_after_interruption() {
        let content = fixture(4096);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        let cfg = StagingConfig {
            expected_size: 4096,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 256, // 16 chunks
            staging_dir: dir.clone(),
        };
        // First pass: cancel after a few chunks are written.
        let served = AtomicUsize::new(0);
        let mut src =
            HttpRangeSource::connect(&server.url(), Some(4096), TestServer::cfg()).unwrap();
        let err = stage_to_file(
            &mut src,
            &out,
            &cfg,
            &mut || served.fetch_add(1, Ordering::Relaxed) >= 10,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, RemoteError::Cancelled), "{err}");
        assert!(!out.exists());
        assert!(dir.join("model.gguf.part.state").exists());
        // Second pass: resumes and publishes.
        let mut src2 =
            HttpRangeSource::connect(&server.url(), Some(4096), TestServer::cfg()).unwrap();
        let pub_ = stage_to_file(&mut src2, &out, &cfg, &mut || false, None).unwrap();
        assert!(pub_.chunks_resumed > 0, "expected resumed chunks");
        assert_eq!(fs::read(&out).unwrap(), content);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn corrupted_resume_state_is_caught_by_whole_file_hash() {
        let content = fixture(1024);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        // Pre-seed a .part of correct size but wrong bytes and a state marking ALL
        // chunks complete. The whole-file hash must still catch the corruption.
        let part = dir.join("model.gguf.part");
        fs::write(&part, vec![0xABu8; 1024]).unwrap();
        let cfg = StagingConfig {
            expected_size: 1024,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 256,
            staging_dir: dir.clone(),
        };
        let chunks = 4;
        save_resume_bitmap(&state_path(&part), &cfg, &vec![true; chunks]).unwrap();
        let mut src =
            HttpRangeSource::connect(&server.url(), Some(1024), TestServer::cfg()).unwrap();
        let err = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::IntegrityMismatch), "{err}");
        assert!(!out.exists());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn output_path_traversal_rejected() {
        let content = fixture(32);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let mut src = HttpRangeSource::connect(&server.url(), Some(32), TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 32,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 16,
            staging_dir: dir.clone(),
        };
        let out = dir.join("../escape.gguf");
        let err = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::Staging(_)), "{err}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn existing_output_is_not_overwritten() {
        let content = fixture(64);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let out = dir.join("model.gguf");
        fs::write(&out, b"precious user file").unwrap();
        let mut src = HttpRangeSource::connect(&server.url(), Some(64), TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 64,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 16,
            staging_dir: dir.clone(),
        };
        let err = stage_to_file(&mut src, &out, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::Staging(_)), "{err}");
        assert_eq!(fs::read(&out).unwrap(), b"precious user file");
        fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn symlink_output_rejected() {
        let content = fixture(32);
        let c2 = content.clone();
        let server = TestServer::spawn(move |_, head| serve_206(&c2, head));
        let dir = staging_dir();
        let target = dir.join("real.bin");
        fs::write(&target, b"x").unwrap();
        let link = dir.join("model.gguf");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let mut src = HttpRangeSource::connect(&server.url(), Some(32), TestServer::cfg()).unwrap();
        let cfg = StagingConfig {
            expected_size: 32,
            expected_sha256: sha_hex(&content),
            chunk_bytes: 16,
            staging_dir: dir.clone(),
        };
        let err = stage_to_file(&mut src, &link, &cfg, &mut || false, None).unwrap_err();
        assert!(matches!(err, RemoteError::Staging(_)), "{err}");
        fs::remove_dir_all(&dir).ok();
    }
}
