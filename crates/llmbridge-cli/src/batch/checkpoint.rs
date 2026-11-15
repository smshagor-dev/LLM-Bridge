//! Verified local checkpoints; hashes are integrity records, not signed receipts.
use super::Options;
use llmbridge_core::storage::{BoundedLocalReader, SourceIdentity};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

pub(super) struct Journal {
    directory: PathBuf,
    identity: String,
    pub input_sha: String,
    _lock: File,
}
fn regular(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => Ok(true),
        Ok(_) => Err(format!(
            "checkpoint path is not a regular file: {}",
            path.display()
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}
fn small(path: &Path) -> Result<Vec<u8>, String> {
    if !regular(path)? {
        return Err(format!("checkpoint record missing: {}", path.display()));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(1025)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > 1024 {
        return Err("checkpoint record exceeds format limit".into());
    }
    Ok(bytes)
}
pub(super) fn hash_file(file: &mut File, cancelled: &AtomicBool) -> Result<(String, u64), String> {
    let before = SourceIdentity::from_metadata(&file.metadata().map_err(|e| e.to_string())?);
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err("batch checkpoint cancelled".into());
        }
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
        bytes = bytes
            .checked_add(count as u64)
            .ok_or("checkpoint length overflow")?;
    }
    if cancelled.load(Ordering::Acquire) {
        return Err("batch checkpoint cancelled".into());
    }
    if before != SourceIdentity::from_metadata(&file.metadata().map_err(|e| e.to_string())?) {
        return Err("checkpoint source changed while hashing".into());
    }
    file.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    Ok((format!("{:x}", hash.finalize()), bytes))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn lock(file: &File) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: the live File owns this descriptor for the Journal lifetime.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(format!(
                "batch output directory is locked or cannot be locked: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        Err("verified batch checkpoints require Linux file locking".into())
    }
}
fn atomic_record(directory: &Path, name: &str, text: &str) -> Result<(), String> {
    let final_path = directory.join(name);
    let temporary = directory.join(format!("{name}.partial"));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    file.write_all(text.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    fs::hard_link(&temporary, &final_path).map_err(|e| e.to_string())?;
    fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())
}
impl Journal {
    pub fn open(
        options: &Options,
        input: &mut File,
        cancelled: &AtomicBool,
    ) -> Result<Self, String> {
        if !cfg!(target_os = "linux") {
            return Err("verified batch checkpoints require Linux file locking".into());
        }
        // Existing runs acquire the kernel lease before expensive input/model scans.
        let existing_lock = if options.resume {
            if !regular(&options.output.join("batch.lock"))? {
                return Err("batch checkpoint lock missing".into());
            }
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(options.output.join("batch.lock"))
                .map_err(|e| e.to_string())?;
            lock(&file)?;
            Some(file)
        } else {
            None
        };
        let expected = options
            .digest
            .as_deref()
            .ok_or("resumable batch requires --sha256")?;
        let mut model =
            BoundedLocalReader::open(&options.model, 65536, 131072).map_err(|e| e.to_string())?;
        let model_sha = model
            .verify_sha256_cancellable(expected, || cancelled.load(Ordering::Acquire))
            .map_err(|e| e.to_string())?;
        let (input_sha, _) = hash_file(input, cancelled)?;
        #[cfg(target_os = "linux")]
        let binary = PathBuf::from("/proc/self/exe");
        #[cfg(not(target_os = "linux"))]
        let binary = std::env::current_exe().map_err(|e| e.to_string())?;
        let (engine_sha, _) = hash_file(
            &mut File::open(binary).map_err(|e| e.to_string())?,
            cancelled,
        )?;
        let settings = format!(
            "ctx={}\ntokens={}\nline_bytes={}\n",
            options.context, options.tokens, options.line_bytes
        );
        let manifest = format!("llmbridge-batch-v1\nmodel_sha256={model_sha}\ninput_sha256={input_sha}\nengine_sha256={engine_sha}\nsettings_sha256={}\n", digest(settings.as_bytes()));
        let lock_file = if let Some(file) = existing_lock {
            if small(&options.output.join("batch.identity"))? != manifest.as_bytes() {
                return Err(
                    "batch checkpoint identity mismatch (model/input/binary/configuration)".into(),
                );
            }
            file
        } else {
            fs::create_dir(&options.output)
                .map_err(|e| format!("cannot create new output directory: {e}"))?;
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(options.output.join("batch.lock"))
                .map_err(|e| e.to_string())?;
            lock(&file)?;
            atomic_record(&options.output, "batch.identity", &manifest)?;
            file
        };
        Ok(Self {
            directory: options.output.clone(),
            identity: digest(manifest.as_bytes()),
            input_sha,
            _lock: lock_file,
        })
    }
    fn receipt(&self, id: u64, prompt: &str, hash: &str, bytes: u64) -> String {
        format!("llmbridge-job-v1\njob={id}\nidentity_sha256={}\nprompt_sha256={}\noutput_sha256={hash}\nbytes={bytes}\n", self.identity, digest(prompt.as_bytes()))
    }
    pub fn before_job(
        &self,
        id: u64,
        prompt: &str,
        cancelled: &AtomicBool,
    ) -> Result<bool, String> {
        let target = self.directory.join(format!("job-{id:08}.txt"));
        let done = self.directory.join(format!("job-{id:08}.done"));
        if regular(&done)? {
            if !regular(&target)? {
                return Err(format!("job {id}: completed output missing"));
            }
            let (hash, bytes) = hash_file(
                &mut File::open(target).map_err(|e| e.to_string())?,
                cancelled,
            )?;
            if small(&done)? != self.receipt(id, prompt, &hash, bytes).as_bytes() {
                return Err(format!("job {id}: checkpoint receipt mismatch"));
            }
            return Ok(true);
        }
        if regular(&target)? {
            return Err(format!(
                "job {id}: unrecorded final output; refusing overwrite"
            ));
        }
        if regular(&self.directory.join(format!("job-{id:08}.done.partial")))? {
            return Err(format!(
                "job {id}: incomplete completion record; review before retry"
            ));
        }
        let partial = self.directory.join(format!("job-{id:08}.partial"));
        if regular(&partial)? {
            let mut attempt = 1u64;
            loop {
                let archive = self
                    .directory
                    .join(format!("job-{id:08}.retry-{attempt}.partial"));
                match fs::hard_link(&partial, archive) {
                    Ok(()) => {
                        fs::remove_file(&partial).map_err(|e| e.to_string())?;
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        attempt = attempt
                            .checked_add(1)
                            .ok_or("retry archive counter overflow")?;
                    }
                    Err(error) => return Err(error.to_string()),
                }
            }
        }
        Ok(false)
    }
    pub fn result_record(
        &self,
        id: u64,
        prompt: &str,
        output: &mut File,
        cancelled: &AtomicBool,
    ) -> Result<String, String> {
        output.sync_all().map_err(|e| e.to_string())?;
        let (hash, bytes) = hash_file(output, cancelled)?;
        Ok(self.receipt(id, prompt, &hash, bytes))
    }
    pub fn complete(&self, id: u64, receipt: &str) -> Result<(), String> {
        atomic_record(&self.directory, &format!("job-{id:08}.done"), receipt)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    fn fixture() -> (PathBuf, Options, File) {
        let root = std::env::temp_dir().join(format!(
            "llmbridge-checkpoint-unit-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let model = root.join("model");
        let input = root.join("input");
        fs::write(&model, b"fixture-model").unwrap();
        fs::write(&input, b"fixture\r\n").unwrap();
        let args = vec![
            model.display().to_string(),
            "--input".into(),
            input.display().to_string(),
            "--output-dir".into(),
            root.join("results").display().to_string(),
            "--sha256".into(),
            digest(b"fixture-model"),
            "--resumable".into(),
        ];
        (
            root,
            super::super::options(&args).unwrap(),
            File::open(input).unwrap(),
        )
    }
    fn complete_fixture(journal: &Journal, flag: &AtomicBool) {
        let partial = journal.directory.join("job-00000001.partial");
        let mut output = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&partial)
            .unwrap();
        output.write_all(b"fixture output").unwrap();
        let record = journal
            .result_record(1, "fixture", &mut output, flag)
            .unwrap();
        drop(output);
        fs::hard_link(&partial, journal.directory.join("job-00000001.txt")).unwrap();
        fs::remove_file(partial).unwrap();
        journal.complete(1, &record).unwrap();
    }
    #[test]
    fn receipt_checks_bytes_prompt_and_job_identity() {
        let (root, options, mut input) = fixture();
        let flag = AtomicBool::new(false);
        let journal = Journal::open(&options, &mut input, &flag).unwrap();
        complete_fixture(&journal, &flag);
        assert!(journal.before_job(1, "fixture", &flag).unwrap());
        assert!(journal
            .before_job(1, "different", &flag)
            .unwrap_err()
            .contains("mismatch"));
        fs::copy(
            journal.directory.join("job-00000001.txt"),
            journal.directory.join("job-00000002.txt"),
        )
        .unwrap();
        fs::copy(
            journal.directory.join("job-00000001.done"),
            journal.directory.join("job-00000002.done"),
        )
        .unwrap();
        assert!(journal
            .before_job(2, "fixture", &flag)
            .unwrap_err()
            .contains("mismatch"));
        fs::write(
            journal.directory.join("job-00000001.txt"),
            b"corrupt output",
        )
        .unwrap();
        assert!(journal
            .before_job(1, "fixture", &flag)
            .unwrap_err()
            .contains("mismatch"));
        drop(journal);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn unfinished_partial_is_archived_and_run_lock_is_exclusive() {
        let (root, mut options, mut input) = fixture();
        let flag = AtomicBool::new(false);
        let journal = Journal::open(&options, &mut input, &flag).unwrap();
        let partial = journal.directory.join("job-00000001.partial");
        fs::write(&partial, b"unfinished").unwrap();
        assert!(!journal.before_job(1, "fixture", &flag).unwrap());
        assert!(!partial.exists());
        assert_eq!(
            fs::read(journal.directory.join("job-00000001.retry-1.partial")).unwrap(),
            b"unfinished"
        );
        options.resume = true;
        options.resumable = false;
        assert!(Journal::open(&options, &mut input, &flag)
            .err()
            .unwrap()
            .contains("locked"));
        drop(journal);
        let reopened = Journal::open(&options, &mut input, &flag).unwrap();
        drop(reopened);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn oversized_and_orphaned_records_refuse_without_overwrite() {
        let (root, options, mut input) = fixture();
        let flag = AtomicBool::new(false);
        let journal = Journal::open(&options, &mut input, &flag).unwrap();
        let output = journal.directory.join("job-00000001.txt");
        fs::write(&output, b"unrecorded").unwrap();
        assert!(journal
            .before_job(1, "fixture", &flag)
            .unwrap_err()
            .contains("unrecorded"));
        assert_eq!(fs::read(output).unwrap(), b"unrecorded");
        fs::write(
            journal.directory.join("job-00000001.done"),
            vec![b'x'; 2048],
        )
        .unwrap();
        assert!(journal
            .before_job(1, "fixture", &flag)
            .unwrap_err()
            .contains("format limit"));
        drop(journal);
        fs::remove_dir_all(root).unwrap();
    }
}
