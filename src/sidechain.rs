//! Side-chain packing integration using the external `dASPR` binary.

use std::env;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::bundle::Bundle;
use crate::geometry::Chain;
use crate::input::Sequence;
use crate::pdb::{bundle_to_pdb_string, to_pdb_string};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Configuration for invoking the external `dASPR` side-chain packer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DasprConfig {
    /// Explicit path to the `dASPR` executable (falls back to `DASPR_BIN` env, `PATH`, or `~/dASPR/dASPR`).
    pub daspr_bin: Option<PathBuf>,
    /// Optional path to `dun2010_mirror.bin` passed via `-r`.
    pub rotlib_path: Option<PathBuf>,
}

/// Errors produced during side-chain packing with `dASPR`.
#[derive(Debug)]
pub enum SidechainError {
    ChainCountMismatch { chains: usize, sequences: usize },
    ResidueCountMismatch { chain_id: char, residues: usize, sequence_len: usize },
    EmptyStructure,
    Io(io::Error),
    DasprNotFound(String),
    DasprFailed(String),
}

impl From<io::Error> for SidechainError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl fmt::Display for SidechainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChainCountMismatch { chains, sequences } => write!(
                f,
                "bundle has {chains} chains, but {sequences} sequences were provided"
            ),
            Self::ResidueCountMismatch {
                chain_id,
                residues,
                sequence_len,
            } => write!(
                f,
                "chain {chain_id} has {residues} residues, but sequence has {sequence_len}"
            ),
            Self::EmptyStructure => write!(f, "cannot pack an empty structure"),
            Self::Io(err) => write!(f, "I/O error during side-chain packing: {err}"),
            Self::DasprNotFound(msg) => write!(f, "dASPR executable not found: {msg}"),
            Self::DasprFailed(msg) => write!(f, "dASPR failed: {msg}"),
        }
    }
}

impl std::error::Error for SidechainError {}

struct TempPathGuard {
    path: PathBuf,
}

impl TempPathGuard {
    fn new(prefix: &str, ext: &str) -> Self {
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        let path = env::temp_dir().join(format!("{prefix}_{pid}_{id}.{ext}"));
        Self { path }
    }
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Resolves the `dASPR` executable path from config, `DASPR_BIN`, `PATH`, or `$HOME/dASPR/dASPR`.
pub fn resolve_daspr_binary(config: &DasprConfig) -> Result<PathBuf, SidechainError> {
    if let Some(bin) = &config.daspr_bin {
        if bin.is_file() {
            return Ok(bin.clone());
        }
        return Err(SidechainError::DasprNotFound(format!(
            "configured path '{}' does not exist",
            bin.display()
        )));
    }
    if let Ok(env_bin) = env::var("DASPR_BIN") {
        let p = PathBuf::from(&env_bin);
        if p.is_file() {
            return Ok(p);
        }
        return Err(SidechainError::DasprNotFound(format!(
            "DASPR_BIN='{env_bin}' does not exist"
        )));
    }
    if let Ok(path_var) = env::var("PATH") {
        for dir in env::split_paths(&path_var) {
            let candidate = dir.join("dASPR");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    if let Ok(home) = env::var("HOME") {
        let candidate = Path::new(&home).join("dASPR").join("dASPR");
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(SidechainError::DasprNotFound(
        "install dASPR on PATH, set DASPR_BIN, or pass --daspr-bin".to_string(),
    ))
}

/// Normalizes `dASPR` PDB output with continuous atom serial numbers and a trailing `END` record.
pub fn normalize_daspr_pdb(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 16);
    let mut serial = 1usize;
    let mut has_end = false;

    for line in raw.lines() {
        if line.starts_with("ATOM  ") && line.len() >= 11 {
            out.push_str(&format!("ATOM  {:>5}{}\n", serial, &line[11..]));
            serial += 1;
        } else if line.starts_with("TER") && line.len() >= 11 {
            out.push_str(&format!("TER   {:>5}{}\n", serial, &line[11..]));
            serial += 1;
        } else if line.starts_with("END") {
            has_end = true;
            out.push_str(line);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !has_end {
        out.push_str("END\n");
    }
    out
}

/// Runs `dASPR` on a backbone PDB string and a concatenated case-encoded target sequence.
pub fn pack_pdb_with_daspr(
    backbone_pdb: &str,
    concatenated_seq: &str,
    config: &DasprConfig,
) -> Result<String, SidechainError> {
    if !backbone_pdb.contains("ATOM  ") || concatenated_seq.is_empty() {
        return Err(SidechainError::EmptyStructure);
    }
    let bin = resolve_daspr_binary(config)?;

    let in_guard = TempPathGuard::new("bhb_daspr_in", "pdb");
    let seq_guard = TempPathGuard::new("bhb_daspr_seq", "fasta");
    let out_guard = TempPathGuard::new("bhb_daspr_out", "pdb");

    fs::write(&in_guard.path, backbone_pdb)?;
    fs::write(&seq_guard.path, format!(">target\n{concatenated_seq}\n"))?;

    let mut cmd = Command::new(&bin);
    cmd.arg("-i")
        .arg(&in_guard.path)
        .arg("-o")
        .arg(&out_guard.path)
        .arg("-s")
        .arg(&seq_guard.path);

    if let Some(rotlib) = &config.rotlib_path {
        cmd.arg("-r").arg(rotlib);
    }

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // dASPR prints "error!" and calls exit(0) on several failure paths, so validate output file too.
    if !output.status.success() || stdout.contains("error!") || stderr.contains("error!") {
        return Err(SidechainError::DasprFailed(format!(
            "exit={}; stdout={}; stderr={}",
            output.status,
            stdout.trim(),
            stderr.trim()
        )));
    }

    let packed_raw = fs::read_to_string(&out_guard.path).map_err(|e| {
        SidechainError::DasprFailed(format!("failed to read dASPR output PDB: {e}"))
    })?;
    if !packed_raw.contains("ATOM  ") {
        return Err(SidechainError::DasprFailed(
            "dASPR output contained no ATOM records".to_string(),
        ));
    }

    Ok(normalize_daspr_pdb(&packed_raw))
}

/// Packs side chains onto a monomeric backbone [`Chain`] for the given target [`Sequence`].
pub fn pack_chain(
    chain: &Chain,
    sequence: &Sequence,
    config: &DasprConfig,
) -> Result<String, SidechainError> {
    if chain.residues.len() != sequence.len() {
        return Err(SidechainError::ResidueCountMismatch {
            chain_id: chain.chain_id,
            residues: chain.residues.len(),
            sequence_len: sequence.len(),
        });
    }
    let backbone_pdb = to_pdb_string(chain);
    pack_pdb_with_daspr(&backbone_pdb, &sequence.to_daspr_string(), config)
}

/// Packs side chains onto all chains in a [`Bundle`] given one [`Sequence`] per chain.
pub fn pack_bundle(
    bundle: &Bundle,
    sequences: &[Sequence],
    config: &DasprConfig,
) -> Result<String, SidechainError> {
    if bundle.chains.len() != sequences.len() {
        return Err(SidechainError::ChainCountMismatch {
            chains: bundle.chains.len(),
            sequences: sequences.len(),
        });
    }
    let mut concatenated = String::new();
    for (chain, seq) in bundle.chains.iter().zip(sequences.iter()) {
        if chain.residues.len() != seq.len() {
            return Err(SidechainError::ResidueCountMismatch {
                chain_id: chain.chain_id,
                residues: chain.residues.len(),
                sequence_len: seq.len(),
            });
        }
        concatenated.push_str(&seq.to_daspr_string());
    }
    let backbone_pdb = bundle_to_pdb_string(bundle);
    pack_pdb_with_daspr(&backbone_pdb, &concatenated, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::build_beta_helix;
    use crate::input::HelixParams;

    #[test]
    fn normalizes_daspr_serial_numbers_and_appends_end() {
        let sample = "\
REMARK repacked structure by dASPR
ATOM      1  N   VAL A   1       1.000   2.000   3.000  1.00  0.00           N  
ATOM      2  CA  VAL A   1       2.000   2.000   3.000  1.00  0.00           C  
ATOM      1  N   GLY A   2       3.000   2.000   3.000  1.00  0.00           N  
TER       2      GLY A   2
ATOM      1  N   ALA B   1       4.000   2.000   3.000  1.00  0.00           N  
TER       2      ALA B   1
";
        let norm = normalize_daspr_pdb(sample);
        let lines: Vec<_> = norm.lines().collect();
        assert!(lines[1].starts_with("ATOM      1 "));
        assert!(lines[2].starts_with("ATOM      2 "));
        assert!(lines[3].starts_with("ATOM      3 "));
        assert!(lines[4].starts_with("TER       4 "));
        assert!(lines[5].starts_with("ATOM      5 "));
        assert!(lines[6].starts_with("TER       6 "));
        assert_eq!(lines.last(), Some(&"END"));
    }

    #[test]
    fn errors_cleanly_when_daspr_binary_missing() {
        let seq = Sequence::parse("VGAl").unwrap();
        let bb = build_beta_helix(&seq.as_poly_ala(true), &HelixParams::beta_6_3(), 'A').unwrap();
        let cfg = DasprConfig {
            daspr_bin: Some(PathBuf::from("/nonexistent/path/to/dASPR")),
            rotlib_path: None,
        };
        let err = pack_chain(&bb, &seq, &cfg).unwrap_err();
        assert!(matches!(err, SidechainError::DasprNotFound(_)));
    }
}
