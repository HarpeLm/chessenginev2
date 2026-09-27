//! Piloter un autre moteur d'échecs (Stockfish, par exemple) avec le
//! protocole UCI : on le lance comme un programme séparé et on lui parle par
//! son entrée et sa sortie standard.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

pub struct UciEngine {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<String>,
    pub name: String,
    options: Vec<String>,
}

impl UciEngine {
    pub fn start(path: &Path) -> Result<UciEngine, String> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("impossible de lancer {} : {e}", path.display()))?;
        let stdin = child.stdin.take().ok_or("pas d'entrée standard")?;
        let stdout = child.stdout.take().ok_or("pas de sortie standard")?;
        // Un fil lit les réponses en continu ; on les reçoit avec un délai maximal.
        let (sender, lines) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        let mut engine = UciEngine {
            child,
            stdin,
            lines,
            name: String::from("moteur inconnu"),
            options: Vec::new(),
        };
        engine.send("uci")?;
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let line = engine.next_line(deadline)?;
            if let Some(name) = line.strip_prefix("id name ") {
                engine.name = name.trim().to_string();
            } else if let Some(option) = line.strip_prefix("option name ") {
                let name = option.split(" type ").next().unwrap_or("").trim();
                engine.options.push(name.to_string());
            } else if line.trim() == "uciok" {
                break;
            }
        }
        Ok(engine)
    }

    pub fn has_option(&self, name: &str) -> bool {
        self.options.iter().any(|o| o == name)
    }

    fn send(&mut self, command: &str) -> Result<(), String> {
        writeln!(self.stdin, "{command}")
            .and_then(|_| self.stdin.flush())
            .map_err(|e| format!("le moteur ne répond plus : {e}"))
    }

    fn next_line(&self, deadline: Instant) -> Result<String, String> {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match self.lines.recv_timeout(remaining) {
            Ok(line) => Ok(line),
            Err(RecvTimeoutError::Timeout) => Err("le moteur ne répond pas à temps".into()),
            Err(RecvTimeoutError::Disconnected) => Err("le moteur s'est arrêté".into()),
        }
    }

    pub fn set_option(&mut self, name: &str, value: &str) -> Result<(), String> {
        self.send(&format!("setoption name {name} value {value}"))
    }

    /// Attend que le moteur ait fini de traiter les commandes précédentes.
    pub fn ready(&mut self) -> Result<(), String> {
        self.send("isready")?;
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.next_line(deadline)?.trim() != "readyok" {}
        Ok(())
    }

    pub fn new_game(&mut self) -> Result<(), String> {
        self.send("ucinewgame")?;
        self.ready()
    }

    /// Demande un coup (notation UCI) pour la position obtenue en jouant
    /// `moves` depuis la position de départ, avec les temps de pendule donnés
    /// (en millisecondes).
    pub fn best_move(
        &mut self,
        moves: &[String],
        clocks: [u64; 2],
        increment: u64,
    ) -> Result<String, String> {
        let position = if moves.is_empty() {
            "position startpos".to_string()
        } else {
            format!("position startpos moves {}", moves.join(" "))
        };
        self.send(&position)?;
        self.send(&format!(
            "go wtime {} btime {} winc {increment} binc {increment}",
            clocks[0], clocks[1]
        ))?;
        // On laisse au moteur son temps de réflexion, plus une marge.
        let deadline = Instant::now() + Duration::from_millis(clocks[0].max(clocks[1]) + 5_000);
        loop {
            let line = self.next_line(deadline)?;
            if let Some(rest) = line.strip_prefix("bestmove") {
                return rest
                    .split_whitespace()
                    .next()
                    .map(str::to_string)
                    .ok_or_else(|| "réponse « bestmove » vide".to_string());
            }
        }
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send("quit");
        thread::sleep(Duration::from_millis(20));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Cherche Stockfish aux endroits habituels (dont Homebrew sur Mac).
pub fn find_stockfish(preferred: Option<&str>) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = preferred.filter(|p| !p.trim().is_empty()) {
        candidates.push(PathBuf::from(path.trim()));
    }
    if let Some(paths) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&paths).map(|dir| dir.join("stockfish")));
    }
    for fixed in [
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/games/stockfish",
        "/usr/bin/stockfish",
    ] {
        candidates.push(PathBuf::from(fixed));
    }
    candidates.into_iter().find(|path| path.is_file())
}
