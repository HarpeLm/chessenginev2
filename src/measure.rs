//! Mesure du niveau : ton moteur affronte Stockfish bridé à un niveau connu
//! (option UCI_Elo). Le score obtenu donne une estimation de son classement,
//! avec une marge d'erreur qui se resserre au fil des parties.
//!
//! L'échelle d'UCI_Elo est celle des listes de moteurs (CCRL), pas le
//! classement FIDE des joueurs humains.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::app::App;
use crate::board::{Board, Color};
use crate::eval::Evaluator;
use crate::external::{find_stockfish, UciEngine};
use crate::movegen::{legal_moves, parse_uci_move};
use crate::search::{clock_limits, Searcher, MOVE_OVERHEAD_MS};
use crate::train::Wdl;

/// Ouvertures équilibrées, jouées chacune deux fois (une fois de chaque couleur).
pub const OPENINGS: [&str; 24] = [
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",
    "e2e4 e7e5 b1c3 g8f6 f1c4 b8c6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4",
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6",
    "d2d4 d7d5 c1f4 g8f6 e2e3 c7c5",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
];

const MAX_PLIES: usize = 400;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeasureRun {
    /// Secondes depuis 1970.
    pub started_at: u64,
    pub opponent: String,
    pub opponent_elo: u32,
    /// Notre moteur : « formule » ou « réseau », et la génération d'entraînement.
    pub engine: String,
    pub games: usize,
    pub base_ms: u64,
    pub increment_ms: u64,
    /// Du point de vue de notre moteur.
    pub wdl: Wdl,
    pub estimate: Option<f64>,
    pub margin: Option<f64>,
    pub finished: bool,
    /// Parties perdues au temps : [par ton moteur, par Stockfish].
    #[serde(default)]
    pub time_losses: [u32; 2],
    /// Estimation après chaque partie : (nombre de parties, estimation, marge).
    pub points: Vec<(usize, f64, f64)>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MeasureState {
    #[serde(skip_deserializing)]
    pub running: bool,
    #[serde(skip_deserializing)]
    pub stockfish: Option<StockfishInfo>,
    #[serde(skip_deserializing)]
    pub error: Option<String>,
    /// Enregistrée après chaque partie : si l'application est fermée pendant
    /// une mesure, les parties déjà jouées ne sont pas perdues.
    #[serde(default)]
    pub current: Option<MeasureRun>,
    pub path: Option<String>,
    pub history: Vec<MeasureRun>,
}

impl MeasureState {
    /// Relit les mesures enregistrées. Une mesure interrompue (application
    /// fermée en cours de route) rejoint l'historique avec ses parties jouées.
    pub fn load(path: &std::path::Path) -> MeasureState {
        let mut state: MeasureState = std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        if let Some(run) = state.current.take() {
            let played = run.wdl.wins + run.wdl.draws + run.wdl.losses;
            let known = state.history.iter().any(|h| h.started_at == run.started_at);
            if played > 0 && !known {
                state.history.push(run);
            }
        }
        state
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StockfishInfo {
    pub path: String,
    pub name: String,
    pub limit_strength: bool,
}

pub struct MeasureConfig {
    pub opponent_elo: u32,
    pub games: usize,
    pub base_ms: u64,
    pub increment_ms: u64,
}

/// Cherche Stockfish et vérifie qu'il accepte UCI_Elo.
pub fn detect(app: &App, preferred: Option<String>) {
    let preferred = preferred.or_else(|| app.measure.lock().unwrap().path.clone());
    let found = find_stockfish(preferred.as_deref());
    let (info, error) = match found {
        None => (
            None,
            Some("Stockfish introuvable. Sur Mac : brew install stockfish".to_string()),
        ),
        Some(path) => match UciEngine::start(&path) {
            Ok(engine) => (
                Some(StockfishInfo {
                    path: path.display().to_string(),
                    name: engine.name.clone(),
                    limit_strength: engine.has_option("UCI_LimitStrength")
                        && engine.has_option("UCI_Elo"),
                }),
                None,
            ),
            Err(e) => (None, Some(e)),
        },
    };
    {
        let mut state = app.measure.lock().unwrap();
        state.stockfish = info;
        state.error = error;
        if preferred.is_some() {
            state.path = preferred;
        }
    }
    app.save_measures();
    app.emit_measure();
}

pub fn start(app: Arc<App>, config: MeasureConfig) -> Result<(), String> {
    let path = {
        let mut state = app.measure.lock().unwrap();
        if state.running {
            return Err("une mesure est déjà en cours".into());
        }
        let info = state
            .stockfish
            .clone()
            .ok_or("Stockfish n'est pas installé (ou pas trouvé)")?;
        if !info.limit_strength {
            return Err(format!(
                "{} ne permet pas de régler son niveau (UCI_Elo)",
                info.name
            ));
        }
        let engine = {
            let training = app.training.lock().unwrap();
            let kind = match &*app.champion() {
                Evaluator::Classic(_) => "formule",
                Evaluator::Nnue(_) => "réseau",
            };
            format!("{kind} · génération {}", training.generation)
        };
        state.running = true;
        state.error = None;
        state.current = Some(MeasureRun {
            started_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            opponent: info.name.clone(),
            opponent_elo: config.opponent_elo,
            engine,
            games: config.games,
            base_ms: config.base_ms,
            increment_ms: config.increment_ms,
            wdl: Wdl::default(),
            estimate: None,
            margin: None,
            finished: false,
            time_losses: [0, 0],
            points: Vec::new(),
        });
        PathBuf::from(info.path)
    };
    let stop = Arc::new(AtomicBool::new(false));
    *app.measure_stop.lock().unwrap() = stop.clone();
    app.emit_measure();
    thread::spawn(move || run(app, config, path, stop));
    Ok(())
}

pub fn stop(app: &App) {
    app.measure_stop
        .lock()
        .unwrap()
        .store(true, Ordering::Relaxed);
}

fn run(app: Arc<App>, config: MeasureConfig, path: PathBuf, stop: Arc<AtomicBool>) {
    // Chaque partie occupe deux cœurs : un pour nous, un pour Stockfish.
    let cores = thread::available_parallelism().map_or(2, |n| n.get());
    let workers = (cores / 2).clamp(1, config.games.max(1));
    let next = AtomicUsize::new(0);
    let champion = app.champion();
    let failure: Mutex<Option<String>> = Mutex::new(None);

    thread::scope(|scope| {
        for worker in 0..workers {
            let (app, config, path, stop, next, champion, failure) =
                (&app, &config, &path, &stop, &next, &champion, &failure);
            scope.spawn(move || {
                let mut opponent = match prepare_opponent(path, config.opponent_elo) {
                    Ok(engine) => engine,
                    Err(e) => {
                        *failure.lock().unwrap() = Some(e);
                        stop.store(true, Ordering::Relaxed);
                        return;
                    }
                };
                let mut searcher = Searcher::new(32);
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= config.games || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let our_color = if index % 2 == 0 {
                        Color::White
                    } else {
                        Color::Black
                    };
                    let opening = OPENINGS[(index / 2) % OPENINGS.len()];
                    let outcome = play_game(
                        app,
                        &mut searcher,
                        &mut opponent,
                        champion,
                        opening,
                        our_color,
                        config,
                        stop,
                        worker == 0,
                        index,
                    );
                    match outcome {
                        Ok(Some((score, lost_on_time))) => {
                            record(app, config.opponent_elo, score, lost_on_time)
                        }
                        Ok(None) => break,
                        Err(e) => {
                            *failure.lock().unwrap() = Some(e);
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            });
        }
    });

    {
        let mut state = app.measure.lock().unwrap();
        state.running = false;
        state.error = failure.into_inner().unwrap();
        if let Some(mut run) = state.current.take() {
            let played = run.wdl.wins + run.wdl.draws + run.wdl.losses;
            run.finished = played as usize >= run.games;
            if played > 0 {
                state.history.push(run.clone());
            }
            state.current = Some(run);
        }
    }
    app.save_measures();
    app.emit_measure();
}

fn prepare_opponent(path: &PathBuf, elo: u32) -> Result<UciEngine, String> {
    let mut engine = UciEngine::start(path)?;
    engine.set_option("Threads", "1")?;
    engine.set_option("Hash", "16")?;
    engine.set_option("UCI_LimitStrength", "true")?;
    engine.set_option("UCI_Elo", &elo.to_string())?;
    engine.ready()?;
    Ok(engine)
}

/// Ajoute le résultat d'une partie et recalcule l'estimation.
fn record(app: &App, opponent_elo: u32, score: f32, lost_on_time: Option<bool>) {
    {
        let mut state = app.measure.lock().unwrap();
        let Some(run) = state.current.as_mut() else {
            return;
        };
        run.wdl.add(score);
        match lost_on_time {
            Some(true) => run.time_losses[0] += 1,
            Some(false) => run.time_losses[1] += 1,
            None => {}
        }
        if let Some((estimate, margin)) = estimate(&run.wdl, opponent_elo) {
            let played = (run.wdl.wins + run.wdl.draws + run.wdl.losses) as usize;
            run.estimate = Some(estimate);
            run.margin = Some(margin);
            run.points.push((played, estimate, margin));
        }
    }
    app.save_measures();
    app.emit_measure();
}

/// Estimation d'ELO et marge d'erreur (intervalle de confiance à 95 %).
///
/// On ajoute une nulle « virtuelle » : sans elle, un score parfait (100 % ou
/// 0 %) donnerait une estimation infinie et une marge nulle, ce qui n'a pas
/// de sens. Avec beaucoup de parties, elle ne change presque rien.
pub fn estimate(wdl: &Wdl, opponent_elo: u32) -> Option<(f64, f64)> {
    if wdl.wins + wdl.draws + wdl.losses == 0 {
        return None;
    }
    let (wins, draws, losses) = (wdl.wins as f64, wdl.draws as f64 + 1.0, wdl.losses as f64);
    let n = wins + draws + losses;
    let score = (wins + 0.5 * draws) / n;
    let variance =
        (wins * (1.0 - score).powi(2) + draws * (0.5 - score).powi(2) + losses * score.powi(2)) / n;
    let standard_error = (variance / n).sqrt();
    let to_elo = |s: f64| {
        let s = s.clamp(0.01, 0.99);
        -400.0 * (1.0 / s - 1.0).log10()
    };
    let low = to_elo(score - 1.96 * standard_error);
    let high = to_elo(score + 1.96 * standard_error);
    Some((opponent_elo as f64 + to_elo(score), (high - low) / 2.0))
}

/// Joue une partie contre Stockfish. Renvoie notre score (1, ½ ou 0) et, en
/// cas de chute du drapeau, qui a perdu au temps (true : notre moteur) ;
/// None si la mesure a été arrêtée, ou une erreur si Stockfish a planté.
#[allow(clippy::too_many_arguments)]
fn play_game(
    app: &App,
    searcher: &mut Searcher,
    opponent: &mut UciEngine,
    champion: &Arc<Evaluator>,
    opening: &str,
    our_color: Color,
    config: &MeasureConfig,
    stop: &Arc<AtomicBool>,
    show: bool,
    index: usize,
) -> Result<Option<(f32, Option<bool>)>, String> {
    searcher.clear();
    opponent.new_game()?;
    let mut board = Board::start_position();
    let mut history = vec![board.hash];
    let mut moves: Vec<String> = Vec::new();
    for text in opening.split_whitespace() {
        let mv = parse_uci_move(&board, text).ok_or(format!("ouverture illégale : {text}"))?;
        board = board.make_move(mv);
        history.push(board.hash);
        moves.push(text.to_string());
    }
    let mut clocks = [config.base_ms as i64, config.base_ms as i64];
    let names = if our_color == Color::White {
        ["Ton moteur".to_string(), opponent.name.clone()]
    } else {
        [opponent.name.clone(), "Ton moteur".to_string()]
    };

    let mut lost_on_time = None;
    let white_score = loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        if show {
            app.emit(json!({
                "type": "measure_live",
                "fen": board.to_fen(),
                "last_move": moves.last(),
                "white": names[0],
                "black": names[1],
                "clocks": clocks,
                "game": index + 1,
                "ply": moves.len(),
            }));
        }
        if legal_moves(&board).is_empty() {
            break if !board.in_check() {
                0.5
            } else if board.side_to_move == Color::White {
                0.0
            } else {
                1.0
            };
        }
        let repeated = history.iter().filter(|&&h| h == board.hash).count() >= 3;
        if board.halfmove_clock >= 100
            || repeated
            || board.is_insufficient_material()
            || moves.len() >= MAX_PLIES
        {
            break 0.5;
        }

        let side = board.side_to_move.index();
        let started = Instant::now();
        let text = if board.side_to_move == our_color {
            let limits = clock_limits(
                clocks[side].max(0) as u64,
                config.increment_ms,
                None,
                MOVE_OVERHEAD_MS,
            );
            let result = searcher.search(
                &board,
                &history,
                champion.clone(),
                limits,
                stop.clone(),
                &mut |_| {},
            );
            match result.best_move {
                Some(mv) => mv.to_uci(),
                None => return Ok(None),
            }
        } else {
            let clock = |c: i64| c.max(1) as u64;
            opponent.best_move(
                &moves,
                [clock(clocks[0]), clock(clocks[1])],
                config.increment_ms,
            )?
        };
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        clocks[side] -= started.elapsed().as_millis() as i64;
        if clocks[side] < 0 {
            // Tombé au temps.
            lost_on_time = Some(board.side_to_move == our_color);
            break if side == 0 { 0.0 } else { 1.0 };
        }
        clocks[side] += config.increment_ms as i64;

        let mv = parse_uci_move(&board, &text).ok_or(format!("coup illégal reçu : {text}"))?;
        board = board.make_move(mv);
        history.push(board.hash);
        moves.push(text);
    };

    if show {
        app.emit(json!({
            "type": "measure_live",
            "fen": board.to_fen(),
            "last_move": moves.last(),
            "white": names[0],
            "black": names[1],
            "clocks": clocks,
            "game": index + 1,
            "ply": moves.len(),
        }));
    }
    let score = if our_color == Color::White {
        white_score
    } else {
        1.0 - white_score
    };
    Ok(Some((score, lost_on_time)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openings_are_legal() {
        for line in OPENINGS {
            let mut board = Board::start_position();
            for text in line.split_whitespace() {
                let mv = parse_uci_move(&board, text).unwrap_or_else(|| panic!("{line} : {text}"));
                board = board.make_move(mv);
            }
        }
    }

    #[test]
    fn interrupted_measure_is_kept() {
        let run = MeasureRun {
            started_at: 42,
            opponent: "Stockfish".into(),
            opponent_elo: 2000,
            engine: "formule".into(),
            games: 200,
            base_ms: 30_000,
            increment_ms: 300,
            wdl: Wdl {
                wins: 5,
                draws: 1,
                losses: 2,
            },
            estimate: Some(2100.0),
            margin: Some(150.0),
            finished: false,
            time_losses: [0, 0],
            points: Vec::new(),
        };
        let state = MeasureState {
            current: Some(run),
            ..MeasureState::default()
        };
        let path = std::env::temp_dir().join("chessengine-measures-test.json");
        std::fs::write(&path, serde_json::to_string(&state).unwrap()).unwrap();
        let loaded = MeasureState::load(&path);
        assert!(loaded.current.is_none());
        assert_eq!(loaded.history.len(), 1);
        assert_eq!(loaded.history[0].wdl.wins, 5);
        // Relue une seconde fois (déjà dans l'historique) : pas de doublon.
        let mut again = loaded.clone();
        again.current = Some(loaded.history[0].clone());
        std::fs::write(&path, serde_json::to_string(&again).unwrap()).unwrap();
        assert_eq!(MeasureState::load(&path).history.len(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn estimate_is_centered_on_even_score() {
        let wdl = Wdl {
            wins: 10,
            draws: 10,
            losses: 10,
        };
        let (elo, margin) = estimate(&wdl, 2000).unwrap();
        assert!((elo - 2000.0).abs() < 1e-6);
        assert!(margin > 50.0 && margin < 200.0);
        let wdl = Wdl {
            wins: 30,
            draws: 10,
            losses: 10,
        };
        assert!(estimate(&wdl, 2000).unwrap().0 > 2100.0);
        // Un score parfait donne une estimation finie, avec une vraie marge.
        let (elo, margin) = estimate(
            &Wdl {
                wins: 4,
                draws: 0,
                losses: 0,
            },
            1600,
        )
        .unwrap();
        assert!(elo > 1700.0 && elo < 2400.0 && margin > 50.0);
    }
}
