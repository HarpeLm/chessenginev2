//! Mode analyse : tu joues les coups des deux camps et le moteur analyse en
//! continu la position affichée pour indiquer le meilleur coup.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

use crate::app::App;
use crate::board::{square_name, Board};
use crate::game::{color_code, position_status, think_payload};
use crate::movegen::{legal_moves, parse_uci_move, to_san, Move};
use crate::search::SearchLimits;

/// Durée maximale d'analyse d'une position (pour ne pas chauffer le Mac pour rien).
const MAX_ANALYSIS: Duration = Duration::from_secs(30);

pub struct Analysis {
    /// Change à chaque modification : les résultats d'une analyse devenue
    /// inutile (position quittée) sont ignorés.
    pub id: u64,
    /// `positions[0]` est la position de départ ; `moves[i]` mène de
    /// `positions[i]` à `positions[i + 1]`.
    positions: Vec<Board>,
    moves: Vec<Move>,
    sans: Vec<String>,
    /// Position affichée (on peut revenir en arrière dans la partie).
    index: usize,
    pub analyzing: bool,
}

impl Analysis {
    pub fn new(start: Board) -> Analysis {
        Analysis {
            id: 0,
            positions: vec![start],
            moves: Vec::new(),
            sans: Vec::new(),
            index: 0,
            analyzing: false,
        }
    }

    fn board(&self) -> &Board {
        &self.positions[self.index]
    }

    fn status(&self) -> (&'static str, Option<String>) {
        position_status(&self.positions[..=self.index])
    }

    pub fn view(&self) -> Value {
        let board = self.board();
        let (status, result) = self.status();
        let legal: Vec<String> = if status == "playing" {
            legal_moves(board).into_iter().map(|m| m.to_uci()).collect()
        } else {
            Vec::new()
        };
        let start = &self.positions[0];
        json!({
            "id": self.id,
            "index": self.index,
            "fen": board.to_fen(),
            "turn": color_code(board.side_to_move),
            "legal": legal,
            "last_move": (self.index > 0).then(|| {
                let m = self.moves[self.index - 1];
                [square_name(m.from), square_name(m.to)]
            }),
            "check": board
                .in_check()
                .then(|| square_name(board.king_square[board.side_to_move.index()])),
            "sans": self.sans,
            "ucis": self.moves.iter().map(|m| m.to_uci()).collect::<Vec<_>>(),
            "start_fen": start.to_fen(),
            "start_fullmove": start.fullmove_number,
            "start_turn": color_code(start.side_to_move),
            "status": status,
            "result": result,
            "analyzing": self.analyzing,
        })
    }
}

fn emit_state(app: &App) {
    let view = app.analysis.lock().unwrap().view();
    app.emit(json!({ "type": "analysis_state", "analysis": view }));
}

/// Relance l'analyse sur la position affichée (et arrête la précédente).
fn restart(app: &Arc<App>) {
    let (board, history, id, playing) = {
        let mut analysis = app.analysis.lock().unwrap();
        let playing = analysis.status().0 == "playing";
        analysis.analyzing = playing;
        let history: Vec<u64> = analysis.positions[..=analysis.index]
            .iter()
            .map(|b| b.hash)
            .collect();
        (*analysis.board(), history, analysis.id, playing)
    };
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut current = app.analysis_stop.lock().unwrap();
        current.store(true, Ordering::Relaxed);
        *current = stop.clone();
    }
    emit_state(app);
    if !playing {
        return;
    }

    let app = app.clone();
    thread::spawn(move || {
        let mut searcher = app.analysis_engine.lock().unwrap();
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let limits = SearchLimits {
            soft_time: Some(MAX_ANALYSIS),
            hard_time: Some(MAX_ANALYSIS),
            ..SearchLimits::infinite()
        };
        searcher.search(
            &board,
            &history,
            app.champion(),
            limits,
            stop,
            &mut |info| {
                if app.analysis.lock().unwrap().id != id {
                    return;
                }
                app.emit(json!({
                    "type": "analysis_info",
                    "id": id,
                    "fen": board.to_fen(),
                    "info": think_payload(&board, info),
                }));
            },
        );
        drop(searcher);
        let finished = {
            let mut analysis = app.analysis.lock().unwrap();
            let current = analysis.id == id;
            if current {
                analysis.analyzing = false;
            }
            current
        };
        if finished {
            emit_state(&app);
        }
    });
}

/// Joue un coup (pour le camp au trait, quel qu'il soit). Si on était revenu en
/// arrière et qu'on joue un autre coup, la suite de la partie est remplacée.
pub fn play(app: &Arc<App>, uci: &str) -> Result<Value, String> {
    {
        let mut analysis = app.analysis.lock().unwrap();
        if analysis.status().0 != "playing" {
            return Err("la partie est terminée dans cette position".into());
        }
        let board = *analysis.board();
        let mv = parse_uci_move(&board, uci).ok_or(format!("coup illégal : {uci}"))?;
        let index = analysis.index;
        if analysis.moves.get(index) == Some(&mv) {
            // Même coup que la suite déjà jouée : on avance simplement.
            analysis.index += 1;
        } else {
            analysis.positions.truncate(index + 1);
            analysis.moves.truncate(index);
            analysis.sans.truncate(index);
            analysis.sans.push(to_san(&board, mv));
            analysis.moves.push(mv);
            analysis.positions.push(board.make_move(mv));
            analysis.index += 1;
        }
        analysis.id += 1;
    }
    restart(app);
    Ok(app.analysis.lock().unwrap().view())
}

/// Se place sur la position après `index` coups.
pub fn goto(app: &Arc<App>, index: usize) -> Value {
    {
        let mut analysis = app.analysis.lock().unwrap();
        analysis.index = index.min(analysis.moves.len());
        analysis.id += 1;
    }
    restart(app);
    app.analysis.lock().unwrap().view()
}

/// Repart de la position initiale, ou d'une position donnée en FEN.
pub fn reset(app: &Arc<App>, fen: Option<&str>) -> Result<Value, String> {
    let start = match fen.map(str::trim).filter(|f| !f.is_empty()) {
        Some(fen) => Board::from_fen(fen)?,
        None => Board::start_position(),
    };
    {
        let mut analysis = app.analysis.lock().unwrap();
        let id = analysis.id + 1;
        *analysis = Analysis::new(start);
        analysis.id = id;
    }
    restart(app);
    Ok(app.analysis.lock().unwrap().view())
}

/// Arrête l'analyse (quand on quitte l'onglet, pour ne pas occuper le processeur).
pub fn stop(app: &Arc<App>) -> Value {
    app.analysis_stop
        .lock()
        .unwrap()
        .store(true, Ordering::Relaxed);
    {
        let mut analysis = app.analysis.lock().unwrap();
        analysis.analyzing = false;
        analysis.id += 1;
    }
    emit_state(app);
    app.analysis.lock().unwrap().view()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playing_after_going_back_replaces_the_line() {
        let mut analysis = Analysis::new(Board::start_position());
        // On simule e4 e5 puis retour au début et d4.
        for uci in ["e2e4", "e7e5"] {
            let board = *analysis.board();
            let mv = parse_uci_move(&board, uci).unwrap();
            analysis.sans.push(to_san(&board, mv));
            analysis.moves.push(mv);
            analysis.positions.push(board.make_move(mv));
            analysis.index += 1;
        }
        assert_eq!(analysis.view()["turn"], "w");
        analysis.index = 0;
        assert_eq!(analysis.view()["legal"].as_array().unwrap().len(), 20);
        assert!(analysis.view()["last_move"].is_null());
    }
}
