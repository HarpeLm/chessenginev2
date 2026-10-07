//! La partie jouée dans le navigateur contre le moteur.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

use crate::app::App;
use crate::board::{square_name, Board, Color};
use crate::movegen::{legal_moves, parse_uci_move, to_san, Move};
use crate::search::{mate_in, SearchInfo, SearchLimits};

pub struct Game {
    /// Change à chaque modification : permet d'ignorer le résultat d'une
    /// réflexion devenue inutile (nouvelle partie, coup annulé…).
    pub id: u64,
    positions: Vec<Board>,
    moves: Vec<Move>,
    sans: Vec<String>,
    pub human: Color,
    pub think_ms: u64,
    pub thinking: bool,
    engine_info: Option<Value>,
}

impl Game {
    pub fn new(human: Color, think_ms: u64) -> Game {
        Game {
            id: 0,
            positions: vec![Board::start_position()],
            moves: Vec::new(),
            sans: Vec::new(),
            human,
            think_ms,
            thinking: false,
            engine_info: None,
        }
    }

    pub fn board(&self) -> &Board {
        self.positions
            .last()
            .expect("il y a toujours au moins une position")
    }

    fn history(&self) -> Vec<u64> {
        self.positions.iter().map(|b| b.hash).collect()
    }

    fn play(&mut self, mv: Move) {
        let board = *self.board();
        self.sans.push(to_san(&board, mv));
        self.moves.push(mv);
        self.positions.push(board.make_move(mv));
        self.id += 1;
    }

    /// État de la partie : code et phrase à afficher.
    fn status(&self) -> (&'static str, Option<String>) {
        position_status(&self.positions)
    }

    fn is_over(&self) -> bool {
        self.status().0 != "playing"
    }

    pub fn view(&self) -> Value {
        let board = self.board();
        let (status, result) = self.status();
        let human_turn = board.side_to_move == self.human && status == "playing" && !self.thinking;
        let legal: Vec<String> = if human_turn {
            legal_moves(board).into_iter().map(|m| m.to_uci()).collect()
        } else {
            Vec::new()
        };
        let check = board
            .in_check()
            .then(|| square_name(board.king_square[board.side_to_move.index()]));
        json!({
            "id": self.id,
            "fen": board.to_fen(),
            "turn": color_code(board.side_to_move),
            "human": color_code(self.human),
            "legal": legal,
            "last_move": self.moves.last().map(|m| [square_name(m.from), square_name(m.to)]),
            "sans": self.sans,
            "status": status,
            "result": result,
            "check": check,
            "thinking": self.thinking,
            "think_ms": self.think_ms,
            "engine": self.engine_info,
        })
    }
}

/// État de la dernière position d'une suite de positions : partie en cours,
/// mat, pat ou nulle (50 coups, triple répétition, matériel insuffisant).
pub fn position_status(positions: &[Board]) -> (&'static str, Option<String>) {
    let board = positions.last().expect("au moins une position");
    if legal_moves(board).is_empty() {
        if board.in_check() {
            let winner = if board.side_to_move == Color::White {
                "les Noirs"
            } else {
                "les Blancs"
            };
            return (
                "checkmate",
                Some(format!("Échec et mat. Victoire pour {winner}.")),
            );
        }
        return ("stalemate", Some("Pat. Partie nulle.".into()));
    }
    if board.halfmove_clock >= 100 {
        return (
            "draw",
            Some("Nulle par la règle des cinquante coups.".into()),
        );
    }
    if positions.iter().filter(|b| b.hash == board.hash).count() >= 3 {
        return ("draw", Some("Nulle par triple répétition.".into()));
    }
    if board.is_insufficient_material() {
        return ("draw", Some("Nulle, matériel insuffisant.".into()));
    }
    ("playing", None)
}

pub fn color_code(color: Color) -> &'static str {
    match color {
        Color::White => "w",
        Color::Black => "b",
    }
}

/// Résumé de la réflexion du moteur pour l'affichage (score du point de vue
/// des Blancs, ligne principale en notation SAN et UCI).
pub fn think_payload(board: &Board, info: &SearchInfo) -> Value {
    let sign = if board.side_to_move == Color::White {
        1
    } else {
        -1
    };
    let mut position = *board;
    let mut pv = Vec::new();
    for &mv in &info.pv {
        pv.push(to_san(&position, mv));
        position = position.make_move(mv);
    }
    let ms = info.elapsed.as_millis().max(1) as u64;
    json!({
        "depth": info.depth,
        "cp": info.score * sign,
        "mate": mate_in(info.score).map(|m| m * sign),
        "nodes": info.nodes,
        "nps": info.nodes * 1000 / ms,
        "time_ms": ms,
        "pv": pv,
        "pv_uci": info.pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>(),
        "pv_start": board.fullmove_number,
        "pv_turn": color_code(board.side_to_move),
    })
}

fn stop_engine(app: &App, game: &mut Game) {
    app.engine_stop
        .lock()
        .unwrap()
        .store(true, Ordering::Relaxed);
    game.thinking = false;
    game.id += 1;
}

fn emit_game(app: &App) {
    let view = app.game.lock().unwrap().view();
    app.emit(json!({ "type": "game", "game": view }));
}

/// Si c'est au moteur de jouer, lance la réflexion dans un fil séparé.
pub fn maybe_start_engine(app: &Arc<App>) {
    let (board, history, id, think_ms) = {
        let mut game = app.game.lock().unwrap();
        if game.thinking || game.is_over() || game.board().side_to_move == game.human {
            return;
        }
        game.thinking = true;
        game.engine_info = None;
        (*game.board(), game.history(), game.id, game.think_ms)
    };
    let stop = Arc::new(AtomicBool::new(false));
    *app.engine_stop.lock().unwrap() = stop.clone();
    emit_game(app);

    let app = app.clone();
    thread::spawn(move || {
        let mut searcher = app.engine.lock().unwrap();
        let limits = SearchLimits::time(Duration::from_millis(think_ms));
        let result = searcher.search(
            &board,
            &history,
            app.champion(),
            limits,
            stop,
            &mut |info| {
                let payload = think_payload(&board, info);
                {
                    let mut game = app.game.lock().unwrap();
                    if game.id != id {
                        return;
                    }
                    game.engine_info = Some(payload.clone());
                }
                app.emit(json!({ "type": "think", "id": id, "info": payload }));
            },
        );
        drop(searcher);

        {
            let mut game = app.game.lock().unwrap();
            if game.id != id {
                return;
            }
            game.thinking = false;
            if let Some(mv) = result.best_move {
                game.play(mv);
            }
        }
        emit_game(&app);
    });
}

pub fn new_game(app: &Arc<App>, human: &str, think_ms: u64) -> Value {
    let human = match human {
        "white" => Color::White,
        "black" => Color::Black,
        _ => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            if nanos % 2 == 0 {
                Color::White
            } else {
                Color::Black
            }
        }
    };
    {
        let mut game = app.game.lock().unwrap();
        stop_engine(app, &mut game);
        let id = game.id + 1;
        *game = Game::new(human, think_ms.clamp(100, 60_000));
        game.id = id;
    }
    maybe_start_engine(app);
    app.game.lock().unwrap().view()
}

pub fn human_move(app: &Arc<App>, uci: &str) -> Result<Value, String> {
    {
        let mut game = app.game.lock().unwrap();
        if game.thinking {
            return Err("le moteur est en train de réfléchir".into());
        }
        if game.is_over() {
            return Err("la partie est terminée".into());
        }
        if game.board().side_to_move != game.human {
            return Err("ce n'est pas ton tour".into());
        }
        let mv = parse_uci_move(game.board(), uci).ok_or(format!("coup illégal : {uci}"))?;
        game.play(mv);
    }
    maybe_start_engine(app);
    Ok(app.game.lock().unwrap().view())
}

/// Reprend le dernier coup du joueur (et la réponse du moteur).
pub fn undo(app: &Arc<App>) -> Value {
    {
        let mut game = app.game.lock().unwrap();
        stop_engine(app, &mut game);
        while !game.moves.is_empty() {
            game.moves.pop();
            game.sans.pop();
            game.positions.pop();
            if game.board().side_to_move == game.human {
                break;
            }
        }
        game.engine_info = None;
    }
    maybe_start_engine(app);
    app.game.lock().unwrap().view()
}

pub fn set_think_time(app: &Arc<App>, think_ms: u64) -> Value {
    let mut game = app.game.lock().unwrap();
    game.think_ms = think_ms.clamp(100, 60_000);
    game.view()
}
