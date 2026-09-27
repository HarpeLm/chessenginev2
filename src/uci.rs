//! Protocole UCI : le langage standard des moteurs d'échecs. Permet de
//! brancher le moteur sur Cute Chess, Arena, Banksia ou lichess-bot.
//! Lancement : `cargo run --release -- uci`

use std::io::{self, BufRead};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::board::{Board, Color};
use crate::eval::{evaluate, Weights};
use crate::movegen::parse_uci_move;
use crate::search::{mate_in, SearchLimits, Searcher, MAX_DEPTH};

/// Marge de sécurité pour ne jamais perdre au temps (communication avec l'interface).
const MOVE_OVERHEAD_MS: u64 = 30;

pub fn run(weights: Arc<Weights>) {
    let mut board = Board::start_position();
    let mut history = vec![board.hash];
    let mut searcher = Some(Searcher::new(64));
    let mut handle: Option<JoinHandle<Searcher>> = None;
    let mut stop = Arc::new(AtomicBool::new(false));

    for line in io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        let Some(&command) = tokens.first() else {
            continue;
        };

        match command {
            "uci" => {
                println!("id name chessengine {}", env!("CARGO_PKG_VERSION"));
                println!("id author HarpeLm");
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                wait(&mut searcher, &mut handle);
                searcher.as_mut().unwrap().clear();
                board = Board::start_position();
                history = vec![board.hash];
            }
            "setoption" => {
                wait(&mut searcher, &mut handle);
                let name = tokens
                    .iter()
                    .position(|&t| t == "name")
                    .and_then(|i| tokens.get(i + 1));
                let value = tokens
                    .iter()
                    .position(|&t| t == "value")
                    .and_then(|i| tokens.get(i + 1));
                if let (Some(&"Hash"), Some(value)) = (name, value) {
                    if let Ok(mb) = value.parse() {
                        searcher.as_mut().unwrap().resize_hash(mb);
                    }
                }
            }
            "position" => {
                wait(&mut searcher, &mut handle);
                match parse_position(&tokens) {
                    Ok((b, h)) => {
                        board = b;
                        history = h;
                    }
                    Err(message) => println!("info string {message}"),
                }
            }
            "go" => {
                wait(&mut searcher, &mut handle);
                let (limits, infinite) = parse_go(&tokens, &board);
                stop = Arc::new(AtomicBool::new(false));
                let mut s = searcher.take().unwrap();
                let (b, h, flag, w) = (board, history.clone(), stop.clone(), weights.clone());
                handle = Some(thread::spawn(move || {
                    let result = s.search(&b, &h, w, limits, flag.clone(), &mut |info| {
                        let score = match mate_in(info.score) {
                            Some(m) => format!("mate {m}"),
                            None => format!("cp {}", info.score),
                        };
                        let ms = info.elapsed.as_millis().max(1) as u64;
                        let pv: Vec<String> = info.pv.iter().map(|m| m.to_uci()).collect();
                        println!(
                            "info depth {} score {} nodes {} nps {} time {} pv {}",
                            info.depth,
                            score,
                            info.nodes,
                            info.nodes * 1000 / ms,
                            ms,
                            pv.join(" ")
                        );
                    });
                    // En mode infini, UCI impose d'attendre « stop » avant de répondre.
                    while infinite && !flag.load(Ordering::Relaxed) {
                        thread::sleep(Duration::from_millis(2));
                    }
                    match result.best_move {
                        Some(mv) => println!("bestmove {}", mv.to_uci()),
                        None => println!("bestmove 0000"),
                    }
                    s
                }));
            }
            "stop" => {
                stop.store(true, Ordering::Relaxed);
                wait(&mut searcher, &mut handle);
            }
            "quit" => {
                stop.store(true, Ordering::Relaxed);
                wait(&mut searcher, &mut handle);
                return;
            }
            "d" => board.print(),
            "eval" => println!("{} cp (camp au trait)", evaluate(&board, &weights)),
            _ => println!("info string commande inconnue : {line}"),
        }
    }
    stop.store(true, Ordering::Relaxed);
    wait(&mut searcher, &mut handle);
}

fn wait(searcher: &mut Option<Searcher>, handle: &mut Option<JoinHandle<Searcher>>) {
    if let Some(h) = handle.take() {
        *searcher = Some(h.join().expect("le fil de recherche a planté"));
    }
}

fn parse_position(tokens: &[&str]) -> Result<(Board, Vec<u64>), String> {
    let moves_index = tokens
        .iter()
        .position(|&t| t == "moves")
        .unwrap_or(tokens.len());
    let mut board = match tokens.get(1) {
        Some(&"startpos") => Board::start_position(),
        Some(&"fen") => Board::from_fen(&tokens[2..moves_index].join(" "))?,
        _ => return Err("position : il faut « startpos » ou « fen »".into()),
    };
    let mut history = vec![board.hash];
    for text in tokens.iter().skip(moves_index + 1) {
        let mv = parse_uci_move(&board, text).ok_or(format!("coup illégal : {text}"))?;
        board = board.make_move(mv);
        history.push(board.hash);
    }
    Ok((board, history))
}

fn parse_go(tokens: &[&str], board: &Board) -> (SearchLimits, bool) {
    let value = |name: &str| -> Option<i64> {
        let i = tokens.iter().position(|&t| t == name)?;
        tokens.get(i + 1)?.parse().ok()
    };
    let mut limits = SearchLimits::infinite();
    if let Some(depth) = value("depth") {
        limits.max_depth = (depth as i32).clamp(1, MAX_DEPTH);
    }
    if let Some(nodes) = value("nodes") {
        limits.max_nodes = Some(nodes.max(1) as u64);
    }
    if tokens.contains(&"infinite") {
        return (limits, true);
    }
    if let Some(ms) = value("movetime") {
        let ms = (ms.max(1) as u64).saturating_sub(MOVE_OVERHEAD_MS).max(1);
        limits.soft_time = Some(Duration::from_millis(ms));
        limits.hard_time = Some(Duration::from_millis(ms));
        return (limits, false);
    }
    let (time, increment) = match board.side_to_move {
        Color::White => (value("wtime"), value("winc")),
        Color::Black => (value("btime"), value("binc")),
    };
    if let Some(time) = time {
        let time = time.max(0) as u64;
        let increment = increment.unwrap_or(0).max(0) as u64;
        let moves_to_go = value("movestogo").unwrap_or(30).max(1) as u64;
        let available = time.saturating_sub(MOVE_OVERHEAD_MS);
        let soft = (available / moves_to_go + increment * 3 / 4).min(available / 2);
        let hard = (soft * 3).min(available * 3 / 4).max(soft);
        limits.soft_time = Some(Duration::from_millis(soft));
        limits.hard_time = Some(Duration::from_millis(hard));
        return (limits, false);
    }
    // « go » tout seul : réflexion infinie.
    let infinite = value("depth").is_none() && value("nodes").is_none();
    (limits, infinite)
}
