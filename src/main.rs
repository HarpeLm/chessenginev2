mod analysis;
mod app;
mod board;
mod eval;
mod external;
mod game;
mod measure;
mod movegen;
mod nnue;
mod perft;
mod search;
mod server;
mod train;
mod tt;
mod uci;

use std::path::PathBuf;
use std::sync::Arc;

use board::{Board, START_FEN};

const DATA_DIR: &str = "data";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("uci") => {
            let champion = app::load_champion(&PathBuf::from(DATA_DIR));
            uci::run(Arc::new(champion));
        }
        Some("perft") => {
            let depth = args.get(1).and_then(|d| d.parse().ok()).unwrap_or(5);
            let fen = if args.len() > 2 {
                args[2..].join(" ")
            } else {
                START_FEN.to_string()
            };
            match Board::from_fen(&fen) {
                Ok(board) => perft::run(&board, depth),
                Err(message) => eprintln!("FEN invalide : {message}"),
            }
        }
        Some("bench") => bench(),
        Some("-h") | Some("--help") | Some("aide") => print_usage(),
        _ => {
            let port = args
                .iter()
                .position(|a| a == "--port")
                .and_then(|i| args.get(i + 1))
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080);
            let open_browser = !args.iter().any(|a| a == "--no-open");
            let app = Arc::new(app::App::new(PathBuf::from(DATA_DIR)));
            tokio::runtime::Runtime::new()
                .expect("démarrage de tokio")
                .block_on(server::serve(app, port, open_browser));
        }
    }
}

/// Mesure la vitesse de recherche du champion enregistré sur quelques positions.
fn bench() {
    use std::sync::atomic::AtomicBool;
    use std::time::Instant;

    let champion = Arc::new(app::load_champion(&PathBuf::from(DATA_DIR)));
    let kind = match &*champion {
        eval::Evaluator::Classic(_) => "formule classique",
        eval::Evaluator::Nnue(_) => "réseau de neurones",
    };
    let fens = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];
    let start = Instant::now();
    let mut nodes = 0;
    for fen in fens {
        let board = Board::from_fen(fen).expect("FEN valide");
        let mut searcher = search::Searcher::new(16);
        let limits = search::SearchLimits {
            max_depth: 10,
            ..search::SearchLimits::infinite()
        };
        let result = searcher.search(
            &board,
            &[],
            champion.clone(),
            limits,
            Arc::new(AtomicBool::new(false)),
            &mut |_| {},
        );
        nodes += result.nodes;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!("Évaluation : {kind}");
    println!("Positions  : {nodes}");
    println!("Temps      : {seconds:.2} s");
    println!("Vitesse    : {:.0} positions/s", nodes as f64 / seconds);
}

fn print_usage() {
    println!("chessengine, un moteur d'échecs en Rust");
    println!();
    println!("  cargo run --release                 interface web (http://localhost:8080)");
    println!("  cargo run --release -- --port 8081  interface web sur un autre port");
    println!("  cargo run --release -- uci          mode UCI (pour Cute Chess, Arena…)");
    println!("  cargo run --release -- perft 6      test de la génération des coups");
    println!("  cargo run --release -- bench        vitesse de recherche du champion");
}
