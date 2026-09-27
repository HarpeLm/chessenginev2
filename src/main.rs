mod app;
mod board;
mod eval;
mod game;
mod movegen;
mod perft;
mod search;
mod server;
mod train;
mod tt;
mod uci;

use std::path::PathBuf;
use std::sync::Arc;

use board::{Board, START_FEN};
use eval::Weights;

const DATA_DIR: &str = "data";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("uci") => {
            let path = PathBuf::from(DATA_DIR).join("weights.json");
            let weights = Weights::load(&path.to_string_lossy()).unwrap_or_else(Weights::classic);
            uci::run(Arc::new(weights));
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

fn print_usage() {
    println!("chessengine, un moteur d'échecs en Rust");
    println!();
    println!("  cargo run --release                 interface web (http://localhost:8080)");
    println!("  cargo run --release -- --port 8081  interface web sur un autre port");
    println!("  cargo run --release -- uci          mode UCI (pour Cute Chess, Arena…)");
    println!("  cargo run --release -- perft 6      test de la génération des coups");
}
