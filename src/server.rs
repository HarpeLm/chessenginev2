//! Serveur web local : sert l'interface et une petite API JSON.
//! Les mises à jour en direct (réflexion du moteur, entraînement) passent
//! par des « Server-Sent Events » sur /api/events.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};

use crate::app::App;
use crate::game;
use crate::train::{self, TrainConfig};

type Shared = State<Arc<App>>;

const INDEX_HTML: &str = include_str!("../web/index.html");
const STYLE_CSS: &str = include_str!("../web/style.css");
const APP_JS: &str = include_str!("../web/app.js");

const PIECES: [(&str, &str); 12] = [
    ("wK.svg", include_str!("../web/pieces/wK.svg")),
    ("wQ.svg", include_str!("../web/pieces/wQ.svg")),
    ("wR.svg", include_str!("../web/pieces/wR.svg")),
    ("wB.svg", include_str!("../web/pieces/wB.svg")),
    ("wN.svg", include_str!("../web/pieces/wN.svg")),
    ("wP.svg", include_str!("../web/pieces/wP.svg")),
    ("bK.svg", include_str!("../web/pieces/bK.svg")),
    ("bQ.svg", include_str!("../web/pieces/bQ.svg")),
    ("bR.svg", include_str!("../web/pieces/bR.svg")),
    ("bB.svg", include_str!("../web/pieces/bB.svg")),
    ("bN.svg", include_str!("../web/pieces/bN.svg")),
    ("bP.svg", include_str!("../web/pieces/bP.svg")),
];

pub async fn serve(app: Arc<App>, port: u16, open_browser: bool) {
    let router = Router::new()
        .route("/", get(|| async { Html(INDEX_HTML) }))
        .route(
            "/style.css",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
                    STYLE_CSS,
                )
            }),
        )
        .route(
            "/app.js",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
                    APP_JS,
                )
            }),
        )
        .route("/pieces/{name}", get(piece))
        .route("/api/events", get(events))
        .route("/api/game", get(game_state))
        .route("/api/game/new", post(new_game))
        .route("/api/game/move", post(play_move))
        .route("/api/game/undo", post(undo))
        .route("/api/game/settings", post(settings))
        .route("/api/train", get(train_state))
        .route("/api/train/start", post(train_start))
        .route("/api/train/stop", post(train_stop))
        .route("/api/train/reset", post(train_reset))
        .with_state(app);

    let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("Impossible d'écouter sur le port {port} : {err}");
            eprintln!("Essaie un autre port : cargo run --release -- --port 8081");
            std::process::exit(1);
        }
    };
    let url = format!("http://localhost:{port}");
    println!("Interface prête : {url}");
    println!("(Ctrl+C pour arrêter)");
    if open_browser {
        open_in_browser(&url);
    }
    axum::serve(listener, router)
        .await
        .expect("le serveur s'est arrêté");
}

fn open_in_browser(url: &str) {
    let command = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = std::process::Command::new(command).arg(url).spawn();
}

fn error(message: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response()
}

async fn piece(Path(name): Path<String>) -> Response {
    match PIECES.iter().find(|(file, _)| *file == name) {
        Some((_, svg)) => (
            [
                (header::CONTENT_TYPE, "image/svg+xml"),
                (header::CACHE_CONTROL, "max-age=86400"),
            ],
            *svg,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn events(State(app): Shared) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(app.subscribe())
        .filter_map(|message| message.ok().map(|data| Ok(Event::default().data(data))));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn game_state(State(app): Shared) -> Json<Value> {
    Json(app.game.lock().unwrap().view())
}

#[derive(Deserialize)]
struct NewGame {
    human: String,
    think_ms: u64,
}

async fn new_game(State(app): Shared, Json(body): Json<NewGame>) -> Json<Value> {
    Json(game::new_game(&app, &body.human, body.think_ms))
}

#[derive(Deserialize)]
struct PlayMove {
    uci: String,
}

async fn play_move(State(app): Shared, Json(body): Json<PlayMove>) -> Response {
    match game::human_move(&app, &body.uci) {
        Ok(view) => Json(view).into_response(),
        Err(message) => error(message),
    }
}

async fn undo(State(app): Shared) -> Json<Value> {
    Json(game::undo(&app))
}

#[derive(Deserialize)]
struct Settings {
    think_ms: u64,
}

async fn settings(State(app): Shared, Json(body): Json<Settings>) -> Json<Value> {
    Json(game::set_think_time(&app, body.think_ms))
}

async fn train_state(State(app): Shared) -> Json<Value> {
    use crate::eval::*;
    Json(json!({
        "state": app.training_snapshot(),
        "champion": &*app.champion(),
        // Position de chaque critère dans les vecteurs de poids.
        "layout": {
            "material": MATERIAL,
            "pst": PST,
            "passed_pawn": PASSED_PAWN,
            "doubled_pawn": DOUBLED_PAWN,
            "isolated_pawn": ISOLATED_PAWN,
            "rook_open_file": ROOK_OPEN_FILE,
            "rook_semi_open_file": ROOK_SEMI_OPEN_FILE,
            "bishop_pair": BISHOP_PAIR,
            "mobility": MOBILITY,
            "king_shield": KING_SHIELD,
            "tempo": TEMPO,
        },
    }))
}

#[derive(Deserialize)]
struct TrainStart {
    games_per_generation: usize,
    nodes_per_move: u64,
}

async fn train_start(State(app): Shared, Json(body): Json<TrainStart>) -> Response {
    let config = TrainConfig {
        games_per_generation: body.games_per_generation.clamp(8, 2_000),
        nodes_per_move: body.nodes_per_move.clamp(500, 200_000),
        ..TrainConfig::default()
    };
    match train::start(app.clone(), config) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(message) => error(message),
    }
}

async fn train_stop(State(app): Shared) -> Json<Value> {
    train::stop(&app);
    Json(json!({ "ok": true }))
}

#[derive(Deserialize)]
struct TrainReset {
    origin: String,
}

async fn train_reset(State(app): Shared, Json(body): Json<TrainReset>) -> Response {
    match app.reset_training(&body.origin) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(message) => error(message),
    }
}
