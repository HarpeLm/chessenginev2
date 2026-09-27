//! État partagé de l'application web : la partie en cours, le moteur,
//! l'entraînement et le canal d'événements vers le navigateur.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::eval::Weights;
use crate::game::Game;
use crate::search::Searcher;
use crate::train::{Sample, TrainState};

pub struct App {
    events: broadcast::Sender<String>,
    champion: RwLock<Arc<Weights>>,
    pub game: Mutex<Game>,
    pub engine: Mutex<Searcher>,
    pub engine_stop: Mutex<Arc<AtomicBool>>,
    pub training: Mutex<TrainState>,
    pub train_stop: Mutex<Arc<AtomicBool>>,
    pub dataset: Mutex<Vec<Sample>>,
    data_dir: PathBuf,
}

impl App {
    pub fn new(data_dir: PathBuf) -> App {
        let _ = std::fs::create_dir_all(&data_dir);
        let weights_path = data_dir.join("weights.json");
        let champion =
            Weights::load(&weights_path.to_string_lossy()).unwrap_or_else(Weights::classic);
        let training = std::fs::read_to_string(data_dir.join("training.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<TrainState>(&text).ok())
            .map(|mut state| {
                state.phase = "idle".into();
                state
            })
            .unwrap_or_else(|| TrainState::new("classic"));

        let (events, _) = broadcast::channel(4096);
        App {
            events,
            champion: RwLock::new(Arc::new(champion)),
            game: Mutex::new(Game::new(crate::board::Color::White, 2000)),
            engine: Mutex::new(Searcher::new(64)),
            engine_stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            training: Mutex::new(training),
            train_stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            dataset: Mutex::new(Vec::new()),
            data_dir,
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<String> {
        self.events.subscribe()
    }

    /// Envoie un événement à tous les navigateurs connectés.
    pub fn emit(&self, event: Value) {
        let _ = self.events.send(event.to_string());
    }

    /// Les poids d'évaluation du meilleur moteur connu.
    pub fn champion(&self) -> Arc<Weights> {
        self.champion.read().unwrap().clone()
    }

    pub fn set_champion(&self, weights: Arc<Weights>) {
        let _ = weights.save(&self.data_dir.join("weights.json").to_string_lossy());
        *self.champion.write().unwrap() = weights;
    }

    pub fn training_snapshot(&self) -> TrainState {
        self.training.lock().unwrap().clone()
    }

    pub fn emit_training(&self) {
        self.emit(json!({ "type": "train_state", "state": self.training_snapshot() }));
    }

    /// Version légère de `emit_training`, envoyée après chaque partie.
    pub fn emit_progress(&self) {
        let state = self.training.lock().unwrap();
        self.emit(json!({
            "type": "train_progress",
            "phase": state.phase,
            "progress": state.progress,
            "selfplay": state.selfplay,
            "matchup": state.matchup,
        }));
    }

    pub fn save_training(&self) {
        let state = self.training_snapshot();
        if let Ok(text) = serde_json::to_string(&state) {
            let _ = std::fs::write(self.data_dir.join("training.json"), text);
        }
    }

    /// Repart de zéro (ou des valeurs classiques) : efface l'historique d'apprentissage.
    pub fn reset_training(&self, origin: &str) -> Result<(), String> {
        let weights = match origin {
            "classic" => Weights::classic(),
            "zero" => Weights::zero(),
            other => return Err(format!("point de départ inconnu : {other}")),
        };
        {
            let mut state = self.training.lock().unwrap();
            if state.running {
                return Err("arrête d'abord l'entraînement".into());
            }
            let config = state.config;
            *state = TrainState::new(origin);
            state.config = config;
        }
        self.dataset.lock().unwrap().clear();
        self.set_champion(Arc::new(weights));
        self.save_training();
        self.emit_training();
        self.emit(
            json!({ "type": "train_weights", "champion": &*self.champion(), "candidate": null }),
        );
        Ok(())
    }
}
