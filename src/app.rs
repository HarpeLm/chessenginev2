//! État partagé de l'application web : la partie en cours, le moteur,
//! l'entraînement et le canal d'événements vers le navigateur.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{json, Value};
use tokio::sync::broadcast;

use crate::eval::{Evaluator, Weights};
use crate::game::Game;
use crate::nnue::{Dataset, Network};
use crate::search::Searcher;
use crate::train::{Sample, TrainState};

/// Le champion enregistré : le réseau de neurones si on est en mode réseau et
/// qu'il existe, sinon la formule classique.
pub fn load_champion(data_dir: &Path) -> Evaluator {
    let mode = std::fs::read_to_string(data_dir.join("training.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<TrainState>(&text).ok())
        .map(|state| state.mode);
    if mode.as_deref() == Some("nnue") {
        if let Some(network) = Network::load(&data_dir.join("nnue.bin").to_string_lossy()) {
            return Evaluator::Nnue(network);
        }
    }
    let weights = Weights::load(&data_dir.join("weights.json").to_string_lossy());
    Evaluator::Classic(weights.unwrap_or_else(Weights::classic))
}

pub struct App {
    events: broadcast::Sender<String>,
    champion: RwLock<Arc<Evaluator>>,
    pub game: Mutex<Game>,
    pub engine: Mutex<Searcher>,
    pub engine_stop: Mutex<Arc<AtomicBool>>,
    pub training: Mutex<TrainState>,
    pub train_stop: Mutex<Arc<AtomicBool>>,
    /// Le dernier candidat entraîné (pour l'affichage).
    pub last_candidate: Mutex<Option<Arc<Evaluator>>>,
    /// Positions pour la formule classique (méthode de Texel).
    pub dataset: Mutex<Vec<Sample>>,
    /// Positions pour le réseau de neurones.
    pub nnue_dataset: Mutex<Dataset>,
    data_dir: PathBuf,
}

impl App {
    pub fn new(data_dir: PathBuf) -> App {
        let _ = std::fs::create_dir_all(&data_dir);
        let training = std::fs::read_to_string(data_dir.join("training.json"))
            .ok()
            .and_then(|text| serde_json::from_str::<TrainState>(&text).ok())
            .map(|mut state| {
                state.phase = "idle".into();
                state
            })
            .unwrap_or_else(|| TrainState::new("classic"));
        let champion = load_champion(&data_dir);

        let positions =
            Dataset::load(&data_dir.join("positions.bin").to_string_lossy()).unwrap_or_default();
        let mut training = training;
        if training.mode == "nnue" {
            training.dataset = positions.len();
        }
        let (events, _) = broadcast::channel(4096);
        App {
            events,
            champion: RwLock::new(Arc::new(champion)),
            game: Mutex::new(Game::new(crate::board::Color::White, 2000)),
            engine: Mutex::new(Searcher::new(64)),
            engine_stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            training: Mutex::new(training),
            train_stop: Mutex::new(Arc::new(AtomicBool::new(false))),
            last_candidate: Mutex::new(None),
            dataset: Mutex::new(Vec::new()),
            nnue_dataset: Mutex::new(positions),
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

    /// L'évaluation du meilleur moteur connu.
    pub fn champion(&self) -> Arc<Evaluator> {
        self.champion.read().unwrap().clone()
    }

    pub fn set_champion(&self, evaluator: Arc<Evaluator>) {
        let _ = match &*evaluator {
            Evaluator::Classic(weights) => {
                weights.save(&self.data_dir.join("weights.json").to_string_lossy())
            }
            Evaluator::Nnue(network) => {
                network.save(&self.data_dir.join("nnue.bin").to_string_lossy())
            }
        };
        *self.champion.write().unwrap() = evaluator;
    }

    pub fn emit_weights(&self, candidate: Option<&Evaluator>) {
        self.emit(json!({
            "type": "train_weights",
            "champion": self.champion().view(),
            "candidate": candidate.map(Evaluator::view),
        }));
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
            "verification": state.verification,
            "dataset": state.dataset,
        }));
    }

    /// Enregistre les positions du réseau (elles représentent des heures de parties).
    pub fn save_positions(&self) {
        let data = self.nnue_dataset.lock().unwrap();
        let path = self.data_dir.join("positions.bin");
        if data.len() == 0 {
            let _ = std::fs::remove_file(path);
        } else {
            let _ = data.save(&path.to_string_lossy());
        }
    }

    pub fn save_training(&self) {
        let state = self.training_snapshot();
        if let Ok(text) = serde_json::to_string(&state) {
            let _ = std::fs::write(self.data_dir.join("training.json"), text);
        }
    }

    /// Change de point de départ :
    /// - « classic » ou « zero » : efface l'apprentissage et repart de la
    ///   formule classique (valeurs écrites à la main, ou tout à zéro) ;
    /// - « nnue » : garde l'historique et le champion actuel (la formule
    ///   classique) ; un réseau de neurones va apprendre à le battre.
    pub fn reset_training(&self, origin: &str) -> Result<(), String> {
        let classic_path = self.data_dir.join("weights.json");
        let weights = match origin {
            "classic" => Weights::classic(),
            "zero" => Weights::zero(),
            "nnue" => {
                Weights::load(&classic_path.to_string_lossy()).unwrap_or_else(Weights::classic)
            }
            other => return Err(format!("point de départ inconnu : {other}")),
        };
        {
            let mut state = self.training.lock().unwrap();
            if state.running {
                return Err("arrête d'abord l'entraînement".into());
            }
            if origin == "nnue" {
                state.mode = "nnue".into();
                state.dataset = 0;
            } else {
                let config = state.config;
                *state = TrainState::new(origin);
                state.config = config;
            }
        }
        self.dataset.lock().unwrap().clear();
        self.nnue_dataset.lock().unwrap().clear();
        *self.last_candidate.lock().unwrap() = None;
        self.save_positions();
        let _ = std::fs::remove_file(self.data_dir.join("nnue.bin"));
        self.set_champion(Arc::new(Evaluator::Classic(weights)));
        self.save_training();
        self.emit_training();
        self.emit_weights(None);
        Ok(())
    }
}
