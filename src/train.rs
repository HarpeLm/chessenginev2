//! Apprentissage par parties contre soi-même.
//!
//! Chaque génération se déroule en trois temps :
//! 1. **Parties** : le champion actuel joue contre lui-même depuis des
//!    ouvertures tirées au hasard. On garde les positions calmes, étiquetées
//!    avec le résultat final de la partie (1 = gain blanc, ½ = nulle, 0 = gain noir).
//! 2. **Ajustement** : on cherche les poids d'évaluation qui prédisent le mieux
//!    ces résultats (méthode de Texel : descente de gradient sur l'erreur entre
//!    sigmoïde(évaluation) et résultat).
//! 3. **Match** : le candidat affronte le champion, chaque ouverture étant jouée
//!    avec les deux couleurs. S'il marque plus de 50 %, il devient champion.
//! 4. **Vérification** : un second match indépendant mesure le gain réel d'ELO.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::app::App;
use crate::board::{Board, Color};
use crate::eval::{self, Weights, MATERIAL, MAX_PHASE, NUM_TERMS};
use crate::movegen::{legal_moves, Move};
use crate::search::{SearchLimits, Searcher, MATE_THRESHOLD};

/// Nombre maximal de positions gardées en mémoire (les plus anciennes sont oubliées).
const MAX_DATASET: usize = 300_000;
const OPENING_PLIES: usize = 8;
const MAX_GAME_PLIES: usize = 300;
const TUNING_EPOCHS: usize = 100;
/// Rappel vers les poids du champion : empêche les poids rarement observés
/// (une case peu visitée, par exemple) de partir n'importe où.
const ANCHOR_STRENGTH: f64 = 2e-6;
/// Pente de la sigmoïde : un avantage de 100 centipions ≈ 64 % de score attendu.
const SIGMOID_SCALE: f64 = std::f64::consts::LN_10 / 400.0;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub games_per_generation: usize,
    pub nodes_per_move: u64,
    pub match_games: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            games_per_generation: 96,
            nodes_per_move: 4_000,
            match_games: 40,
        }
    }
}

/// Victoires / nulles / défaites.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Wdl {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl Wdl {
    fn add(&mut self, score: f32) {
        if score > 0.75 {
            self.wins += 1;
        } else if score < 0.25 {
            self.losses += 1;
        } else {
            self.draws += 1;
        }
    }

    fn score(&self) -> f64 {
        let games = (self.wins + self.draws + self.losses).max(1) as f64;
        (self.wins as f64 + 0.5 * self.draws as f64) / games
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationSummary {
    pub index: u32,
    pub games: usize,
    pub samples: usize,
    pub dataset: usize,
    pub loss_before: f64,
    pub loss_after: f64,
    /// Résultats des parties d'entraînement, du point de vue des Blancs.
    pub selfplay: Wdl,
    /// Résultats du match de sélection, du point de vue du candidat.
    pub matchup: Wdl,
    /// Match de vérification (seulement si le candidat a été adopté).
    pub verification: Option<Wdl>,
    /// Gain d'ELO mesuré par la vérification (0 si le candidat est rejeté).
    pub elo_gain: f64,
    pub elo: f64,
    pub accepted: bool,
    pub seconds: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainState {
    #[serde(skip_deserializing)]
    pub running: bool,
    #[serde(skip_deserializing)]
    pub phase: String,
    #[serde(skip_deserializing)]
    pub progress: f64,
    pub generation: u32,
    /// ELO du champion, relatif au point de départ (somme des gains vérifiés).
    pub elo: f64,
    pub total_games: u64,
    #[serde(skip_deserializing)]
    pub dataset: usize,
    /// Point de départ : « classic » ou « zero ».
    pub origin: String,
    pub config: TrainConfig,
    pub history: Vec<GenerationSummary>,
    #[serde(skip_deserializing)]
    pub losses: Vec<f64>,
    #[serde(skip_deserializing)]
    pub selfplay: Wdl,
    #[serde(skip_deserializing)]
    pub matchup: Wdl,
    #[serde(skip_deserializing)]
    pub verification: Wdl,
}

impl TrainState {
    pub fn new(origin: &str) -> TrainState {
        TrainState {
            running: false,
            phase: "idle".into(),
            progress: 0.0,
            generation: 0,
            elo: 0.0,
            total_games: 0,
            dataset: 0,
            origin: origin.into(),
            config: TrainConfig::default(),
            history: Vec::new(),
            losses: Vec::new(),
            selfplay: Wdl::default(),
            matchup: Wdl::default(),
            verification: Wdl::default(),
        }
    }
}

/// Une position d'entraînement : ses termes d'évaluation et le résultat de la partie.
pub struct Sample {
    features: Vec<(u16, i16)>,
    phase: u8,
    result: f32,
}

pub fn start(app: Arc<App>, config: TrainConfig) -> Result<(), String> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut state = app.training.lock().unwrap();
        if state.running {
            return Err("l'entraînement est déjà en cours".into());
        }
        state.running = true;
        state.config = config;
        state.phase = "selfplay".into();
        state.progress = 0.0;
    }
    *app.train_stop.lock().unwrap() = stop.clone();
    app.emit_training();
    thread::spawn(move || run(app, config, stop));
    Ok(())
}

pub fn stop(app: &App) {
    app.train_stop
        .lock()
        .unwrap()
        .store(true, Ordering::Relaxed);
}

fn set_phase(app: &App, phase: &str) {
    {
        let mut state = app.training.lock().unwrap();
        state.phase = phase.into();
        state.progress = 0.0;
        if phase == "selfplay" {
            state.selfplay = Wdl::default();
            state.matchup = Wdl::default();
            state.verification = Wdl::default();
            state.losses.clear();
        }
    }
    app.emit_training();
}

fn run(app: Arc<App>, config: TrainConfig, stop: Arc<AtomicBool>) {
    let threads = thread::available_parallelism().map_or(2, |n| n.get());
    let mut seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos() as u64)
        | 1;
    // Partir de zéro demande de grands changements ; affiner des valeurs
    // classiques demande de la délicatesse.
    let learning_rate = if app.training.lock().unwrap().origin == "zero" {
        1.5
    } else {
        0.5
    };

    while !stop.load(Ordering::Relaxed) {
        let started = Instant::now();
        let champion = app.champion();

        // 1. Parties du champion contre lui-même.
        set_phase(&app, "selfplay");
        seed = next_random(seed);
        let selfplay_seed = seed;
        let Some(games) = play_many(
            &app,
            "selfplay",
            config.games_per_generation,
            threads,
            &stop,
            |i| {
                (
                    champion.clone(),
                    champion.clone(),
                    selfplay_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )
            },
        ) else {
            break;
        };
        let mut selfplay = Wdl::default();
        let mut samples = Vec::new();
        for game in &games {
            selfplay.add(game.result);
            for board in &game.quiet_positions {
                let (features, phase) = eval::trace(board);
                samples.push(Sample {
                    features,
                    phase: phase as u8,
                    result: game.result,
                });
            }
        }
        let new_samples = samples.len();

        // 2. Ajustement des poids sur toutes les positions connues.
        set_phase(&app, "tuning");
        let mut dataset = app.dataset.lock().unwrap();
        dataset.extend(samples);
        if dataset.len() > MAX_DATASET {
            let excess = dataset.len() - MAX_DATASET;
            dataset.drain(..excess);
        }
        let dataset_len = dataset.len();
        app.training.lock().unwrap().dataset = dataset_len;
        let loss_before = loss_and_gradient(&dataset, &to_params(&champion), threads, false).0;
        let Some(candidate) = tune(&app, &dataset, &champion, learning_rate, threads, &stop) else {
            break;
        };
        let loss_after = loss_and_gradient(&dataset, &to_params(&candidate), threads, false).0;
        drop(dataset);
        let candidate = Arc::new(candidate);

        // 3. Match candidat contre champion : décide si le candidat est adopté.
        set_phase(&app, "match");
        seed = next_random(seed);
        let Some(matchup) = run_match(
            &app,
            "match",
            &candidate,
            &champion,
            seed,
            config.match_games,
            threads,
            &stop,
        ) else {
            break;
        };
        let accepted = matchup.score() > 0.5;

        // 4. Vérification : un second match, sur d'autres ouvertures, mesure le
        //    vrai gain. Le premier match a servi à choisir le candidat ; il
        //    surestime donc son niveau (on retient les candidats chanceux).
        let mut verification = None;
        if accepted {
            set_phase(&app, "verify");
            seed = next_random(seed);
            let Some(wdl) = run_match(
                &app,
                "verify",
                &candidate,
                &champion,
                seed,
                config.match_games,
                threads,
                &stop,
            ) else {
                break;
            };
            verification = Some(wdl);
            app.set_champion(candidate.clone());
        }

        {
            let mut state = app.training.lock().unwrap();
            state.generation += 1;
            let elo_gain = verification.map_or(0.0, |wdl| elo_difference(wdl.score()));
            state.elo += elo_gain;
            let games_played = config.games_per_generation
                + config.match_games * (1 + verification.is_some() as usize);
            state.total_games += games_played as u64;
            let summary = GenerationSummary {
                index: state.generation,
                games: config.games_per_generation,
                samples: new_samples,
                dataset: dataset_len,
                loss_before,
                loss_after,
                selfplay,
                matchup,
                verification,
                elo_gain,
                elo: state.elo,
                accepted,
                seconds: started.elapsed().as_secs_f64(),
            };
            state.history.push(summary);
        }
        app.save_training();
        app.emit_training();
        app.emit(json!({ "type": "train_weights", "champion": &*app.champion(), "candidate": &*candidate }));
    }

    {
        let mut state = app.training.lock().unwrap();
        state.running = false;
        state.phase = "idle".into();
        state.progress = 0.0;
    }
    app.emit_training();
}

/// Match aller-retour : chaque ouverture est jouée deux fois, couleurs inversées.
/// Résultats du point de vue du candidat.
#[allow(clippy::too_many_arguments)]
fn run_match(
    app: &App,
    phase: &str,
    candidate: &Arc<Weights>,
    champion: &Arc<Weights>,
    seed: u64,
    games: usize,
    threads: usize,
    stop: &Arc<AtomicBool>,
) -> Option<Wdl> {
    let records = play_many(app, phase, games, threads, stop, |i| {
        let opening = seed ^ ((i / 2) as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        if i % 2 == 0 {
            (candidate.clone(), champion.clone(), opening)
        } else {
            (champion.clone(), candidate.clone(), opening)
        }
    })?;
    let mut wdl = Wdl::default();
    for game in &records {
        wdl.add(if game.index % 2 == 0 {
            game.result
        } else {
            1.0 - game.result
        });
    }
    Some(wdl)
}

fn next_random(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Écart d'ELO correspondant à un score moyen (0,5 = même niveau).
fn elo_difference(score: f64) -> f64 {
    let score = score.clamp(0.01, 0.99);
    -400.0 * (1.0 / score - 1.0).log10()
}

struct GameRecord {
    index: usize,
    /// Score des Blancs : 1, 0,5 ou 0.
    result: f32,
    quiet_positions: Vec<Board>,
}

/// Joue `total` parties en parallèle. `setup(i)` donne les poids des Blancs,
/// des Noirs et la graine de l'ouverture de la partie `i`.
fn play_many<F>(
    app: &App,
    phase: &str,
    total: usize,
    threads: usize,
    stop: &Arc<AtomicBool>,
    setup: F,
) -> Option<Vec<GameRecord>>
where
    F: Fn(usize) -> (Arc<Weights>, Arc<Weights>, u64) + Sync,
{
    let next = AtomicUsize::new(0);
    let finished = Mutex::new(Vec::with_capacity(total));
    let nodes = app.training.lock().unwrap().config.nodes_per_move;

    thread::scope(|scope| {
        for worker in 0..threads.min(total.max(1)) {
            let (next, finished, setup) = (&next, &finished, &setup);
            scope.spawn(move || {
                let mut searchers = [Searcher::new(4), Searcher::new(4)];
                let mut last_live = Instant::now();
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= total || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let (white, black, opening) = setup(index);
                    let labels = if phase != "selfplay" {
                        if index % 2 == 0 {
                            ["Candidat", "Champion"]
                        } else {
                            ["Champion", "Candidat"]
                        }
                    } else {
                        ["Champion", "Champion"]
                    };
                    // Seul le premier fil d'exécution montre sa partie en direct.
                    let mut live = |board: &Board, last: Option<Move>, ply: usize| {
                        if worker == 0 && last_live.elapsed().as_millis() >= 140 {
                            last_live = Instant::now();
                            app.emit(json!({
                                "type": "train_live",
                                "fen": board.to_fen(),
                                "last_move": last.map(|m| m.to_uci()),
                                "white": labels[0],
                                "black": labels[1],
                                "ply": ply,
                                "game": index + 1,
                                "phase": phase,
                            }));
                        }
                    };
                    let Some((result, quiet_positions)) = play_game(
                        &white,
                        &black,
                        opening,
                        nodes,
                        &mut searchers,
                        stop,
                        &mut live,
                    ) else {
                        break;
                    };

                    let done = {
                        let mut finished = finished.lock().unwrap();
                        finished.push(GameRecord {
                            index,
                            result,
                            quiet_positions,
                        });
                        finished.len()
                    };
                    {
                        let mut state = app.training.lock().unwrap();
                        state.progress = done as f64 / total as f64;
                        let candidate_score = if index % 2 == 0 { result } else { 1.0 - result };
                        match phase {
                            "match" => state.matchup.add(candidate_score),
                            "verify" => state.verification.add(candidate_score),
                            _ => state.selfplay.add(result),
                        }
                    }
                    app.emit_progress();
                }
            });
        }
    });

    if stop.load(Ordering::Relaxed) {
        return None;
    }
    let mut games = finished.into_inner().unwrap();
    games.sort_by_key(|g| g.index);
    Some(games)
}

/// Ouverture aléatoire : quelques coups tirés au sort pour varier les parties.
fn random_opening(mut seed: u64) -> (Board, Vec<u64>) {
    loop {
        let mut board = Board::start_position();
        let mut history = vec![board.hash];
        let mut ok = true;
        for _ in 0..OPENING_PLIES {
            let moves = legal_moves(&board);
            if moves.is_empty() {
                ok = false;
                break;
            }
            seed = next_random(seed);
            board = board.make_move(moves[(seed % moves.len() as u64) as usize]);
            history.push(board.hash);
        }
        if ok && !legal_moves(&board).is_empty() {
            return (board, history);
        }
        seed = next_random(seed ^ 0xA5A5_A5A5);
    }
}

/// Joue une partie complète. Renvoie le score des Blancs et les positions
/// calmes rencontrées, ou None si l'entraînement a été arrêté.
fn play_game(
    white: &Arc<Weights>,
    black: &Arc<Weights>,
    opening_seed: u64,
    nodes: u64,
    searchers: &mut [Searcher; 2],
    stop: &Arc<AtomicBool>,
    live: &mut dyn FnMut(&Board, Option<Move>, usize),
) -> Option<(f32, Vec<Board>)> {
    let (mut board, mut history) = random_opening(opening_seed);
    let mut quiet_positions = Vec::new();
    let mut last_move = None;
    // Adjudication : si les deux camps voient un avantage décisif plusieurs
    // coups de suite, inutile de jouer la fin.
    let mut decisive_streak = 0;
    let mut decisive_sign = 0;

    let result = loop {
        live(&board, last_move, history.len() - 1);
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let moves = legal_moves(&board);
        if moves.is_empty() {
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
            || history.len() > MAX_GAME_PLIES
        {
            break 0.5;
        }

        let side = board.side_to_move;
        let weights = if side == Color::White { white } else { black };
        let result = searchers[side.index()].search(
            &board,
            &history,
            weights.clone(),
            SearchLimits::nodes(nodes),
            stop.clone(),
            &mut |_| {},
        );
        let mv = result.best_move.expect("il reste des coups légaux");

        let white_score = if side == Color::White {
            result.score
        } else {
            -result.score
        };
        if white_score.abs() >= 1000 {
            let sign = white_score.signum();
            decisive_streak = if sign == decisive_sign {
                decisive_streak + 1
            } else {
                1
            };
            decisive_sign = sign;
        } else {
            decisive_streak = 0;
        }

        let quiet = board.captured_kind(mv).is_none() && mv.promotion.is_none();
        if quiet && !board.in_check() && result.score.abs() < MATE_THRESHOLD {
            quiet_positions.push(board);
        }

        board = board.make_move(mv);
        history.push(board.hash);
        last_move = Some(mv);

        if decisive_streak >= 8 {
            break if decisive_sign > 0 { 1.0 } else { 0.0 };
        }
    };
    live(&board, last_move, history.len() - 1);
    Some((result, quiet_positions))
}

fn to_params(weights: &Weights) -> Vec<f64> {
    weights
        .mg
        .iter()
        .chain(weights.eg.iter())
        .map(|&v| v as f64)
        .collect()
}

fn from_params(params: &[f64]) -> Weights {
    let round = |v: &f64| v.round() as i32;
    Weights {
        mg: params[..NUM_TERMS].iter().map(round).collect(),
        eg: params[NUM_TERMS..].iter().map(round).collect(),
    }
}

/// Erreur quadratique moyenne entre sigmoïde(évaluation) et résultat,
/// et (si demandé) son gradient par rapport à chaque poids.
fn loss_and_gradient(
    data: &[Sample],
    params: &[f64],
    threads: usize,
    with_gradient: bool,
) -> (f64, Vec<f64>) {
    if data.is_empty() {
        return (0.0, vec![0.0; params.len()]);
    }
    let chunk_size = data.len().div_ceil(threads.max(1));
    let partials: Vec<(f64, Vec<f64>)> = thread::scope(|scope| {
        let handles: Vec<_> = data
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut loss = 0.0;
                    let mut gradient = vec![0.0; if with_gradient { params.len() } else { 0 }];
                    for sample in chunk {
                        let phase = sample.phase as f64 / MAX_PHASE as f64;
                        let mut score = 0.0;
                        for &(term, coef) in &sample.features {
                            let t = term as usize;
                            score += coef as f64
                                * (params[t] * phase + params[NUM_TERMS + t] * (1.0 - phase));
                        }
                        let predicted = 1.0 / (1.0 + (-score * SIGMOID_SCALE).exp());
                        let error = predicted - sample.result as f64;
                        loss += error * error;
                        if with_gradient {
                            let g = 2.0 * error * predicted * (1.0 - predicted) * SIGMOID_SCALE;
                            for &(term, coef) in &sample.features {
                                let t = term as usize;
                                gradient[t] += g * coef as f64 * phase;
                                gradient[NUM_TERMS + t] += g * coef as f64 * (1.0 - phase);
                            }
                        }
                    }
                    (loss, gradient)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let n = data.len() as f64;
    let mut loss = 0.0;
    let mut gradient = vec![0.0; params.len()];
    for (partial_loss, partial_gradient) in partials {
        loss += partial_loss;
        for (g, p) in gradient.iter_mut().zip(partial_gradient) {
            *g += p;
        }
    }
    gradient.iter_mut().for_each(|g| *g /= n);
    (loss / n, gradient)
}

/// Descente de gradient (optimiseur Adam) en partant des poids du champion.
fn tune(
    app: &App,
    data: &[Sample],
    start: &Weights,
    learning_rate: f64,
    threads: usize,
    stop: &AtomicBool,
) -> Option<Weights> {
    let anchor = to_params(start);
    let mut params = anchor.clone();
    let mut m = vec![0.0; params.len()];
    let mut v = vec![0.0; params.len()];
    let (beta1, beta2, epsilon) = (0.9, 0.999, 1e-8);

    for epoch in 1..=TUNING_EPOCHS {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        let (loss, mut gradient) = loss_and_gradient(data, &params, threads, true);
        for i in 0..params.len() {
            gradient[i] += ANCHOR_STRENGTH * (params[i] - anchor[i]);
            m[i] = beta1 * m[i] + (1.0 - beta1) * gradient[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * gradient[i] * gradient[i];
            let m_hat = m[i] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = v[i] / (1.0 - beta2.powi(epoch as i32));
            params[i] -= learning_rate * m_hat / (v_hat.sqrt() + epsilon);
        }
        // Le roi n'a pas de valeur matérielle (il y en a toujours un de chaque côté).
        params[MATERIAL + 5] = 0.0;
        params[NUM_TERMS + MATERIAL + 5] = 0.0;

        {
            let mut state = app.training.lock().unwrap();
            state.losses.push(loss);
            state.progress = epoch as f64 / TUNING_EPOCHS as f64;
        }
        let weights = (epoch % 8 == 0 || epoch == TUNING_EPOCHS).then(|| from_params(&params));
        app.emit(json!({
            "type": "train_epoch",
            "epoch": epoch,
            "epochs": TUNING_EPOCHS,
            "loss": loss,
            "candidate": weights,
        }));
    }
    Some(from_params(&params))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elo_of_even_score_is_zero() {
        assert!(elo_difference(0.5).abs() < 1e-9);
        assert!(elo_difference(0.64) > 90.0 && elo_difference(0.64) < 110.0);
    }

    #[test]
    fn random_openings_are_playable() {
        for seed in 1..50 {
            let (board, history) = random_opening(seed);
            assert_eq!(history.len(), OPENING_PLIES + 1);
            assert!(!legal_moves(&board).is_empty());
        }
    }

    #[test]
    fn gradient_descent_learns_material() {
        // Des positions où les Blancs ont une dame de plus et gagnent :
        // la valeur de la dame doit devenir positive.
        let board =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let (features, phase) = eval::trace(&board);
        let data: Vec<Sample> = (0..50)
            .map(|_| Sample {
                features: features.clone(),
                phase: phase as u8,
                result: 1.0,
            })
            .collect();
        let mut params = to_params(&Weights::zero());
        for _ in 0..50 {
            let (_, gradient) = loss_and_gradient(&data, &params, 2, true);
            for (p, g) in params.iter_mut().zip(gradient) {
                *p -= 20_000.0 * g;
            }
        }
        let queen = crate::eval::MATERIAL + crate::board::PieceKind::Queen.index();
        assert!(params[queen] > 0.0);
    }
}
