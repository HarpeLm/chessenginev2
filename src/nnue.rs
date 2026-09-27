//! Réseau de neurones d'évaluation (NNUE : « efficiently updatable neural network »).
//!
//! Architecture : 768 entrées → 2 × 128 neurones → 1 sortie.
//!
//! - **Entrées** : une par (couleur, type de pièce, case) = 2 × 6 × 64 = 768.
//!   Une entrée vaut 1 si cette pièce est sur cette case, sinon 0. Une position
//!   n'en active qu'une trentaine.
//! - **Couche cachée** (« accumulateur ») : 128 neurones, calculés deux fois :
//!   une fois vue par les Blancs, une fois vue par les Noirs (échiquier
//!   retourné). Les mêmes poids servent aux deux points de vue.
//! - **Sortie** : on concatène le point de vue du camp au trait et celui de
//!   l'adversaire, on écrête chaque neurone entre 0 et 1 (CReLU) et on fait
//!   une somme pondérée. Le résultat × 400 donne un score en centipions.
//!
//! Le « UE » de NNUE : quand un coup est joué, seules 2 à 4 entrées changent.
//! Au lieu de tout recalculer, on ajoute et retire quelques colonnes de poids
//! à l'accumulateur. C'est ce qui rend le réseau assez rapide pour la recherche.
//!
//! Pour la vitesse, la recherche utilise une version en nombres entiers
//! (« quantifiée ») des poids ; l'entraînement travaille en nombres réels.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use serde_json::{json, Value};

use crate::board::{Board, Color, Piece, PieceKind};

pub const INPUTS: usize = 768;
pub const HIDDEN: usize = 128;
/// Échelle de quantification de la couche cachée : 1,0 devient 255.
const QA: i32 = 255;
/// Échelle de quantification de la sortie : 1,0 devient 64.
const QB: i32 = 64;
/// Les poids restent dans [−1,98 ; 1,98] pour que la version entière ne déborde pas.
const CLIP: f32 = 1.98;
/// Sortie du réseau × SCALE = score en centipions.
pub const SCALE: i32 = 400;
/// On garde l'évaluation loin des scores de mat.
const MAX_EVAL: i32 = 20_000;

const MAGIC: &[u8; 4] = b"CENN";
const DATASET_MAGIC: &[u8; 4] = b"CEDS";
const VERSION: u32 = 1;

/// Numéro de l'entrée correspondant à une pièce sur une case, vue d'un camp.
fn feature_index(perspective: usize, color: usize, kind: usize, sq: usize) -> usize {
    let sq = if perspective == 0 { sq } else { sq ^ 56 };
    let side = if color == perspective { 0 } else { 384 };
    side + kind * 64 + sq
}

fn feature(perspective: usize, piece: Piece, sq: usize) -> usize {
    feature_index(perspective, piece.color.index(), piece.kind.index(), sq)
}

/// La couche cachée, vue par les Blancs (0) et par les Noirs (1).
#[derive(Clone, Copy)]
pub struct Accumulator {
    values: [[i16; HIDDEN]; 2],
}

impl Default for Accumulator {
    fn default() -> Self {
        Accumulator {
            values: [[0; HIDDEN]; 2],
        }
    }
}

#[derive(Clone)]
pub struct Network {
    // Poids réels (entraînement, sauvegarde).
    ft_weights: Vec<f32>,
    ft_bias: Vec<f32>,
    out_weights: Vec<f32>,
    out_bias: f32,
    /// Nombre total de passes d'entraînement effectuées.
    pub epochs: u32,
    // Poids entiers (recherche).
    q_ft_weights: Vec<i16>,
    q_ft_bias: Vec<i16>,
    q_out_weights: Vec<i16>,
    q_out_bias: i32,
}

impl Network {
    /// Un réseau neuf, aux poids tirés au hasard (petits).
    pub fn random(mut seed: u64) -> Network {
        let mut uniform = |scale: f32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale
        };
        let ft_weights = (0..INPUTS * HIDDEN).map(|_| uniform(0.1)).collect();
        let ft_bias = (0..HIDDEN).map(|_| uniform(0.1) + 0.1).collect();
        let out_weights = (0..2 * HIDDEN).map(|_| uniform(0.1)).collect();
        let mut network = Network {
            ft_weights,
            ft_bias,
            out_weights,
            out_bias: 0.0,
            epochs: 0,
            q_ft_weights: Vec::new(),
            q_ft_bias: Vec::new(),
            q_out_weights: Vec::new(),
            q_out_bias: 0,
        };
        network.quantize();
        network
    }

    pub fn parameter_count() -> usize {
        INPUTS * HIDDEN + HIDDEN + 2 * HIDDEN + 1
    }

    /// Recalcule les poids entiers à partir des poids réels.
    fn quantize(&mut self) {
        let q = |v: f32, scale: i32| (v.clamp(-CLIP, CLIP) * scale as f32).round() as i16;
        self.q_ft_weights = self.ft_weights.iter().map(|&v| q(v, QA)).collect();
        self.q_ft_bias = self.ft_bias.iter().map(|&v| q(v, QA)).collect();
        self.q_out_weights = self.out_weights.iter().map(|&v| q(v, QB)).collect();
        self.q_out_bias = (self.out_bias * (QA * QB) as f32).round() as i32;
    }

    fn add(&self, acc: &mut Accumulator, piece: Piece, sq: usize) {
        for perspective in 0..2 {
            let f = feature(perspective, piece, sq);
            let column = &self.q_ft_weights[f * HIDDEN..(f + 1) * HIDDEN];
            for (a, &w) in acc.values[perspective].iter_mut().zip(column) {
                *a += w;
            }
        }
    }

    fn remove(&self, acc: &mut Accumulator, piece: Piece, sq: usize) {
        for perspective in 0..2 {
            let f = feature(perspective, piece, sq);
            let column = &self.q_ft_weights[f * HIDDEN..(f + 1) * HIDDEN];
            for (a, &w) in acc.values[perspective].iter_mut().zip(column) {
                *a -= w;
            }
        }
    }

    /// Calcule l'accumulateur depuis zéro.
    pub fn refresh(&self, board: &Board) -> Accumulator {
        let mut acc = Accumulator::default();
        for perspective in 0..2 {
            acc.values[perspective].copy_from_slice(&self.q_ft_bias);
        }
        for sq in 0..64 {
            if let Some(piece) = board.squares[sq] {
                self.add(&mut acc, piece, sq);
            }
        }
        acc
    }

    /// Met à jour l'accumulateur de `before` pour qu'il corresponde à `after` :
    /// seules les cases qui ont changé sont traitées.
    pub fn update(&self, acc: &mut Accumulator, before: &Board, after: &Board) {
        for sq in 0..64 {
            let (old, new) = (before.squares[sq], after.squares[sq]);
            if old != new {
                if let Some(piece) = old {
                    self.remove(acc, piece, sq);
                }
                if let Some(piece) = new {
                    self.add(acc, piece, sq);
                }
            }
        }
    }

    /// Score en centipions du point de vue du camp au trait.
    pub fn evaluate(&self, acc: &Accumulator, side_to_move: Color) -> i32 {
        let us = side_to_move.index();
        let them = 1 - us;
        let mut sum: i32 = 0;
        for i in 0..HIDDEN {
            sum += (acc.values[us][i] as i32).clamp(0, QA) * self.q_out_weights[i] as i32;
            sum +=
                (acc.values[them][i] as i32).clamp(0, QA) * self.q_out_weights[HIDDEN + i] as i32;
        }
        let score = (sum + self.q_out_bias) as i64 * SCALE as i64 / (QA * QB) as i64;
        (score as i32).clamp(-MAX_EVAL, MAX_EVAL)
    }

    pub fn evaluate_board(&self, board: &Board) -> i32 {
        self.evaluate(&self.refresh(board), board.side_to_move)
    }

    /// Calcul en nombres réels (sert à vérifier la version entière).
    #[cfg(test)]
    fn evaluate_float(&self, board: &Board) -> f32 {
        let mut hidden = [[0f32; HIDDEN]; 2];
        for (perspective, values) in hidden.iter_mut().enumerate() {
            values.copy_from_slice(&self.ft_bias);
            for sq in 0..64 {
                if let Some(piece) = board.squares[sq] {
                    let f = feature(perspective, piece, sq);
                    for (v, w) in values.iter_mut().zip(&self.ft_weights[f * HIDDEN..]) {
                        *v += w;
                    }
                }
            }
        }
        let us = board.side_to_move.index();
        let mut out = self.out_bias;
        for i in 0..HIDDEN {
            out += hidden[us][i].clamp(0.0, 1.0) * self.out_weights[i];
            out += hidden[1 - us][i].clamp(0.0, 1.0) * self.out_weights[HIDDEN + i];
        }
        out * SCALE as f32
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(16 + 4 * Network::parameter_count());
        bytes.extend_from_slice(MAGIC);
        for value in [VERSION, INPUTS as u32, HIDDEN as u32, self.epochs] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let floats = self
            .ft_weights
            .iter()
            .chain(&self.ft_bias)
            .chain(&self.out_weights)
            .chain(std::iter::once(&self.out_bias));
        for value in floats {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::File::create(path)?.write_all(&bytes)
    }

    pub fn load(path: &str) -> Option<Network> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        if bytes.len() != 20 + 4 * Network::parameter_count() || &bytes[..4] != MAGIC {
            return None;
        }
        let word = |i: usize| u32::from_le_bytes(bytes[4 + 4 * i..8 + 4 * i].try_into().unwrap());
        if word(0) != VERSION || word(1) != INPUTS as u32 || word(2) != HIDDEN as u32 {
            return None;
        }
        let floats: Vec<f32> = bytes[20..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let (ft_weights, rest) = floats.split_at(INPUTS * HIDDEN);
        let (ft_bias, rest) = rest.split_at(HIDDEN);
        let (out_weights, rest) = rest.split_at(2 * HIDDEN);
        let mut network = Network {
            ft_weights: ft_weights.to_vec(),
            ft_bias: ft_bias.to_vec(),
            out_weights: out_weights.to_vec(),
            out_bias: rest[0],
            epochs: word(3),
            q_ft_weights: Vec::new(),
            q_ft_bias: Vec::new(),
            q_out_weights: Vec::new(),
            q_out_bias: 0,
        };
        network.quantize();
        Some(network)
    }

    /// « Interroge » le réseau pour rendre visible ce qu'il a appris : on pose
    /// une pièce blanche sur chaque case d'une position de référence et on
    /// mesure de combien l'évaluation change. En finale, la référence ne
    /// contient que les deux rois ; en milieu de partie, les rois et les pions.
    pub fn probe(&self) -> Value {
        let references = [
            ("mg", "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"),
            ("eg", "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        ];
        let kinds = [
            PieceKind::Pawn,
            PieceKind::Knight,
            PieceKind::Bishop,
            PieceKind::Rook,
            PieceKind::Queen,
            PieceKind::King,
        ];
        let mut result = serde_json::Map::new();
        for (phase, fen) in references {
            let base = Board::from_fen(fen).expect("position de référence valide");
            let mut values = Vec::new();
            let mut tables = Vec::new();
            for kind in kinds {
                let mut reference = base;
                if kind == PieceKind::King {
                    reference.squares[4] = None;
                }
                let reference_eval = self.evaluate_board(&reference);
                let piece = Piece::new(kind, Color::White);
                let raw: Vec<Option<i32>> = (0..64)
                    .map(|sq| {
                        let rank = sq / 8;
                        if reference.squares[sq].is_some()
                            || (kind == PieceKind::Pawn && (rank == 0 || rank == 7))
                        {
                            return None;
                        }
                        let mut position = reference;
                        position.squares[sq] = Some(piece);
                        Some(self.evaluate_board(&position) - reference_eval)
                    })
                    .collect();
                let known: Vec<i32> = raw.iter().flatten().copied().collect();
                let mean = known.iter().sum::<i32>() / known.len().max(1) as i32;
                // Le roi est toujours là : sa « valeur » n'a pas de sens, seule sa case compte.
                values.push(if kind == PieceKind::King { 0 } else { mean });
                tables.push(raw.iter().map(|v| v.map(|v| v - mean)).collect::<Vec<_>>());
            }
            result.insert(phase.into(), json!({ "values": values, "pst": tables }));
        }
        Value::Object(result)
    }
}

// ═══════════════════════════ Données d'entraînement ═══════════════════════════

/// Positions d'entraînement, stockées de façon compacte (≈ 60 octets chacune).
#[derive(Default)]
pub struct Dataset {
    /// Pour chaque pièce : (case << 4) | numéro de pièce (0..12).
    pieces: Vec<u16>,
    /// `offsets[i]..offsets[i + 1]` : les pièces de la position `i`.
    offsets: Vec<u32>,
    side_to_move: Vec<u8>,
    /// Score de la recherche, en centipions, du point de vue du camp au trait.
    scores: Vec<i16>,
    /// Résultat de la partie pour le camp au trait : 0 = perdu, 1 = nulle, 2 = gagné.
    results: Vec<u8>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.side_to_move.len()
    }

    pub fn push(&mut self, board: &Board, score: i32, white_result: f32) {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        for sq in 0..64 {
            if let Some(piece) = board.squares[sq] {
                self.pieces.push(((sq as u16) << 4) | piece.index() as u16);
            }
        }
        self.offsets.push(self.pieces.len() as u32);
        let stm = board.side_to_move;
        self.side_to_move.push(stm.index() as u8);
        self.scores.push(score.clamp(-3000, 3000) as i16);
        let result = if stm == Color::White {
            white_result
        } else {
            1.0 - white_result
        };
        self.results.push((result * 2.0).round() as u8);
    }

    /// Oublie les `count` positions les plus anciennes.
    pub fn drop_oldest(&mut self, count: usize) {
        let count = count.min(self.len());
        if count == 0 {
            return;
        }
        let cut = self.offsets[count];
        self.pieces.drain(..cut as usize);
        self.offsets.drain(..count);
        for offset in self.offsets.iter_mut() {
            *offset -= cut;
        }
        self.side_to_move.drain(..count);
        self.scores.drain(..count);
        self.results.drain(..count);
    }

    pub fn clear(&mut self) {
        *self = Dataset::default();
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(16 + self.pieces.len() * 2 + self.len() * 8);
        bytes.extend_from_slice(DATASET_MAGIC);
        bytes.extend_from_slice(&(self.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.pieces.len() as u32).to_le_bytes());
        for value in &self.pieces {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in self.offsets.iter().skip(1) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.side_to_move);
        for value in &self.scores {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&self.results);
        // Écrit d'abord dans un fichier temporaire : un arrêt brutal ne corrompt pas l'ancien.
        let temporary = format!("{path}.tmp");
        std::fs::File::create(&temporary)?.write_all(&bytes)?;
        std::fs::rename(temporary, path)
    }

    pub fn load(path: &str) -> Option<Dataset> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .ok()?
            .read_to_end(&mut bytes)
            .ok()?;
        if bytes.len() < 12 || &bytes[..4] != DATASET_MAGIC {
            return None;
        }
        let read_u32 = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let count = read_u32(4) as usize;
        let piece_count = read_u32(8) as usize;
        let expected = 12 + piece_count * 2 + count * 4 + count + count * 2 + count;
        if bytes.len() != expected {
            return None;
        }
        let mut at = 12;
        let pieces = bytes[at..at + piece_count * 2]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        at += piece_count * 2;
        let mut offsets = vec![0u32];
        offsets.extend(
            bytes[at..at + count * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
        );
        at += count * 4;
        let side_to_move = bytes[at..at + count].to_vec();
        at += count;
        let scores = bytes[at..at + count * 2]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        at += count * 2;
        let results = bytes[at..at + count].to_vec();
        if offsets.last().copied() != Some(piece_count as u32) {
            return None;
        }
        Some(Dataset {
            pieces,
            offsets: if count == 0 { Vec::new() } else { offsets },
            side_to_move,
            scores,
            results,
        })
    }

    /// Cible d'apprentissage : un mélange du score de la recherche (plus précis)
    /// et du résultat réel de la partie (la vérité, mais bruitée).
    fn target(&self, i: usize, score_weight: f32) -> f32 {
        let score = 1.0 / (1.0 + (-(self.scores[i] as f32) / SCALE as f32).exp());
        let result = self.results[i] as f32 / 2.0;
        score_weight * score + (1.0 - score_weight) * result
    }

    fn pieces_of(&self, i: usize) -> &[u16] {
        &self.pieces[self.offsets[i] as usize..self.offsets[i + 1] as usize]
    }
}

// ═══════════════════════════ Entraînement ═══════════════════════════

pub struct TrainOptions {
    pub epochs: usize,
    pub learning_rate: f32,
    /// Part du score de la recherche dans la cible (le reste : le résultat).
    pub score_weight: f32,
    pub batch_size: usize,
    pub threads: usize,
    pub seed: u64,
}

pub struct TrainProgress {
    pub epoch: usize,
    pub epochs: usize,
    pub step: usize,
    pub steps: usize,
    /// Erreur moyenne sur les derniers lots d'entraînement.
    pub train_loss: f64,
    /// Erreur sur les positions de validation (jamais utilisées pour apprendre),
    /// calculée à la fin de chaque passe.
    pub validation_loss: Option<f64>,
}

/// Gradients (ou moments d'Adam) pour tous les paramètres.
#[derive(Clone)]
struct Params {
    ft_weights: Vec<f32>,
    ft_bias: Vec<f32>,
    out_weights: Vec<f32>,
    out_bias: f32,
}

impl Params {
    fn zeros() -> Params {
        Params {
            ft_weights: vec![0.0; INPUTS * HIDDEN],
            ft_bias: vec![0.0; HIDDEN],
            out_weights: vec![0.0; 2 * HIDDEN],
            out_bias: 0.0,
        }
    }

    fn clear(&mut self) {
        self.ft_weights.fill(0.0);
        self.ft_bias.fill(0.0);
        self.out_weights.fill(0.0);
        self.out_bias = 0.0;
    }
}

/// Une position sur 20 sert à la validation : on vérifie ainsi que le réseau
/// généralise au lieu d'apprendre par cœur.
fn is_validation(i: usize) -> bool {
    i % 20 == 0
}

/// Passe avant (et arrière si `gradient` est fourni) sur une position.
/// Avec `mirror`, l'échiquier est retourné de gauche à droite (colonne a ↔ h) :
/// la position reste aussi bonne ou mauvaise, ce qui double les exemples.
/// Renvoie l'erreur au carré.
fn process(
    network: &Network,
    data: &Dataset,
    i: usize,
    score_weight: f32,
    mirror: bool,
    gradient: Option<&mut Params>,
) -> f32 {
    let pieces = data.pieces_of(i);
    let mut hidden = [[0f32; HIDDEN]; 2];
    let mut features = [[0usize; 32]; 2];
    let count = pieces.len().min(32);
    for (perspective, values) in hidden.iter_mut().enumerate() {
        values.copy_from_slice(&network.ft_bias);
        for (slot, &packed) in pieces.iter().take(32).enumerate() {
            let sq = (packed >> 4) as usize ^ if mirror { 7 } else { 0 };
            let index = (packed & 15) as usize;
            let f = feature_index(perspective, index / 6, index % 6, sq);
            features[perspective][slot] = f;
            for (v, w) in values
                .iter_mut()
                .zip(&network.ft_weights[f * HIDDEN..(f + 1) * HIDDEN])
            {
                *v += w;
            }
        }
    }

    let us = data.side_to_move[i] as usize;
    let them = 1 - us;
    let mut out = network.out_bias;
    for j in 0..HIDDEN {
        out += hidden[us][j].clamp(0.0, 1.0) * network.out_weights[j];
        out += hidden[them][j].clamp(0.0, 1.0) * network.out_weights[HIDDEN + j];
    }
    let predicted = 1.0 / (1.0 + (-out).exp());
    let error = predicted - data.target(i, score_weight);

    if let Some(grad) = gradient {
        let g = 2.0 * error * predicted * (1.0 - predicted);
        grad.out_bias += g;
        for (side, offset) in [(us, 0), (them, HIDDEN)] {
            let mut delta = [0f32; HIDDEN];
            for j in 0..HIDDEN {
                let h = hidden[side][j];
                grad.out_weights[offset + j] += g * h.clamp(0.0, 1.0);
                if h > 0.0 && h < 1.0 {
                    delta[j] = g * network.out_weights[offset + j];
                }
            }
            for j in 0..HIDDEN {
                grad.ft_bias[j] += delta[j];
            }
            for &f in &features[side][..count] {
                for (gw, d) in grad.ft_weights[f * HIDDEN..(f + 1) * HIDDEN]
                    .iter_mut()
                    .zip(&delta)
                {
                    *gw += d;
                }
            }
        }
    }
    error * error
}

/// Erreur moyenne sur les positions de validation.
pub fn validation_loss(
    network: &Network,
    data: &Dataset,
    score_weight: f32,
    threads: usize,
) -> f64 {
    let indices: Vec<usize> = (0..data.len()).filter(|&i| is_validation(i)).collect();
    if indices.is_empty() {
        return 0.0;
    }
    let chunk = indices.len().div_ceil(threads.max(1));
    let total: f64 = thread::scope(|scope| {
        let handles: Vec<_> = indices
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|&i| process(network, data, i, score_weight, false, None) as f64)
                        .sum::<f64>()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    total / indices.len() as f64
}

/// Entraîne le réseau (descente de gradient par lots, optimiseur Adam).
/// Renvoie false si l'entraînement a été interrompu.
pub fn train(
    network: &mut Network,
    data: &Dataset,
    options: &TrainOptions,
    stop: &AtomicBool,
    on_progress: &mut dyn FnMut(&TrainProgress, &Network),
) -> bool {
    let mut indices: Vec<usize> = (0..data.len()).filter(|&i| !is_validation(i)).collect();
    if indices.is_empty() {
        return true;
    }
    let threads = options.threads.max(1);
    let steps_per_epoch = indices.len().div_ceil(options.batch_size);
    let total_steps = steps_per_epoch * options.epochs;
    let mut gradients: Vec<Params> = (0..threads).map(|_| Params::zeros()).collect();
    let mut m = Params::zeros();
    let mut v = Params::zeros();
    let (beta1, beta2, epsilon) = (0.9f32, 0.999f32, 1e-8f32);
    let mut seed = options.seed | 1;
    let mut step = 0;
    let mut recent_loss = 0.0;
    let mut recent_count = 0;

    for epoch in 1..=options.epochs {
        // Mélange (Fisher-Yates) pour que chaque lot soit varié.
        for i in (1..indices.len()).rev() {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            indices.swap(i, (seed % (i as u64 + 1)) as usize);
        }

        for batch in indices.chunks(options.batch_size) {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            step += 1;
            let chunk = batch.len().div_ceil(threads);
            let net: &Network = network;
            let losses: Vec<f64> = thread::scope(|scope| {
                let handles: Vec<_> = gradients
                    .iter_mut()
                    .zip(batch.chunks(chunk))
                    .map(|(grad, part)| {
                        scope.spawn(move || {
                            grad.clear();
                            part.iter()
                                .map(|&i| {
                                    // Une fois sur deux, et pas la même d'une passe à
                                    // l'autre, on retourne l'échiquier.
                                    let mirror =
                                        ((i.wrapping_mul(0x9E37_79B9) >> 7) ^ epoch) & 1 == 1;
                                    process(net, data, i, options.score_weight, mirror, Some(grad))
                                        as f64
                                })
                                .sum::<f64>()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            recent_loss += losses.iter().sum::<f64>();
            recent_count += batch.len();

            // Somme des gradients des fils, puis pas d'Adam.
            let used = batch.chunks(chunk).count();
            let (first, others) = gradients.split_at_mut(1);
            let total = &mut first[0];
            for other in &others[..used - 1] {
                for (a, b) in total.ft_weights.iter_mut().zip(&other.ft_weights) {
                    *a += b;
                }
                for (a, b) in total.ft_bias.iter_mut().zip(&other.ft_bias) {
                    *a += b;
                }
                for (a, b) in total.out_weights.iter_mut().zip(&other.out_weights) {
                    *a += b;
                }
                total.out_bias += other.out_bias;
            }
            // Le pas d'apprentissage décroît doucement au fil de l'entraînement.
            let progress = step as f32 / total_steps as f32;
            let lr = options.learning_rate
                * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()));
            let scale = 1.0 / batch.len() as f32;
            let correction1 = 1.0 - beta1.powi(step as i32);
            let correction2 = 1.0 - beta2.powi(step as i32);
            let adam = |param: &mut f32, grad: f32, m: &mut f32, v: &mut f32| {
                let g = grad * scale;
                *m = beta1 * *m + (1.0 - beta1) * g;
                *v = beta2 * *v + (1.0 - beta2) * g * g;
                *param -= lr * (*m / correction1) / ((*v / correction2).sqrt() + epsilon);
                *param = param.clamp(-CLIP, CLIP);
            };
            for i in 0..network.ft_weights.len() {
                adam(
                    &mut network.ft_weights[i],
                    total.ft_weights[i],
                    &mut m.ft_weights[i],
                    &mut v.ft_weights[i],
                );
            }
            for i in 0..HIDDEN {
                adam(
                    &mut network.ft_bias[i],
                    total.ft_bias[i],
                    &mut m.ft_bias[i],
                    &mut v.ft_bias[i],
                );
            }
            for i in 0..2 * HIDDEN {
                adam(
                    &mut network.out_weights[i],
                    total.out_weights[i],
                    &mut m.out_weights[i],
                    &mut v.out_weights[i],
                );
            }
            adam(
                &mut network.out_bias,
                total.out_bias,
                &mut m.out_bias,
                &mut v.out_bias,
            );

            let end_of_epoch = step % steps_per_epoch == 0;
            if step % 8 == 0 || end_of_epoch {
                network.quantize();
                let validation = end_of_epoch
                    .then(|| validation_loss(network, data, options.score_weight, threads));
                on_progress(
                    &TrainProgress {
                        epoch,
                        epochs: options.epochs,
                        step,
                        steps: total_steps,
                        train_loss: recent_loss / recent_count.max(1) as f64,
                        validation_loss: validation,
                    },
                    network,
                );
                recent_loss = 0.0;
                recent_count = 0;
            }
        }
        network.epochs += 1;
    }
    network.quantize();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::START_FEN;
    use crate::movegen::legal_moves;

    const FENS: [&str; 4] = [
        START_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
    ];

    #[test]
    fn quantized_matches_float() {
        let network = Network::random(42);
        for fen in FENS {
            let board = Board::from_fen(fen).unwrap();
            let exact = network.evaluate_float(&board);
            let quantized = network.evaluate_board(&board) as f32;
            assert!(
                (exact - quantized).abs() < 5.0 + exact.abs() * 0.02,
                "{fen}: {exact} vs {quantized}"
            );
        }
    }

    #[test]
    fn incremental_update_matches_refresh() {
        let network = Network::random(7);
        fn walk(network: &Network, board: &Board, acc: &Accumulator, depth: u32) {
            let fresh = network.refresh(board);
            assert_eq!(acc.values, fresh.values, "{}", board.to_fen());
            if depth == 0 {
                return;
            }
            for mv in legal_moves(board) {
                let child = board.make_move(mv);
                let mut next = *acc;
                network.update(&mut next, board, &child);
                walk(network, &child, &next, depth - 1);
            }
        }
        // Kiwipete : roques, prises en passant et promotions en deux coups.
        let board = Board::from_fen(FENS[1]).unwrap();
        walk(&network, &board, &network.refresh(&board), 2);
    }

    #[test]
    fn save_and_load() {
        let network = Network::random(3);
        let path = std::env::temp_dir().join("chessengine-test.nnue");
        let path = path.to_string_lossy();
        network.save(&path).unwrap();
        let loaded = Network::load(&path).unwrap();
        let board = Board::from_fen(FENS[3]).unwrap();
        assert_eq!(
            network.evaluate_board(&board),
            loaded.evaluate_board(&board)
        );
        let _ = std::fs::remove_file(&*path);
    }

    #[test]
    fn learns_extra_queen_is_good() {
        // Positions où le camp au trait a une dame de plus et gagne, et
        // l'inverse : le réseau doit apprendre à les distinguer.
        let mut data = Dataset::default();
        let up =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        let down =
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1").unwrap();
        for _ in 0..200 {
            data.push(&up, 900, 1.0);
            data.push(&down, -900, 0.0);
        }
        let mut network = Network::random(11);
        let options = TrainOptions {
            epochs: 30,
            learning_rate: 0.01,
            score_weight: 0.5,
            batch_size: 64,
            threads: 2,
            seed: 5,
        };
        let before = validation_loss(&network, &data, 0.5, 2);
        assert!(train(
            &mut network,
            &data,
            &options,
            &AtomicBool::new(false),
            &mut |_, _| {}
        ));
        let after = validation_loss(&network, &data, 0.5, 2);
        assert!(after < before * 0.5, "{before} → {after}");
        assert!(network.evaluate_board(&up) > 200);
        assert!(network.evaluate_board(&down) < -200);
    }

    #[test]
    fn dataset_save_and_load() {
        let mut data = Dataset::default();
        for fen in FENS {
            data.push(&Board::from_fen(fen).unwrap(), 37, 0.5);
        }
        let path = std::env::temp_dir().join("chessengine-test.data");
        let path = path.to_string_lossy();
        data.save(&path).unwrap();
        let loaded = Dataset::load(&path).unwrap();
        assert_eq!(loaded.len(), FENS.len());
        assert_eq!(loaded.pieces, data.pieces);
        assert_eq!(loaded.offsets, data.offsets);
        assert_eq!(loaded.scores, data.scores);
        let _ = std::fs::remove_file(&*path);
    }

    #[test]
    fn dataset_drops_oldest() {
        let mut data = Dataset::default();
        let a = Board::from_fen(FENS[0]).unwrap();
        let b = Board::from_fen(FENS[2]).unwrap();
        data.push(&a, 10, 1.0);
        data.push(&b, -20, 0.0);
        data.drop_oldest(1);
        assert_eq!(data.len(), 1);
        assert_eq!(data.pieces_of(0).len(), 10);
        assert_eq!(data.scores[0], -20);
    }
}
