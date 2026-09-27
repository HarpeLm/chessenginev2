//! Évaluation d'une position.
//!
//! L'évaluation est une somme pondérée de critères (« termes ») : valeur des
//! pièces, tables de position, pions passés, mobilité… Chaque terme a deux
//! poids : un pour le milieu de partie (mg) et un pour la finale (eg). Le
//! score final mélange les deux selon la quantité de pièces restantes.
//!
//! Comme l'évaluation est linéaire en ses poids, l'entraînement (voir
//! `train.rs`) peut ajuster automatiquement chaque poids à partir de parties.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::board::*;
use crate::nnue::{Network, HIDDEN, INPUTS};

/// L'évaluation utilisée par la recherche : la formule classique à poids
/// ajustables, ou le réseau de neurones.
pub enum Evaluator {
    Classic(Weights),
    Nnue(Network),
}

impl Evaluator {
    /// Score en centipions, du point de vue du camp au trait.
    pub fn evaluate(&self, board: &Board) -> i32 {
        match self {
            Evaluator::Classic(weights) => evaluate(board, weights),
            Evaluator::Nnue(network) => network.evaluate_board(board),
        }
    }

    /// Ce que l'interface affiche : les poids de la formule classique, ou ce
    /// qu'on obtient en interrogeant le réseau.
    pub fn view(&self) -> Value {
        match self {
            Evaluator::Classic(weights) => {
                json!({ "kind": "classic", "mg": weights.mg, "eg": weights.eg })
            }
            Evaluator::Nnue(network) => json!({
                "kind": "nnue",
                "probe": network.probe(),
                "inputs": INPUTS,
                "hidden": HIDDEN,
                "parameters": Network::parameter_count(),
                "epochs": network.epochs,
            }),
        }
    }
}

// Disposition des termes dans les vecteurs de poids.
pub const MATERIAL: usize = 0; // 6 termes, un par type de pièce
pub const PST: usize = MATERIAL + 6; // 6 × 64 : bonus selon la case
pub const PASSED_PAWN: usize = PST + 6 * 64; // 8 : selon la rangée
pub const DOUBLED_PAWN: usize = PASSED_PAWN + 8;
pub const ISOLATED_PAWN: usize = DOUBLED_PAWN + 1;
pub const ROOK_OPEN_FILE: usize = ISOLATED_PAWN + 1;
pub const ROOK_SEMI_OPEN_FILE: usize = ROOK_OPEN_FILE + 1;
pub const BISHOP_PAIR: usize = ROOK_SEMI_OPEN_FILE + 1;
pub const MOBILITY: usize = BISHOP_PAIR + 1; // 4 : cavalier, fou, tour, dame
pub const KING_SHIELD: usize = MOBILITY + 4;
pub const TEMPO: usize = KING_SHIELD + 1;
pub const NUM_TERMS: usize = TEMPO + 1;

/// Poids de chaque type de pièce dans le calcul de la phase de jeu.
const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];
pub const MAX_PHASE: i32 = 24;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Weights {
    pub mg: Vec<i32>,
    pub eg: Vec<i32>,
}

impl Weights {
    /// Tous les poids à zéro : le moteur ne sait rien des échecs,
    /// à part les règles et la recherche de mat.
    pub fn zero() -> Weights {
        Weights {
            mg: vec![0; NUM_TERMS],
            eg: vec![0; NUM_TERMS],
        }
    }

    /// Valeurs classiques, inspirées de la « Simplified Evaluation Function »
    /// de Tomasz Michniewski.
    pub fn classic() -> Weights {
        let mut w = Weights::zero();

        let material_mg = [100, 320, 330, 500, 900, 0];
        let material_eg = [110, 300, 320, 520, 930, 0];
        for kind in 0..6 {
            w.mg[MATERIAL + kind] = material_mg[kind];
            w.eg[MATERIAL + kind] = material_eg[kind];
        }

        let tables_mg = [
            &PAWN_TABLE,
            &KNIGHT_TABLE,
            &BISHOP_TABLE,
            &ROOK_TABLE,
            &QUEEN_TABLE,
            &KING_MG_TABLE,
        ];
        let tables_eg = [
            &PAWN_TABLE,
            &KNIGHT_TABLE,
            &BISHOP_TABLE,
            &ROOK_TABLE,
            &QUEEN_TABLE,
            &KING_EG_TABLE,
        ];
        for kind in 0..6 {
            for sq in 0..64 {
                // Les tables sont écrites « vues des Blancs », rangée 8 en haut :
                // la case a1 (0) correspond à l'indice 56 de la table.
                w.mg[PST + kind * 64 + sq] = tables_mg[kind][sq ^ 56];
                w.eg[PST + kind * 64 + sq] = tables_eg[kind][sq ^ 56];
            }
        }

        let passed_mg = [0, 5, 10, 15, 25, 40, 60, 0];
        let passed_eg = [0, 10, 20, 35, 60, 100, 150, 0];
        for rank in 0..8 {
            w.mg[PASSED_PAWN + rank] = passed_mg[rank];
            w.eg[PASSED_PAWN + rank] = passed_eg[rank];
        }

        let scalars = [
            (DOUBLED_PAWN, -10, -20),
            (ISOLATED_PAWN, -10, -15),
            (ROOK_OPEN_FILE, 20, 10),
            (ROOK_SEMI_OPEN_FILE, 10, 5),
            (BISHOP_PAIR, 30, 50),
            (MOBILITY, 4, 4),
            (MOBILITY + 1, 4, 5),
            (MOBILITY + 2, 2, 4),
            (MOBILITY + 3, 1, 2),
            (KING_SHIELD, 10, 0),
            (TEMPO, 10, 10),
        ];
        for (term, mg, eg) in scalars {
            w.mg[term] = mg;
            w.eg[term] = eg;
        }
        w
    }

    pub fn load(path: &str) -> Option<Weights> {
        let text = std::fs::read_to_string(path).ok()?;
        let weights: Weights = serde_json::from_str(&text).ok()?;
        (weights.mg.len() == NUM_TERMS && weights.eg.len() == NUM_TERMS).then_some(weights)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(
            path,
            serde_json::to_string(self).expect("sérialisation des poids"),
        )
    }
}

/// Reçoit les termes activés par une position. `coef` est positif pour les
/// Blancs et négatif pour les Noirs.
trait Sink {
    fn add(&mut self, term: usize, coef: i32);
}

struct Scorer<'a> {
    weights: &'a Weights,
    mg: i32,
    eg: i32,
}

impl Sink for Scorer<'_> {
    #[inline]
    fn add(&mut self, term: usize, coef: i32) {
        self.mg += coef * self.weights.mg[term];
        self.eg += coef * self.weights.eg[term];
    }
}

struct Tracer {
    coefs: Vec<i32>,
}

impl Sink for Tracer {
    fn add(&mut self, term: usize, coef: i32) {
        self.coefs[term] += coef;
    }
}

/// Score de la position en centipions, du point de vue du camp au trait.
pub fn evaluate(board: &Board, weights: &Weights) -> i32 {
    let mut scorer = Scorer {
        weights,
        mg: 0,
        eg: 0,
    };
    let phase = collect_terms(board, &mut scorer);
    let score = (scorer.mg * phase + scorer.eg * (MAX_PHASE - phase)) / MAX_PHASE;
    match board.side_to_move {
        Color::White => score,
        Color::Black => -score,
    }
}

/// Termes activés par la position (du point de vue des Blancs) et phase de jeu.
/// Sert à l'entraînement : evaluate = Σ coef × (poids_mg × phase + poids_eg × (24 − phase)) / 24.
pub fn trace(board: &Board) -> (Vec<(u16, i16)>, i32) {
    let mut tracer = Tracer {
        coefs: vec![0; NUM_TERMS],
    };
    let phase = collect_terms(board, &mut tracer);
    let features = tracer
        .coefs
        .iter()
        .enumerate()
        .filter(|(_, &c)| c != 0)
        .map(|(term, &c)| (term as u16, c as i16))
        .collect();
    (features, phase)
}

fn collect_terms<S: Sink>(board: &Board, sink: &mut S) -> i32 {
    // Première passe : où sont les pions, colonne par colonne ?
    let mut pawns_on_file = [[0i32; 8]; 2];
    let mut lowest_white_pawn = [8i32; 8];
    let mut highest_black_pawn = [-1i32; 8];
    for sq in 0..64 {
        if let Some(piece) = board.squares[sq] {
            if piece.kind == PieceKind::Pawn {
                let file = file_of(sq);
                let rank = rank_of(sq) as i32;
                pawns_on_file[piece.color.index()][file] += 1;
                match piece.color {
                    Color::White => lowest_white_pawn[file] = lowest_white_pawn[file].min(rank),
                    Color::Black => highest_black_pawn[file] = highest_black_pawn[file].max(rank),
                }
            }
        }
    }

    let mut phase = 0;
    let mut bishops = [0; 2];

    for sq in 0..64 {
        let Some(piece) = board.squares[sq] else {
            continue;
        };
        let color = piece.color;
        let c = color.index();
        let sign = if color == Color::White { 1 } else { -1 };
        // Case « relative » : vue du camp de la pièce (les Noirs sont retournés).
        let relative = if color == Color::White { sq } else { sq ^ 56 };
        let kind = piece.kind.index();
        let file = file_of(sq);
        let rank = rank_of(sq) as i32;

        phase += PHASE_WEIGHTS[kind];
        sink.add(MATERIAL + kind, sign);
        sink.add(PST + kind * 64 + relative, sign);

        match piece.kind {
            PieceKind::Pawn => {
                if pawns_on_file[c][file] > 1 {
                    sink.add(DOUBLED_PAWN, sign);
                }
                let left = file > 0 && pawns_on_file[c][file - 1] > 0;
                let right = file < 7 && pawns_on_file[c][file + 1] > 0;
                if !left && !right {
                    sink.add(ISOLATED_PAWN, sign);
                }
                let lo = file.saturating_sub(1);
                let hi = (file + 1).min(7);
                let passed = (lo..=hi).all(|f| match color {
                    Color::White => highest_black_pawn[f] <= rank,
                    Color::Black => lowest_white_pawn[f] >= rank,
                });
                if passed {
                    sink.add(PASSED_PAWN + rank_of(relative), sign);
                }
            }
            PieceKind::Knight => {
                sink.add(
                    MOBILITY,
                    sign * step_mobility(board, sq, color, &KNIGHT_DELTAS),
                );
            }
            PieceKind::Bishop => {
                bishops[c] += 1;
                sink.add(
                    MOBILITY + 1,
                    sign * slide_mobility(board, sq, color, &BISHOP_DIRECTIONS),
                );
            }
            PieceKind::Rook => {
                sink.add(
                    MOBILITY + 2,
                    sign * slide_mobility(board, sq, color, &ROOK_DIRECTIONS),
                );
                if pawns_on_file[0][file] + pawns_on_file[1][file] == 0 {
                    sink.add(ROOK_OPEN_FILE, sign);
                } else if pawns_on_file[c][file] == 0 {
                    sink.add(ROOK_SEMI_OPEN_FILE, sign);
                }
            }
            PieceKind::Queen => {
                let mobility = slide_mobility(board, sq, color, &ROOK_DIRECTIONS)
                    + slide_mobility(board, sq, color, &BISHOP_DIRECTIONS);
                sink.add(MOBILITY + 3, sign * mobility);
            }
            PieceKind::King => {
                // Bouclier : nos pions juste devant le roi.
                let forward = if color == Color::White { 1 } else { -1 };
                let pawn = Some(Piece::new(PieceKind::Pawn, color));
                let shield = (-1..=1)
                    .filter_map(|df| square_at(file as i32 + df, rank + forward))
                    .filter(|&s| board.squares[s] == pawn)
                    .count() as i32;
                if shield > 0 {
                    sink.add(KING_SHIELD, sign * shield);
                }
            }
        }
    }

    for (c, sign) in [(0, 1), (1, -1)] {
        if bishops[c] >= 2 {
            sink.add(BISHOP_PAIR, sign);
        }
    }
    sink.add(
        TEMPO,
        if board.side_to_move == Color::White {
            1
        } else {
            -1
        },
    );

    phase.min(MAX_PHASE)
}

fn step_mobility(board: &Board, sq: usize, color: Color, deltas: &[(i32, i32)]) -> i32 {
    let file = file_of(sq) as i32;
    let rank = rank_of(sq) as i32;
    deltas
        .iter()
        .filter_map(|&(df, dr)| square_at(file + df, rank + dr))
        .filter(|&to| !matches!(board.squares[to], Some(p) if p.color == color))
        .count() as i32
}

fn slide_mobility(board: &Board, sq: usize, color: Color, directions: &[(i32, i32)]) -> i32 {
    let mut count = 0;
    for &(df, dr) in directions {
        let mut file = file_of(sq) as i32 + df;
        let mut rank = rank_of(sq) as i32 + dr;
        while let Some(to) = square_at(file, rank) {
            match board.squares[to] {
                None => count += 1,
                Some(p) => {
                    if p.color != color {
                        count += 1;
                    }
                    break;
                }
            }
            file += df;
            rank += dr;
        }
    }
    count
}

// Tables de position « vues des Blancs » : la première ligne est la rangée 8.
#[rustfmt::skip]
const PAWN_TABLE: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_TABLE: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_TABLE: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK_TABLE: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN_TABLE: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const KING_MG_TABLE: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
];

#[rustfmt::skip]
const KING_EG_TABLE: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Retourne l'échiquier verticalement en échangeant les couleurs.
    fn mirror(board: &Board) -> Board {
        let mut m = Board::empty();
        for sq in 0..64 {
            m.squares[sq ^ 56] = board.squares[sq].map(|p| Piece::new(p.kind, p.color.opponent()));
        }
        m.side_to_move = board.side_to_move.opponent();
        m.castling = ((board.castling & 3) << 2) | (board.castling >> 2);
        m.en_passant = board.en_passant.map(|sq| sq ^ 56);
        m.king_square = [board.king_square[1] ^ 56, board.king_square[0] ^ 56];
        m.hash = m.compute_hash();
        m
    }

    #[test]
    fn evaluation_is_symmetric() {
        let weights = Weights::classic();
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(
                evaluate(&board, &weights),
                evaluate(&mirror(&board), &weights),
                "{fen}"
            );
        }
    }

    #[test]
    fn trace_matches_evaluate() {
        let weights = Weights::classic();
        let board = Board::from_fen(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 b - - 0 10",
        )
        .unwrap();
        let (features, phase) = trace(&board);
        let (mut mg, mut eg) = (0, 0);
        for (term, coef) in features {
            mg += coef as i32 * weights.mg[term as usize];
            eg += coef as i32 * weights.eg[term as usize];
        }
        let white_score = (mg * phase + eg * (MAX_PHASE - phase)) / MAX_PHASE;
        assert_eq!(-white_score, evaluate(&board, &weights));
    }

    #[test]
    fn extra_material_is_good() {
        let weights = Weights::classic();
        // Les Blancs ont une dame de plus.
        let board =
            Board::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap();
        assert!(evaluate(&board, &weights) > 700);
    }
}
