//! Recherche du meilleur coup : négamax avec élagage alpha-bêta,
//! approfondissement itératif, table de transposition, quiescence,
//! coup nul, réductions des coups tardifs et tri des coups.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::board::{Board, PieceKind};
use crate::eval::{Evaluator, Weights};
use crate::movegen::{generate_moves, legal_moves, Move};
use crate::nnue::Accumulator;
use crate::tt::{Bound, Entry, TranspositionTable};

pub const INFINITY: i32 = 32_000;
pub const MATE: i32 = 31_000;
pub const MAX_PLY: usize = 128;
pub const MAX_DEPTH: i32 = 64;
/// Au-delà de ce score, il s'agit d'un mat trouvé.
pub const MATE_THRESHOLD: i32 = MATE - MAX_PLY as i32;

/// Valeurs utilisées uniquement pour trier les captures (MVV-LVA).
const ORDER_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 1000];

#[derive(Clone, Copy, Debug)]
pub struct SearchLimits {
    pub max_depth: i32,
    /// Après ce temps, on ne commence plus de nouvelle itération.
    pub soft_time: Option<Duration>,
    /// Après ce temps, on s'arrête immédiatement.
    pub hard_time: Option<Duration>,
    /// Nombre maximal de positions à visiter.
    pub max_nodes: Option<u64>,
}

impl SearchLimits {
    pub fn infinite() -> SearchLimits {
        SearchLimits {
            max_depth: MAX_DEPTH,
            soft_time: None,
            hard_time: None,
            max_nodes: None,
        }
    }

    pub fn time(duration: Duration) -> SearchLimits {
        SearchLimits {
            soft_time: Some(duration),
            hard_time: Some(duration),
            ..SearchLimits::infinite()
        }
    }

    pub fn nodes(max_nodes: u64) -> SearchLimits {
        SearchLimits {
            max_nodes: Some(max_nodes),
            ..SearchLimits::infinite()
        }
    }
}

/// Marge de sécurité pour ne jamais perdre au temps (communication avec l'interface).
pub const MOVE_OVERHEAD_MS: u64 = 30;

/// Gestion du temps en partie avec pendule : combien réfléchir pour ce coup,
/// selon le temps restant, l'incrément et le nombre de coups avant le
/// prochain contrôle (30 par défaut).
pub fn clock_limits(
    time_left_ms: u64,
    increment_ms: u64,
    moves_to_go: Option<u64>,
) -> SearchLimits {
    let available = time_left_ms.saturating_sub(MOVE_OVERHEAD_MS);
    let moves_to_go = moves_to_go.unwrap_or(30).max(1);
    let soft = (available / moves_to_go + increment_ms * 3 / 4).min(available / 2);
    let hard = (soft * 3).min(available * 3 / 4).max(soft);
    SearchLimits {
        soft_time: Some(Duration::from_millis(soft)),
        hard_time: Some(Duration::from_millis(hard)),
        ..SearchLimits::infinite()
    }
}

/// Informations envoyées après chaque itération (pour l'affichage en direct).
#[derive(Clone, Debug)]
pub struct SearchInfo {
    pub depth: i32,
    /// Score du point de vue du camp au trait.
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub pv: Vec<Move>,
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
}

pub struct Searcher {
    tt: TranspositionTable,
    evaluator: Arc<Evaluator>,
    /// Position à chaque profondeur du chemin en cours, et l'accumulateur du
    /// réseau de neurones correspondant (mis à jour incrémentalement).
    boards: Vec<Board>,
    accumulators: Vec<Accumulator>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[i32; 64]; 64],
    /// Clés des positions de la partie et du chemin en cours (détection des répétitions).
    path: Vec<u64>,
    nodes: u64,
    start: Instant,
    limits: SearchLimits,
    stop: Arc<AtomicBool>,
    stopped: bool,
    root_depth: i32,
    root_best: Option<Move>,
    /// Réflexion sur le temps de l'adversaire (« ponder ») : tant que ce drapeau
    /// est levé, on ignore la pendule. Quand il retombe (l'adversaire a joué le
    /// coup prévu), le chronomètre démarre.
    ponder: Option<Arc<AtomicBool>>,
    pondering: bool,
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Searcher {
        Searcher {
            tt: TranspositionTable::new(hash_mb),
            evaluator: Arc::new(Evaluator::Classic(Weights::classic())),
            boards: vec![Board::empty(); MAX_PLY],
            accumulators: vec![Accumulator::default(); MAX_PLY],
            killers: [[None; 2]; MAX_PLY],
            history: [[0; 64]; 64],
            path: Vec::new(),
            nodes: 0,
            start: Instant::now(),
            limits: SearchLimits::infinite(),
            stop: Arc::new(AtomicBool::new(false)),
            stopped: false,
            root_depth: 0,
            root_best: None,
            ponder: None,
            pondering: false,
        }
    }

    /// Les prochaines recherches réfléchiront sur le temps de l'adversaire tant
    /// que `flag` vaut vrai. `None` : recherche normale.
    pub fn set_ponder(&mut self, flag: Option<Arc<AtomicBool>>) {
        self.ponder = flag;
    }

    /// Est-on encore en train de réfléchir sur le temps de l'adversaire ?
    /// Au moment où ça s'arrête, la pendule démarre.
    fn still_pondering(&mut self) -> bool {
        if !self.pondering {
            return false;
        }
        if self
            .ponder
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return true;
        }
        self.pondering = false;
        self.start = Instant::now();
        false
    }

    /// Oublie tout ce qui a été appris pendant les parties précédentes.
    pub fn clear(&mut self) {
        self.tt.clear();
        self.history = [[0; 64]; 64];
    }

    pub fn resize_hash(&mut self, hash_mb: usize) {
        self.tt = TranspositionTable::new(hash_mb);
    }

    /// Cherche le meilleur coup. `game_history` contient les clés des positions
    /// déjà jouées (pour éviter ou rechercher les répétitions). `on_info` est
    /// appelé après chaque profondeur terminée.
    pub fn search(
        &mut self,
        board: &Board,
        game_history: &[u64],
        evaluator: Arc<Evaluator>,
        limits: SearchLimits,
        stop: Arc<AtomicBool>,
        on_info: &mut dyn FnMut(&SearchInfo),
    ) -> SearchResult {
        if !Arc::ptr_eq(&self.evaluator, &evaluator) {
            // Nouvelle évaluation : les scores en cache ne sont plus valables.
            self.tt.clear();
            self.evaluator = evaluator;
        }
        self.limits = limits;
        self.stop = stop;
        self.stopped = false;
        self.nodes = 0;
        self.start = Instant::now();
        self.pondering = self
            .ponder
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed));
        self.killers = [[None; 2]; MAX_PLY];
        for row in self.history.iter_mut() {
            for value in row.iter_mut() {
                *value /= 8;
            }
        }
        self.path.clear();
        self.path.extend_from_slice(game_history);
        if self.path.last() != Some(&board.hash) {
            self.path.push(board.hash);
        }
        self.root_best = None;

        let root_moves = legal_moves(board);
        let mut result = SearchResult {
            best_move: root_moves.first().copied(),
            score: 0,
            depth: 0,
            nodes: 0,
        };
        if root_moves.is_empty() {
            result.score = if board.in_check() { -MATE } else { 0 };
            return result;
        }

        for depth in 1..=limits.max_depth.clamp(1, MAX_DEPTH) {
            self.root_depth = depth;
            let score = self.negamax(board, depth, -INFINITY, INFINITY, 0, false);
            // Même interrompue, une itération peut avoir trouvé un meilleur coup
            // (le coup précédent est toujours examiné en premier).
            if let Some(mv) = self.root_best {
                result.best_move = Some(mv);
            }
            if self.stopped {
                break;
            }
            result.score = score;
            result.depth = depth;
            on_info(&SearchInfo {
                depth,
                score,
                nodes: self.nodes,
                elapsed: self.start.elapsed(),
                pv: self.principal_variation(board, depth),
            });
            if !self.still_pondering() {
                if let Some(soft) = limits.soft_time {
                    if self.start.elapsed() >= soft {
                        break;
                    }
                }
            }
            if let Some(max_nodes) = limits.max_nodes {
                if self.nodes >= max_nodes / 2 {
                    break;
                }
            }
            // Mat trouvé : inutile de chercher plus loin.
            if score.abs() >= MATE_THRESHOLD && depth >= 2 * (MATE - score.abs()) + 2 {
                break;
            }
        }
        result.nodes = self.nodes;
        result
    }

    fn check_limits(&mut self) {
        // La profondeur 1 va toujours au bout, pour avoir au moins un coup.
        if self.root_depth <= 1 {
            return;
        }
        if let Some(max_nodes) = self.limits.max_nodes {
            if self.nodes >= max_nodes {
                self.stopped = true;
            }
        }
        if self.nodes & 1023 == 0 {
            if self.stop.load(Ordering::Relaxed) {
                self.stopped = true;
            }
            if !self.still_pondering() {
                if let Some(hard) = self.limits.hard_time {
                    if self.start.elapsed() >= hard {
                        self.stopped = true;
                    }
                }
            }
        }
    }

    fn is_repetition(&self, board: &Board) -> bool {
        // Le dernier élément de `path` est la position actuelle. On remonte de
        // deux en deux (même camp au trait) sans dépasser le dernier coup
        // irréversible (capture ou coup de pion).
        let len = self.path.len();
        let limit = (board.halfmove_clock as usize).min(len.saturating_sub(1));
        let mut back = 2;
        while back <= limit {
            if self.path[len - 1 - back] == board.hash {
                return true;
            }
            back += 2;
        }
        false
    }

    /// Mémorise la position de cette profondeur et, avec le réseau de neurones,
    /// met à jour l'accumulateur à partir de celui de la position parente.
    fn enter(&mut self, board: &Board, ply: usize) {
        if let Evaluator::Nnue(network) = &*self.evaluator {
            if ply == 0 {
                self.accumulators[0] = network.refresh(board);
            } else {
                let mut acc = self.accumulators[ply - 1];
                network.update(&mut acc, &self.boards[ply - 1], board);
                self.accumulators[ply] = acc;
            }
        }
        self.boards[ply] = *board;
    }

    /// Évaluation de la position (du point de vue du camp au trait).
    fn static_eval(&self, board: &Board, ply: usize) -> i32 {
        match &*self.evaluator {
            Evaluator::Classic(weights) => crate::eval::evaluate(board, weights),
            Evaluator::Nnue(network) => {
                network.evaluate(&self.accumulators[ply], board.side_to_move)
            }
        }
    }

    fn negamax(
        &mut self,
        board: &Board,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
    ) -> i32 {
        self.nodes += 1;
        self.check_limits();
        if self.stopped {
            return 0;
        }
        self.enter(board, ply);

        let is_root = ply == 0;
        if !is_root
            && (board.halfmove_clock >= 100
                || self.is_repetition(board)
                || board.is_insufficient_material())
        {
            return 0;
        }
        if ply >= MAX_PLY - 1 {
            return self.static_eval(board, ply);
        }

        let in_check = board.in_check();
        // Extension d'échec : on ne s'arrête pas au milieu d'une attaque.
        let depth = if in_check { depth + 1 } else { depth };
        if depth <= 0 {
            return self.quiescence(board, alpha, beta, ply);
        }

        let mut tt_move = None;
        if let Some(entry) = self.tt.probe(board.hash) {
            tt_move = entry.best_move;
            if !is_root && entry.depth >= depth {
                let score = score_from_tt(entry.score, ply);
                match entry.bound {
                    Bound::Exact => return score,
                    Bound::Lower if score >= beta => return score,
                    Bound::Upper if score <= alpha => return score,
                    _ => {}
                }
            }
        }

        let us = board.side_to_move;
        let pv_node = beta - alpha > 1;

        // Coup nul : si même en passant son tour on reste au-dessus de bêta,
        // la position est si bonne qu'on peut couper sans chercher plus.
        if allow_null
            && !pv_node
            && !in_check
            && depth >= 3
            && beta.abs() < MATE_THRESHOLD
            && board.has_non_pawn_material(us)
            && self.static_eval(board, ply) >= beta
        {
            let reduction = if depth > 6 { 3 } else { 2 };
            let child = board.make_null_move();
            self.path.push(child.hash);
            let score = -self.negamax(
                &child,
                depth - 1 - reduction,
                -beta,
                -beta + 1,
                ply + 1,
                false,
            );
            self.path.pop();
            if self.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }

        let moves = self.ordered_moves(board, generate_moves(board, false), tt_move, ply);

        let original_alpha = alpha;
        let mut best_score = -INFINITY;
        let mut best_move = None;
        let mut legal = 0;

        for mv in moves {
            let child = board.make_move(mv);
            if child.is_square_attacked(child.king_square[us.index()], us.opponent()) {
                continue;
            }
            legal += 1;
            let quiet = board.captured_kind(mv).is_none() && mv.promotion.is_none();

            self.path.push(child.hash);
            let score = if legal == 1 {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, true)
            } else {
                // Les coups tardifs et calmes sont probablement mauvais : on les
                // cherche d'abord moins profondément, avec une fenêtre nulle.
                let reduction =
                    if depth >= 3 && legal > 3 && quiet && !in_check && !child.in_check() {
                        if legal > 10 {
                            2
                        } else {
                            1
                        }
                    } else {
                        0
                    };
                let mut score = -self.negamax(
                    &child,
                    depth - 1 - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    true,
                );
                if score > alpha && reduction > 0 {
                    score = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                }
                if score > alpha && score < beta {
                    score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, true);
                }
                score
            };
            self.path.pop();

            if self.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                if is_root {
                    self.root_best = Some(mv);
                }
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if quiet {
                    let killers = &mut self.killers[ply];
                    if killers[0] != Some(mv) {
                        killers[1] = killers[0];
                        killers[0] = Some(mv);
                    }
                    let entry = &mut self.history[mv.from][mv.to];
                    *entry = (*entry + depth * depth).min(50_000);
                }
                break;
            }
        }

        if legal == 0 {
            // Mat (le plus rapide est le meilleur) ou pat.
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        let bound = if best_score >= beta {
            Bound::Lower
        } else if best_score > original_alpha {
            Bound::Exact
        } else {
            Bound::Upper
        };
        self.tt.store(Entry {
            key: board.hash,
            best_move,
            score: score_to_tt(best_score, ply),
            depth,
            bound,
        });
        best_score
    }

    /// Quiescence : en bout de recherche, on continue les captures pour ne pas
    /// évaluer une position au milieu d'un échange.
    fn quiescence(&mut self, board: &Board, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.nodes += 1;
        self.check_limits();
        if self.stopped {
            return 0;
        }
        self.enter(board, ply);
        if ply >= MAX_PLY - 1 {
            return self.static_eval(board, ply);
        }

        let in_check = board.in_check();
        let mut best_score;
        if in_check {
            // En échec, on ne peut pas « ne rien faire » : on regarde toutes les parades.
            best_score = -MATE + ply as i32;
        } else {
            let stand_pat = self.static_eval(board, ply);
            if stand_pat >= beta {
                return stand_pat;
            }
            alpha = alpha.max(stand_pat);
            best_score = stand_pat;
        }

        let us = board.side_to_move;
        let moves = self.ordered_moves(board, generate_moves(board, !in_check), None, ply);
        for mv in moves {
            let child = board.make_move(mv);
            if child.is_square_attacked(child.king_square[us.index()], us.opponent()) {
                continue;
            }
            let score = -self.quiescence(&child, -beta, -alpha, ply + 1);
            if self.stopped {
                return 0;
            }
            if score > best_score {
                best_score = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best_score
    }

    /// Trie les coups du plus prometteur au moins prometteur : coup de la table
    /// de transposition, captures (grosse victime, petit attaquant), promotions,
    /// coups « tueurs », puis historique.
    fn ordered_moves(
        &self,
        board: &Board,
        moves: Vec<Move>,
        tt_move: Option<Move>,
        ply: usize,
    ) -> Vec<Move> {
        let mut scored: Vec<(i32, Move)> = moves
            .into_iter()
            .map(|mv| {
                let score = if Some(mv) == tt_move {
                    1_000_000
                } else if let Some(victim) = board.captured_kind(mv) {
                    let attacker = board.squares[mv.from].map_or(PieceKind::Pawn, |p| p.kind);
                    100_000 + 10 * ORDER_VALUES[victim.index()] - ORDER_VALUES[attacker.index()]
                } else if mv.promotion == Some(PieceKind::Queen) {
                    90_000
                } else if self.killers[ply][0] == Some(mv) {
                    80_000
                } else if self.killers[ply][1] == Some(mv) {
                    79_000
                } else {
                    self.history[mv.from][mv.to]
                };
                (score, mv)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().map(|(_, mv)| mv).collect()
    }

    /// Ligne principale : la suite de coups que le moteur juge la meilleure.
    fn principal_variation(&self, board: &Board, depth: i32) -> Vec<Move> {
        let mut pv = Vec::new();
        let Some(first) = self.root_best else {
            return pv;
        };
        let mut position = board.make_move(first);
        pv.push(first);
        let mut seen = vec![board.hash, position.hash];
        while pv.len() < depth.max(1) as usize {
            let Some(mv) = self.tt.probe(position.hash).and_then(|e| e.best_move) else {
                break;
            };
            if !legal_moves(&position).contains(&mv) {
                break;
            }
            position = position.make_move(mv);
            if seen.contains(&position.hash) {
                break;
            }
            seen.push(position.hash);
            pv.push(mv);
        }
        pv
    }
}

// Les scores de mat dépendent de la distance à la racine ; dans la table on
// les stocke relativement à la position elle-même.
fn score_to_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply as i32
    } else if score <= -MATE_THRESHOLD {
        score - ply as i32
    } else {
        score
    }
}

fn score_from_tt(score: i32, ply: usize) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply as i32
    } else if score <= -MATE_THRESHOLD {
        score + ply as i32
    } else {
        score
    }
}

/// Nombre de coups avant le mat (positif : on mate, négatif : on se fait mater).
pub fn mate_in(score: i32) -> Option<i32> {
    if score >= MATE_THRESHOLD {
        Some((MATE - score + 1) / 2)
    } else if score <= -MATE_THRESHOLD {
        Some(-(MATE + score) / 2)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::parse_uci_move;

    fn best_move(fen: &str, depth: i32) -> (String, i32) {
        let board = Board::from_fen(fen).unwrap();
        let mut searcher = Searcher::new(16);
        let limits = SearchLimits {
            max_depth: depth,
            ..SearchLimits::infinite()
        };
        let result = searcher.search(
            &board,
            &[],
            Arc::new(Evaluator::Classic(Weights::classic())),
            limits,
            Arc::new(AtomicBool::new(false)),
            &mut |_| {},
        );
        (result.best_move.unwrap().to_uci(), result.score)
    }

    #[test]
    fn finds_mate_in_one() {
        let (mv, score) = best_move("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", 3);
        assert_eq!(mv, "d1d8");
        assert_eq!(mate_in(score), Some(1));
    }

    #[test]
    fn finds_mate_in_two() {
        // 1. Kb6 Kb8 2. Rh8#
        let (_, score) = best_move("k7/8/2K5/8/8/8/8/7R w - - 0 1", 5);
        assert_eq!(mate_in(score), Some(2));
    }

    #[test]
    fn wins_hanging_queen() {
        let (mv, _) = best_move("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", 4);
        assert_eq!(mv, "d2d5");
    }

    #[test]
    fn best_move_is_legal() {
        let board = Board::start_position();
        let (mv, _) = best_move(crate::board::START_FEN, 5);
        assert!(parse_uci_move(&board, &mv).is_some());
    }

    #[test]
    fn avoids_stalemate_when_winning() {
        // Les Blancs ont une dame : Qg6 ou Qf7 pataient, il faut trouver Qg7#.
        let (_, score) = best_move("7k/8/5K2/8/8/8/8/6Q1 w - - 0 1", 6);
        assert!(mate_in(score).is_some_and(|m| m > 0));
    }
}

#[cfg(test)]
mod speed {
    use super::*;

    // `cargo test --release -- --ignored --nocapture compare_speed`
    #[test]
    #[ignore]
    fn compare_speed() {
        let evaluators = [
            ("formule", Arc::new(Evaluator::Classic(Weights::classic()))),
            (
                "réseau",
                Arc::new(Evaluator::Nnue(crate::nnue::Network::random(1))),
            ),
        ];
        for (name, evaluator) in evaluators {
            let board = Board::from_fen(
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            )
            .unwrap();
            let mut searcher = Searcher::new(16);
            let start = Instant::now();
            let result = searcher.search(
                &board,
                &[],
                evaluator,
                SearchLimits::nodes(3_000_000),
                Arc::new(AtomicBool::new(false)),
                &mut |_| {},
            );
            let seconds = start.elapsed().as_secs_f64();
            println!("{name}: {:.0} positions/s", result.nodes as f64 / seconds);
        }
    }
}
