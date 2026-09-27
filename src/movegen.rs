//! Génération des coups : pseudo-légaux (le roi peut rester en échec),
//! légaux, et conversion en notation UCI (e2e4) et SAN (Nf3, exd5, O-O…).

use crate::board::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Move {
    pub from: usize,
    pub to: usize,
    pub promotion: Option<PieceKind>,
}

impl Move {
    pub fn new(from: usize, to: usize) -> Move {
        Move {
            from,
            to,
            promotion: None,
        }
    }

    pub fn with_promotion(from: usize, to: usize, kind: PieceKind) -> Move {
        Move {
            from,
            to,
            promotion: Some(kind),
        }
    }

    /// Notation UCI : case de départ + case d'arrivée (+ pièce de promotion).
    pub fn to_uci(self) -> String {
        let mut s = format!("{}{}", square_name(self.from), square_name(self.to));
        if let Some(kind) = self.promotion {
            s.push(Piece::new(kind, Color::Black).to_char());
        }
        s
    }
}

const PROMOTIONS: [PieceKind; 4] = [
    PieceKind::Queen,
    PieceKind::Rook,
    PieceKind::Bishop,
    PieceKind::Knight,
];

/// Coups pseudo-légaux. Avec `captures_only`, seulement les captures et les
/// promotions en dame (utilisé par la recherche de quiescence).
pub fn generate_moves(board: &Board, captures_only: bool) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    let us = board.side_to_move;
    for from in 0..64 {
        let piece = match board.squares[from] {
            Some(p) if p.color == us => p,
            _ => continue,
        };
        match piece.kind {
            PieceKind::Pawn => pawn_moves(board, from, captures_only, &mut moves),
            PieceKind::Knight => step_moves(board, from, &KNIGHT_DELTAS, captures_only, &mut moves),
            PieceKind::Bishop => {
                slide_moves(board, from, &BISHOP_DIRECTIONS, captures_only, &mut moves)
            }
            PieceKind::Rook => {
                slide_moves(board, from, &ROOK_DIRECTIONS, captures_only, &mut moves)
            }
            PieceKind::Queen => {
                slide_moves(board, from, &ROOK_DIRECTIONS, captures_only, &mut moves);
                slide_moves(board, from, &BISHOP_DIRECTIONS, captures_only, &mut moves);
            }
            PieceKind::King => {
                step_moves(board, from, &KING_DELTAS, captures_only, &mut moves);
                if !captures_only {
                    castling_moves(board, &mut moves);
                }
            }
        }
    }
    moves
}

/// Un coup pseudo-légal laisse-t-il notre roi en sécurité ?
pub fn is_legal(board: &Board, mv: Move) -> bool {
    let after = board.make_move(mv);
    let us = board.side_to_move;
    !after.is_square_attacked(after.king_square[us.index()], us.opponent())
}

pub fn legal_moves(board: &Board) -> Vec<Move> {
    generate_moves(board, false)
        .into_iter()
        .filter(|&mv| is_legal(board, mv))
        .collect()
}

/// Retrouve un coup légal à partir de sa notation UCI (« e2e4 », « e7e8q »).
pub fn parse_uci_move(board: &Board, text: &str) -> Option<Move> {
    let text = text.trim().to_ascii_lowercase();
    legal_moves(board)
        .into_iter()
        .find(|mv| mv.to_uci() == text)
}

fn step_moves(
    board: &Board,
    from: usize,
    deltas: &[(i32, i32)],
    captures_only: bool,
    moves: &mut Vec<Move>,
) {
    let file = file_of(from) as i32;
    let rank = rank_of(from) as i32;
    for &(df, dr) in deltas {
        if let Some(to) = square_at(file + df, rank + dr) {
            match board.squares[to] {
                None if !captures_only => moves.push(Move::new(from, to)),
                Some(p) if p.color != board.side_to_move => moves.push(Move::new(from, to)),
                _ => {}
            }
        }
    }
}

fn slide_moves(
    board: &Board,
    from: usize,
    directions: &[(i32, i32)],
    captures_only: bool,
    moves: &mut Vec<Move>,
) {
    for &(df, dr) in directions {
        let mut file = file_of(from) as i32 + df;
        let mut rank = rank_of(from) as i32 + dr;
        while let Some(to) = square_at(file, rank) {
            match board.squares[to] {
                None => {
                    if !captures_only {
                        moves.push(Move::new(from, to));
                    }
                }
                Some(p) => {
                    if p.color != board.side_to_move {
                        moves.push(Move::new(from, to));
                    }
                    break;
                }
            }
            file += df;
            rank += dr;
        }
    }
}

fn pawn_moves(board: &Board, from: usize, captures_only: bool, moves: &mut Vec<Move>) {
    let us = board.side_to_move;
    let (dir, start_rank, promo_rank) = match us {
        Color::White => (1, 1, 7),
        Color::Black => (-1, 6, 0),
    };
    let file = file_of(from) as i32;
    let rank = rank_of(from) as i32;

    // Avance d'une case, puis de deux depuis la rangée de départ.
    if let Some(to) = square_at(file, rank + dir) {
        if board.squares[to].is_none() {
            if rank + dir == promo_rank {
                push_promotions(from, to, captures_only, moves);
            } else if !captures_only {
                moves.push(Move::new(from, to));
                if rank == start_rank {
                    if let Some(to2) = square_at(file, rank + 2 * dir) {
                        if board.squares[to2].is_none() {
                            moves.push(Move::new(from, to2));
                        }
                    }
                }
            }
        }
    }

    // Captures en diagonale (y compris en passant).
    for df in [-1, 1] {
        if let Some(to) = square_at(file + df, rank + dir) {
            let enemy = matches!(board.squares[to], Some(p) if p.color != us);
            if enemy || board.en_passant == Some(to) {
                if rank + dir == promo_rank {
                    push_promotions(from, to, captures_only, moves);
                } else {
                    moves.push(Move::new(from, to));
                }
            }
        }
    }
}

fn push_promotions(from: usize, to: usize, captures_only: bool, moves: &mut Vec<Move>) {
    if captures_only {
        moves.push(Move::with_promotion(from, to, PieceKind::Queen));
    } else {
        for kind in PROMOTIONS {
            moves.push(Move::with_promotion(from, to, kind));
        }
    }
}

fn castling_moves(board: &Board, moves: &mut Vec<Move>) {
    let us = board.side_to_move;
    let them = us.opponent();
    let (king_sq, kingside, queenside) = match us {
        Color::White => (4, WHITE_KINGSIDE, WHITE_QUEENSIDE),
        Color::Black => (60, BLACK_KINGSIDE, BLACK_QUEENSIDE),
    };
    if board.king_square[us.index()] != king_sq || board.castling & (kingside | queenside) == 0 {
        return;
    }
    // On ne roque pas pour sortir d'un échec.
    if board.is_square_attacked(king_sq, them) {
        return;
    }
    let rook = Some(Piece::new(PieceKind::Rook, us));
    let empty = |sq: usize| board.squares[sq].is_none();
    let safe = |sq: usize| !board.is_square_attacked(sq, them);

    if board.castling & kingside != 0
        && board.squares[king_sq + 3] == rook
        && empty(king_sq + 1)
        && empty(king_sq + 2)
        && safe(king_sq + 1)
        && safe(king_sq + 2)
    {
        moves.push(Move::new(king_sq, king_sq + 2));
    }
    if board.castling & queenside != 0
        && board.squares[king_sq - 4] == rook
        && empty(king_sq - 1)
        && empty(king_sq - 2)
        && empty(king_sq - 3)
        && safe(king_sq - 1)
        && safe(king_sq - 2)
    {
        moves.push(Move::new(king_sq, king_sq - 2));
    }
}

/// Notation algébrique standard (SAN), celle des livres d'échecs : Nf3, exd5, O-O, e8=Q+…
/// Le coup doit être légal dans `board`.
pub fn to_san(board: &Board, mv: Move) -> String {
    let piece = board.squares[mv.from].expect("aucune pièce sur la case de départ");
    let mut san = String::new();

    if piece.kind == PieceKind::King && mv.from.abs_diff(mv.to) == 2 {
        san.push_str(if mv.to > mv.from { "O-O" } else { "O-O-O" });
    } else {
        let is_capture = board.captured_kind(mv).is_some();
        if piece.kind == PieceKind::Pawn {
            if is_capture {
                san.push((b'a' + file_of(mv.from) as u8) as char);
            }
        } else {
            san.push(Piece::new(piece.kind, Color::White).to_char());
            // Désambiguïsation : une autre pièce du même type peut-elle aller sur la même case ?
            let rivals: Vec<Move> = legal_moves(board)
                .into_iter()
                .filter(|m| {
                    m.to == mv.to && m.from != mv.from && board.squares[m.from] == Some(piece)
                })
                .collect();
            if !rivals.is_empty() {
                let same_file = rivals.iter().any(|m| file_of(m.from) == file_of(mv.from));
                let same_rank = rivals.iter().any(|m| rank_of(m.from) == rank_of(mv.from));
                let name = square_name(mv.from);
                if !same_file {
                    san.push_str(&name[..1]);
                } else if !same_rank {
                    san.push_str(&name[1..]);
                } else {
                    san.push_str(&name);
                }
            }
        }
        if is_capture {
            san.push('x');
        }
        san.push_str(&square_name(mv.to));
        if let Some(kind) = mv.promotion {
            san.push('=');
            san.push(Piece::new(kind, Color::White).to_char());
        }
    }

    let after = board.make_move(mv);
    if after.in_check() {
        san.push(if legal_moves(&after).is_empty() {
            '#'
        } else {
            '+'
        });
    }
    san
}

#[cfg(test)]
mod tests {
    use super::*;

    fn san_of(fen: &str, uci: &str) -> String {
        let board = Board::from_fen(fen).unwrap();
        to_san(&board, parse_uci_move(&board, uci).expect("coup légal"))
    }

    #[test]
    fn san_notation() {
        assert_eq!(san_of(START_FEN, "g1f3"), "Nf3");
        assert_eq!(san_of(START_FEN, "e2e4"), "e4");
        assert_eq!(
            san_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            "O-O"
        );
        assert_eq!(
            san_of("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
            "O-O-O"
        );
        // Deux cavaliers peuvent aller en d2 : il faut préciser la colonne.
        assert_eq!(san_of("4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1", "b1d2"), "Nbd2");
        // Deux tours sur la même colonne : il faut préciser la rangée.
        assert_eq!(san_of("4k3/R7/8/8/8/8/R7/4K3 w - - 0 1", "a2a5"), "R2a5");
        assert_eq!(san_of("4k3/3P4/8/8/8/8/8/4K3 w - - 0 1", "d7d8q"), "d8=Q+");
        assert_eq!(
            san_of(
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
                "e5f6"
            ),
            "exf6"
        );
        // Mat du berger.
        assert_eq!(
            san_of(
                "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4",
                "f3f7"
            ),
            "Qxf7#"
        );
    }

    #[test]
    fn cannot_castle_through_check() {
        // La tour noire en f8 contrôle f1 : le petit roque blanc est interdit.
        let board = Board::from_fen("5r1k/8/8/8/8/8/8/4K2R w K - 0 1").unwrap();
        assert!(parse_uci_move(&board, "e1g1").is_none());
    }
}
