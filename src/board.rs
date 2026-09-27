//! Représentation de l'échiquier : pièces, cases, lecture/écriture FEN,
//! détection des attaques et exécution d'un coup.

use crate::movegen::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opponent(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// 0 pour les Blancs, 1 pour les Noirs (pratique pour indexer des tableaux).
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

impl PieceKind {
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

impl Piece {
    pub fn new(kind: PieceKind, color: Color) -> Piece {
        Piece { kind, color }
    }

    /// Lettre de la pièce au format FEN : majuscule = Blancs, minuscule = Noirs.
    pub fn to_char(self) -> char {
        let c = match self.kind {
            PieceKind::Pawn => 'p',
            PieceKind::Knight => 'n',
            PieceKind::Bishop => 'b',
            PieceKind::Rook => 'r',
            PieceKind::Queen => 'q',
            PieceKind::King => 'k',
        };
        match self.color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let kind = match c.to_ascii_lowercase() {
            'p' => PieceKind::Pawn,
            'n' => PieceKind::Knight,
            'b' => PieceKind::Bishop,
            'r' => PieceKind::Rook,
            'q' => PieceKind::Queen,
            'k' => PieceKind::King,
            _ => return None,
        };
        Some(Piece::new(kind, color))
    }

    /// Numéro unique entre 0 et 11, utilisé pour le hachage Zobrist.
    pub fn index(self) -> usize {
        self.color.index() * 6 + self.kind.index()
    }
}

// Droits de roque, stockés comme des bits dans un u8.
pub const WHITE_KINGSIDE: u8 = 1;
pub const WHITE_QUEENSIDE: u8 = 2;
pub const BLACK_KINGSIDE: u8 = 4;
pub const BLACK_QUEENSIDE: u8 = 8;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Déplacements (colonne, rangée) de chaque type de pièce.
pub const KNIGHT_DELTAS: [(i32, i32); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
pub const KING_DELTAS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
pub const ROOK_DIRECTIONS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
pub const BISHOP_DIRECTIONS: [(i32, i32); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

// Les cases sont numérotées de a1 = 0 à h8 = 63, rangée par rangée.
pub fn square_index(file: usize, rank: usize) -> usize {
    rank * 8 + file
}

pub fn file_of(sq: usize) -> usize {
    sq % 8
}

pub fn rank_of(sq: usize) -> usize {
    sq / 8
}

/// Renvoie la case (colonne, rangée) si elle est sur l'échiquier, sinon None.
pub fn square_at(file: i32, rank: i32) -> Option<usize> {
    if (0..8).contains(&file) && (0..8).contains(&rank) {
        Some((rank * 8 + file) as usize)
    } else {
        None
    }
}

pub fn square_name(sq: usize) -> String {
    let file = (b'a' + file_of(sq) as u8) as char;
    let rank = (b'1' + rank_of(sq) as u8) as char;
    format!("{}{}", file, rank)
}

pub fn parse_square(name: &str) -> Option<usize> {
    let mut chars = name.chars();
    let file = chars.next()?;
    let rank = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    if !('a'..='h').contains(&file) || !('1'..='8').contains(&rank) {
        return None;
    }
    let file = (file as u8 - b'a') as usize;
    let rank = (rank as u8 - b'1') as usize;
    Some(square_index(file, rank))
}

// ---------------------------------------------------------------------------
// Hachage Zobrist : chaque position reçoit un nombre de 64 bits (presque)
// unique. Il sert de clé pour la table de transposition et pour détecter
// les répétitions.
// ---------------------------------------------------------------------------

const ZOBRIST_SIDE: usize = 12 * 64;
const ZOBRIST_CASTLING: usize = ZOBRIST_SIDE + 1;
const ZOBRIST_EN_PASSANT: usize = ZOBRIST_CASTLING + 4;
const ZOBRIST_SIZE: usize = ZOBRIST_EN_PASSANT + 8;

const fn splitmix64(state: u64) -> (u64, u64) {
    let state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state, z ^ (z >> 31))
}

// Nombres pseudo-aléatoires calculés à la compilation.
static ZOBRIST: [u64; ZOBRIST_SIZE] = {
    let mut table = [0u64; ZOBRIST_SIZE];
    let mut state = 0x1234_5678_9ABC_DEF0u64;
    let mut i = 0;
    while i < ZOBRIST_SIZE {
        let (next_state, value) = splitmix64(state);
        state = next_state;
        table[i] = value;
        i += 1;
    }
    table
};

fn zobrist_piece(piece: Piece, sq: usize) -> u64 {
    ZOBRIST[piece.index() * 64 + sq]
}

fn zobrist_castling(rights: u8) -> u64 {
    let mut hash = 0;
    for i in 0..4 {
        if rights & (1 << i) != 0 {
            hash ^= ZOBRIST[ZOBRIST_CASTLING + i];
        }
    }
    hash
}

fn zobrist_en_passant(sq: usize) -> u64 {
    ZOBRIST[ZOBRIST_EN_PASSANT + file_of(sq)]
}

// Quand une pièce part de (ou arrive sur) une de ces cases, on retire les
// droits de roque correspondants : roi ou tour qui bouge, tour capturée.
static CASTLING_MASK: [u8; 64] = {
    let mut mask = [15u8; 64];
    mask[0] = 15 & !WHITE_QUEENSIDE; // a1
    mask[7] = 15 & !WHITE_KINGSIDE; // h1
    mask[4] = 15 & !(WHITE_KINGSIDE | WHITE_QUEENSIDE); // e1
    mask[56] = 15 & !BLACK_QUEENSIDE; // a8
    mask[63] = 15 & !BLACK_KINGSIDE; // h8
    mask[60] = 15 & !(BLACK_KINGSIDE | BLACK_QUEENSIDE); // e8
    mask
};

#[derive(Clone, Copy, Debug)]
pub struct Board {
    pub squares: [Option<Piece>; 64],
    pub side_to_move: Color,
    pub castling: u8,
    pub en_passant: Option<usize>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
    /// Case du roi de chaque camp, gardée à jour pour tester l'échec rapidement.
    pub king_square: [usize; 2],
    pub hash: u64,
}

impl Board {
    pub fn empty() -> Board {
        Board {
            squares: [None; 64],
            side_to_move: Color::White,
            castling: 0,
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            king_square: [0; 2],
            hash: 0,
        }
    }

    pub fn start_position() -> Board {
        Board::from_fen(START_FEN).expect("le FEN de départ est valide")
    }

    pub fn from_fen(fen: &str) -> Result<Board, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err(format!("FEN incomplet : {fen}"));
        }

        let mut board = Board::empty();

        let rows: Vec<&str> = parts[0].split('/').collect();
        if rows.len() != 8 {
            return Err(format!("il faut 8 rangées dans le FEN : {fen}"));
        }
        for (i, row) in rows.iter().enumerate() {
            let rank = 7 - i;
            let mut file = 0;
            for c in row.chars() {
                if let Some(n) = c.to_digit(10) {
                    file += n as usize;
                    if file > 8 {
                        return Err(format!("rangée {} trop longue", rank + 1));
                    }
                } else {
                    let piece = Piece::from_char(c).ok_or(format!("pièce inconnue '{c}'"))?;
                    if file >= 8 {
                        return Err(format!("rangée {} trop longue", rank + 1));
                    }
                    board.squares[square_index(file, rank)] = Some(piece);
                    file += 1;
                }
            }
            if file != 8 {
                return Err(format!("rangée {} incomplète", rank + 1));
            }
        }

        board.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            other => return Err(format!("trait invalide '{other}'")),
        };

        for c in parts[2].chars() {
            match c {
                'K' => board.castling |= WHITE_KINGSIDE,
                'Q' => board.castling |= WHITE_QUEENSIDE,
                'k' => board.castling |= BLACK_KINGSIDE,
                'q' => board.castling |= BLACK_QUEENSIDE,
                '-' => {}
                other => return Err(format!("droit de roque invalide '{other}'")),
            }
        }

        board.en_passant = match parts[3] {
            "-" => None,
            name => Some(parse_square(name).ok_or(format!("case en passant invalide '{name}'"))?),
        };

        board.halfmove_clock = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        board.fullmove_number = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);

        for color in [Color::White, Color::Black] {
            let king = Some(Piece::new(PieceKind::King, color));
            let kings: Vec<usize> = (0..64).filter(|&sq| board.squares[sq] == king).collect();
            if kings.len() != 1 {
                return Err(format!("il faut exactement un roi {:?}", color));
            }
            board.king_square[color.index()] = kings[0];
        }

        board.hash = board.compute_hash();
        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                match self.squares[square_index(file, rank)] {
                    Some(piece) => {
                        if empty > 0 {
                            fen.push_str(&empty.to_string());
                            empty = 0;
                        }
                        fen.push(piece.to_char());
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side_to_move == Color::White {
            'w'
        } else {
            'b'
        });

        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            for (flag, c) in [
                (WHITE_KINGSIDE, 'K'),
                (WHITE_QUEENSIDE, 'Q'),
                (BLACK_KINGSIDE, 'k'),
                (BLACK_QUEENSIDE, 'q'),
            ] {
                if self.castling & flag != 0 {
                    fen.push(c);
                }
            }
        }

        fen.push(' ');
        match self.en_passant {
            Some(sq) => fen.push_str(&square_name(sq)),
            None => fen.push('-'),
        }

        fen.push_str(&format!(
            " {} {}",
            self.halfmove_clock, self.fullmove_number
        ));
        fen
    }

    /// Recalcule la clé Zobrist depuis zéro (make_move la met à jour
    /// incrémentalement, ceci sert à l'initialisation et aux tests).
    pub fn compute_hash(&self) -> u64 {
        let mut hash = 0;
        for sq in 0..64 {
            if let Some(piece) = self.squares[sq] {
                hash ^= zobrist_piece(piece, sq);
            }
        }
        if self.side_to_move == Color::Black {
            hash ^= ZOBRIST[ZOBRIST_SIDE];
        }
        hash ^= zobrist_castling(self.castling);
        if let Some(sq) = self.en_passant {
            hash ^= zobrist_en_passant(sq);
        }
        hash
    }

    fn put_piece(&mut self, sq: usize, piece: Piece) {
        self.squares[sq] = Some(piece);
        self.hash ^= zobrist_piece(piece, sq);
    }

    fn remove_piece(&mut self, sq: usize) {
        if let Some(piece) = self.squares[sq].take() {
            self.hash ^= zobrist_piece(piece, sq);
        }
    }

    /// La case `sq` est-elle attaquée par une pièce de couleur `by` ?
    pub fn is_square_attacked(&self, sq: usize, by: Color) -> bool {
        let file = file_of(sq) as i32;
        let rank = rank_of(sq) as i32;

        // Pions : un pion blanc attaque en diagonale vers le haut, donc il se
        // trouve une rangée en dessous de la case attaquée.
        let pawn_rank = if by == Color::White {
            rank - 1
        } else {
            rank + 1
        };
        let pawn = Some(Piece::new(PieceKind::Pawn, by));
        for df in [-1, 1] {
            if let Some(from) = square_at(file + df, pawn_rank) {
                if self.squares[from] == pawn {
                    return true;
                }
            }
        }

        let knight = Some(Piece::new(PieceKind::Knight, by));
        for (df, dr) in KNIGHT_DELTAS {
            if let Some(from) = square_at(file + df, rank + dr) {
                if self.squares[from] == knight {
                    return true;
                }
            }
        }

        let king = Some(Piece::new(PieceKind::King, by));
        for (df, dr) in KING_DELTAS {
            if let Some(from) = square_at(file + df, rank + dr) {
                if self.squares[from] == king {
                    return true;
                }
            }
        }

        // Pièces à longue portée : on avance dans chaque direction jusqu'à
        // rencontrer une pièce.
        let sliders = [
            (ROOK_DIRECTIONS, PieceKind::Rook),
            (BISHOP_DIRECTIONS, PieceKind::Bishop),
        ];
        for (directions, kind) in sliders {
            for (df, dr) in directions {
                let (mut f, mut r) = (file + df, rank + dr);
                while let Some(from) = square_at(f, r) {
                    if let Some(piece) = self.squares[from] {
                        if piece.color == by
                            && (piece.kind == kind || piece.kind == PieceKind::Queen)
                        {
                            return true;
                        }
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }

        false
    }

    /// Le camp au trait est-il en échec ?
    pub fn in_check(&self) -> bool {
        let us = self.side_to_move;
        self.is_square_attacked(self.king_square[us.index()], us.opponent())
    }

    /// Type de la pièce capturée par ce coup (en tenant compte de la prise en passant).
    pub fn captured_kind(&self, mv: Move) -> Option<PieceKind> {
        match self.squares[mv.to] {
            Some(piece) => Some(piece.kind),
            None => {
                let is_pawn = matches!(self.squares[mv.from], Some(p) if p.kind == PieceKind::Pawn);
                if is_pawn && Some(mv.to) == self.en_passant {
                    Some(PieceKind::Pawn)
                } else {
                    None
                }
            }
        }
    }

    /// Joue un coup et renvoie la nouvelle position (l'ancienne n'est pas
    /// modifiée : c'est la technique « copy-make »). Le coup doit être
    /// pseudo-légal ; le roi peut rester en échec, c'est à l'appelant de
    /// vérifier.
    pub fn make_move(&self, mv: Move) -> Board {
        let mut board = *self;
        let us = self.side_to_move;
        let piece = self.squares[mv.from].expect("aucune pièce sur la case de départ");

        board.hash ^= zobrist_castling(self.castling);
        if let Some(sq) = self.en_passant {
            board.hash ^= zobrist_en_passant(sq);
        }
        board.en_passant = None;

        let mut is_capture = self.squares[mv.to].is_some();
        if piece.kind == PieceKind::Pawn && Some(mv.to) == self.en_passant && !is_capture {
            // Prise en passant : le pion capturé n'est pas sur la case d'arrivée.
            let captured_sq = if us == Color::White {
                mv.to - 8
            } else {
                mv.to + 8
            };
            board.remove_piece(captured_sq);
            is_capture = true;
        } else {
            board.remove_piece(mv.to);
        }

        board.remove_piece(mv.from);
        let placed = match mv.promotion {
            Some(kind) => Piece::new(kind, us),
            None => piece,
        };
        board.put_piece(mv.to, placed);

        if piece.kind == PieceKind::King {
            board.king_square[us.index()] = mv.to;
            // Roque : le roi bouge de deux cases, on déplace aussi la tour.
            if mv.to == mv.from + 2 {
                let rook = Piece::new(PieceKind::Rook, us);
                board.remove_piece(mv.from + 3);
                board.put_piece(mv.from + 1, rook);
            } else if mv.from == mv.to + 2 {
                let rook = Piece::new(PieceKind::Rook, us);
                board.remove_piece(mv.from - 4);
                board.put_piece(mv.from - 1, rook);
            }
        }

        if piece.kind == PieceKind::Pawn && mv.from.abs_diff(mv.to) == 16 {
            let sq = (mv.from + mv.to) / 2;
            board.en_passant = Some(sq);
            board.hash ^= zobrist_en_passant(sq);
        }

        board.castling &= CASTLING_MASK[mv.from] & CASTLING_MASK[mv.to];
        board.hash ^= zobrist_castling(board.castling);

        if piece.kind == PieceKind::Pawn || is_capture {
            board.halfmove_clock = 0;
        } else {
            board.halfmove_clock += 1;
        }
        if us == Color::Black {
            board.fullmove_number += 1;
        }

        board.side_to_move = us.opponent();
        board.hash ^= ZOBRIST[ZOBRIST_SIDE];
        board
    }

    /// « Passe » son tour. Illégal aux échecs, mais très utile dans la
    /// recherche (null move pruning).
    pub fn make_null_move(&self) -> Board {
        let mut board = *self;
        if let Some(sq) = self.en_passant {
            board.hash ^= zobrist_en_passant(sq);
        }
        board.en_passant = None;
        // Remis à zéro pour que la détection des répétitions ne traverse
        // pas un coup nul.
        board.halfmove_clock = 0;
        board.side_to_move = self.side_to_move.opponent();
        board.hash ^= ZOBRIST[ZOBRIST_SIDE];
        board
    }

    /// Le camp possède-t-il autre chose que des pions et son roi ?
    pub fn has_non_pawn_material(&self, color: Color) -> bool {
        self.squares
            .iter()
            .flatten()
            .any(|p| p.color == color && p.kind != PieceKind::Pawn && p.kind != PieceKind::King)
    }

    /// Aucun camp ne peut mater : roi contre roi, ou roi + une pièce mineure contre roi.
    pub fn is_insufficient_material(&self) -> bool {
        let mut minors = 0;
        for piece in self.squares.iter().flatten() {
            match piece.kind {
                PieceKind::King => {}
                PieceKind::Knight | PieceKind::Bishop => minors += 1,
                _ => return false,
            }
        }
        minors <= 1
    }

    pub fn print(&self) {
        for rank in (0..8).rev() {
            print!("{} ", rank + 1);
            for file in 0..8 {
                let c = match self.squares[square_index(file, rank)] {
                    Some(piece) => piece.to_char(),
                    None => '.',
                };
                print!("{} ", c);
            }
            println!();
        }
        println!("  a b c d e f g h");
        println!(
            "Trait : {}",
            if self.side_to_move == Color::White {
                "Blancs"
            } else {
                "Noirs"
            }
        );
        println!("FEN   : {}", self.to_fen());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::legal_moves;

    #[test]
    fn square_conversions() {
        assert_eq!(square_index(0, 0), 0);
        assert_eq!(square_name(63), "h8");
        assert_eq!(parse_square("e4"), Some(28));
        assert_eq!(parse_square("i9"), None);
        assert_eq!(parse_square("e44"), None);
    }

    #[test]
    fn start_position_has_kings() {
        let board = Board::start_position();
        let e1 = parse_square("e1").unwrap();
        let e8 = parse_square("e8").unwrap();
        assert_eq!(
            board.squares[e1],
            Some(Piece::new(PieceKind::King, Color::White))
        );
        assert_eq!(
            board.squares[e8],
            Some(Piece::new(PieceKind::King, Color::Black))
        );
        assert!(board.squares[parse_square("d4").unwrap()].is_none());
    }

    #[test]
    fn fen_round_trip() {
        let fens = [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "8/8/8/8/8/8/8/K6k b - - 42 99",
        ];
        for fen in fens {
            assert_eq!(Board::from_fen(fen).unwrap().to_fen(), fen);
        }
    }

    #[test]
    fn invalid_fens_are_rejected() {
        assert!(Board::from_fen("").is_err());
        assert!(Board::from_fen("8/8/8/8/8/8/8/8 w - -").is_err()); // pas de rois
        assert!(
            Board::from_fen("rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").is_err()
        );
        assert!(
            Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1").is_err()
        );
    }

    #[test]
    fn incremental_hash_matches_full_hash() {
        fn walk(board: &Board, depth: u32) {
            assert_eq!(board.hash, board.compute_hash(), "FEN : {}", board.to_fen());
            if depth == 0 {
                return;
            }
            for mv in legal_moves(board) {
                walk(&board.make_move(mv), depth - 1);
            }
        }
        let kiwipete =
            Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        walk(&kiwipete, 3);
        walk(&Board::start_position(), 3);
    }
}
