//! Perft : compte toutes les positions atteignables à une profondeur donnée.
//! Comparé aux valeurs de référence connues, c'est LE test qui prouve que la
//! génération des coups est correcte.

use std::time::Instant;

use crate::board::Board;
use crate::movegen::legal_moves;

pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = legal_moves(board);
    if depth == 1 {
        return moves.len() as u64;
    }
    moves
        .iter()
        .map(|&mv| perft(&board.make_move(mv), depth - 1))
        .sum()
}

/// Affiche le nombre de positions sous chaque coup (pratique pour trouver un bug
/// en comparant avec un autre moteur), puis le total.
pub fn run(board: &Board, depth: u32) {
    let start = Instant::now();
    let mut total = 0;
    for mv in legal_moves(board) {
        let count = if depth == 0 {
            1
        } else {
            perft(&board.make_move(mv), depth - 1)
        };
        println!("{}: {}", mv.to_uci(), count);
        total += count;
    }
    let seconds = start.elapsed().as_secs_f64();
    println!();
    println!("Total     : {total}");
    println!("Temps     : {seconds:.2} s");
    println!(
        "Vitesse   : {:.0} positions/s",
        total as f64 / seconds.max(1e-9)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // Positions de référence : https://www.chessprogramming.org/Perft_Results
    fn check(fen: &str, expected: &[u64]) {
        let board = Board::from_fen(fen).unwrap();
        for (i, &count) in expected.iter().enumerate() {
            let depth = i as u32 + 1;
            assert_eq!(perft(&board, depth), count, "profondeur {depth}, FEN {fen}");
        }
    }

    #[test]
    fn perft_start_position() {
        check(crate::board::START_FEN, &[20, 400, 8_902, 197_281]);
    }

    #[test]
    fn perft_kiwipete() {
        check(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2_039, 97_862],
        );
    }

    #[test]
    fn perft_position_3() {
        check(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[14, 191, 2_812, 43_238],
        );
    }

    #[test]
    fn perft_position_4() {
        check(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9_467],
        );
    }

    #[test]
    fn perft_position_5() {
        check(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1_486, 62_379],
        );
    }

    #[test]
    fn perft_position_6() {
        check(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2_079, 89_890],
        );
    }

    // Plus long : `cargo test --release -- --ignored`
    #[test]
    #[ignore]
    fn perft_deep() {
        check(
            crate::board::START_FEN,
            &[20, 400, 8_902, 197_281, 4_865_609],
        );
        check(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            &[48, 2_039, 97_862, 4_085_603],
        );
        check(
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            &[14, 191, 2_812, 43_238, 674_624],
        );
        check(
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            &[6, 264, 9_467, 422_333],
        );
        check(
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            &[44, 1_486, 62_379, 2_103_487],
        );
        check(
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            &[46, 2_079, 89_890, 3_894_594],
        );
    }
}
