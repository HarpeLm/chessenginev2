//! Table de transposition : un cache des positions déjà analysées.
//! La même position peut être atteinte par des ordres de coups différents ;
//! on évite ainsi de la recalculer.

use crate::movegen::Move;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// Score exact.
    Exact,
    /// Le vrai score est au moins celui-ci (coupure bêta).
    Lower,
    /// Le vrai score est au plus celui-ci (aucun coup n'a dépassé alpha).
    Upper,
}

#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub key: u64,
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: i32,
    pub bound: Bound,
}

pub struct TranspositionTable {
    entries: Vec<Option<Entry>>,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> TranspositionTable {
        let entry_size = std::mem::size_of::<Option<Entry>>();
        let wanted = size_mb.max(1) * 1024 * 1024 / entry_size;
        // Une puissance de deux permet de calculer l'indice avec un simple « et » binaire.
        let mut count = 1;
        while count * 2 <= wanted {
            count *= 2;
        }
        TranspositionTable {
            entries: vec![None; count],
        }
    }

    fn index(&self, key: u64) -> usize {
        (key as usize) & (self.entries.len() - 1)
    }

    pub fn probe(&self, key: u64) -> Option<Entry> {
        self.entries[self.index(key)].filter(|entry| entry.key == key)
    }

    pub fn store(&mut self, entry: Entry) {
        let index = self.index(entry.key);
        self.entries[index] = Some(entry);
    }

    pub fn clear(&mut self) {
        self.entries.fill(None);
    }
}
