//! The fixed action layout shared with `bbar/actions.py`. Parameters are queue positions before the move.
//!
//! ```text
//! 0           skunk
//! 1..4        parrot [t]
//! 5           kangaroo []           (queue empty)
//! 6, 7        kangaroo [1], [2]
//! 8           monkey
//! 9..12       chameleon [t]         copies a parameterless animal at t
//! 13..28      chameleon [t, x]      13 + 4t + x; copies the parrot at t (x = target) or kangaroo at t (x = hop)
//! 29..35      seal, zebra, giraffe, snake, crocodile, hippo, lion
//! ```

use crate::animal::*;

pub const NUM_ACTIONS: usize = 36;

/// A decoded action: the animal played and its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub animal: u8,
    params: [u8; 2],
    len: u8,
}

impl Move {
    const fn new(animal: u8, params: &[u8]) -> Move {
        match *params {
            [] => Move { animal, params: [0, 0], len: 0 },
            [a] => Move { animal, params: [a, 0], len: 1 },
            [a, b] => Move { animal, params: [a, b], len: 2 },
            _ => panic!("at most two parameters"),
        }
    }

    pub fn params(&self) -> &[u8] {
        &self.params[..self.len as usize]
    }

    /// Parameters padded with zeros: `[target or hop, copied parameter]`.
    #[inline]
    pub(crate) fn raw_params(&self) -> [u8; 2] {
        self.params
    }
}

const fn decode_const(action: u8) -> Move {
    match action {
        0 => Move::new(SKUNK, &[]),
        1..=4 => Move::new(PARROT, &[action - 1]),
        5 => Move::new(KANGAROO, &[]),
        6 | 7 => Move::new(KANGAROO, &[action - 5]),
        8 => Move::new(MONKEY, &[]),
        9..=12 => Move::new(CHAMELEON, &[action - 9]),
        13..=28 => Move::new(CHAMELEON, &[(action - 13) / 4, (action - 13) % 4]),
        29..=35 => Move::new(action - 24, &[]),
        _ => panic!("action out of range"),
    }
}

const MOVES: [Move; NUM_ACTIONS] = {
    let mut moves = [Move::new(0, &[]); NUM_ACTIONS];
    let mut a = 0;
    while a < NUM_ACTIONS {
        moves[a] = decode_const(a as u8);
        a += 1;
    }
    moves
};

#[inline]
pub fn decode(action: u8) -> Move {
    MOVES[action as usize]
}

/// The action index of `animal` with `params`, if that shape exists in the layout (legality is not checked).
pub fn encode(animal: u8, params: &[u8]) -> Option<u8> {
    let action = match (animal, params) {
        (PARROT, &[t]) if t < 4 => 1 + t,
        (KANGAROO, &[]) => 5,
        (KANGAROO, &[hop]) if (1..=2).contains(&hop) => 5 + hop,
        (CHAMELEON, &[t]) if t < 4 => 9 + t,
        (CHAMELEON, &[t, x]) if t < 4 && x < 4 => 13 + 4 * t + x,
        (SKUNK, &[]) => 0,
        (MONKEY, &[]) => 8,
        (SEAL..=LION, &[]) => animal + 24,
        _ => return None,
    };
    Some(action)
}

/// A set of action indices as a bitmask; iterates in increasing order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ActionSet(pub u64);

impl ActionSet {
    #[inline]
    pub fn contains(self, action: u8) -> bool {
        (action as usize) < NUM_ACTIONS && self.0 >> action & 1 != 0
    }

    #[inline]
    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The `k`-th smallest action in the set. Panics if `k >= len()`.
    #[inline]
    pub fn nth_action(self, k: usize) -> u8 {
        let mut bits = self.0;
        for _ in 0..k {
            bits &= bits - 1;
        }
        assert!(bits != 0, "index out of range");
        bits.trailing_zeros() as u8
    }
}

impl Iterator for ActionSet {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            return None;
        }
        let action = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(action)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.count_ones() as usize;
        (n, Some(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_inverts_decode() {
        for a in 0..NUM_ACTIONS as u8 {
            let m = decode(a);
            assert_eq!(encode(m.animal, m.params()), Some(a));
        }
    }
}
