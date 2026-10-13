//! Static evaluation of a position, from one player's point of view.

use crate::animal::{POINTS, animal, owner};
use crate::state::State;

/// Weight of a queue card's points by its position (0 = front, next to the bar). Tunable.
pub const POS_WEIGHT: [f32; 4] = [0.6, 0.45, 0.25, 0.1];

/// Terminal bonus: any win outranks any point margin.
pub const WIN_VALUE: f32 = 1000.0;

/// Value of `state` for player `me`. Terminal: `WIN_VALUE * sign(margin) + margin`. Otherwise the bar margin plus
/// the queue cards' points weighted by position, mine positive and theirs negative.
///
/// Exactly antisymmetric (`evaluate(s, 0) == -evaluate(s, 1)`), which negamax relies on.
#[inline]
pub fn evaluate(state: &State, me: u8) -> f32 {
    let margin = state.scores[0] as f32 - state.scores[1] as f32;
    let v0 = if state.is_over() {
        WIN_VALUE * sign(margin) + margin
    } else {
        let mut queue = 0.0;
        for (i, &c) in state.queue().iter().enumerate() {
            let points = POS_WEIGHT[i] * POINTS[animal(c) as usize] as f32;
            queue += if owner(c) == 0 { points } else { -points };
        }
        margin + queue
    };
    if me == 0 { v0 } else { -v0 }
}

#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animal::*;

    #[test]
    fn terminal_and_horizon_values() {
        let mut s = State::new(0, [0, 0], [&[], &[]]);
        s.scores = [10, 7];
        assert!(s.is_over());
        assert_eq!(evaluate(&s, 0), 1003.0);
        assert_eq!(evaluate(&s, 1), -1003.0);
        s.scores = [7, 7];
        assert_eq!(evaluate(&s, 1), 0.0);

        let mut s = State::new(0, [ALL_ANIMALS, ALL_ANIMALS], [&[], &[]]);
        s.scores = [4, 6];
        s.queue[..2].copy_from_slice(&[card(0, ZEBRA), card(1, LION)]);
        s.queue_len = 2;
        // -2 + 0.6 * 4 - 0.45 * 2
        assert!((evaluate(&s, 0) - (-2.0 + 2.4 - 0.9)).abs() < 1e-6);
        assert_eq!(evaluate(&s, 1), -evaluate(&s, 0));
    }
}
