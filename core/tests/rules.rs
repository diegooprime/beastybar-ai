//! Rule tests for the subtle cases. Positions are written as (animal, owner), front (bar side) first.

use bbcore::animal::*;
use bbcore::{QUEUE_CAP, Rng, State, encode, random_playout};

/// Player 0 to move, holding only `hand`, with `queue` on the table. Player 1 holds every animal not in
/// the queue, so the game is not over.
fn position(queue: &[(u8, u8)], hand: &[u8]) -> State {
    let mut s = State::new(0, [0, ALL_ANIMALS], [&[], &[]]);
    for &a in hand {
        s.hands[0] |= 1 << a;
    }
    for &(a, owner) in queue {
        s.queue[s.queue_len as usize] = card(owner, a);
        s.queue_len += 1;
        if owner == 1 {
            s.hands[1] &= !(1 << a);
        }
    }
    s
}

/// Player 0 plays `animal` with `params` onto `queue`.
fn play(queue: &[(u8, u8)], animal: u8, params: &[u8]) -> State {
    let mut s = position(queue, &[animal]);
    let action = encode(animal, params).expect("action exists");
    assert!(s.legal_actions().contains(action), "{animal} {params:?} illegal on {queue:?}");
    s.apply(action);
    s
}

fn cards(cs: &[u8]) -> Vec<(u8, u8)> {
    cs.iter().map(|&c| (animal(c), owner(c))).collect()
}

fn legal(queue: &[(u8, u8)], hand: &[u8]) -> Vec<u8> {
    position(queue, hand).legal_actions().collect()
}

#[test]
fn parameters_follow_the_queue() {
    // Parrot needs a target; kangaroo hops at most the queue length, and has no hop on an empty queue.
    assert_eq!(legal(&[], &[SKUNK, PARROT, KANGAROO]), [0, 5]);
    assert_eq!(legal(&[(SEAL, 1)], &[PARROT, KANGAROO]), [1, 6]);
    assert_eq!(legal(&[(SEAL, 1), (ZEBRA, 1)], &[PARROT, KANGAROO]), [1, 2, 6, 7]);
}

#[test]
fn chameleon_needs_a_non_chameleon_to_copy() {
    assert_eq!(legal(&[], &[CHAMELEON, ZEBRA]), [30]);
    assert_eq!(legal(&[(CHAMELEON, 1)], &[CHAMELEON, ZEBRA]), [30]);
    // Copying the seal at 1 (9 + 1), the parrot at 0 with any target (13 + x), never the chameleon at 2.
    assert_eq!(legal(&[(PARROT, 1), (SEAL, 1), (CHAMELEON, 1)], &[CHAMELEON]), [10, 13, 14, 15]);
    // Copying a kangaroo takes a hop of at most the queue length.
    assert_eq!(legal(&[(KANGAROO, 1)], &[CHAMELEON]), [14]);
}

#[test]
fn chameleon_takes_the_copied_parameter() {
    let s = play(&[(LION, 1), (PARROT, 1)], CHAMELEON, &[1, 0]);
    assert_eq!(cards(s.queue()), [(PARROT, 1), (CHAMELEON, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(LION, 1)]);

    let s = play(&[(SEAL, 1), (KANGAROO, 1), (ZEBRA, 1)], CHAMELEON, &[1, 2]);
    assert_eq!(cards(s.queue()), [(SEAL, 1), (CHAMELEON, 0), (KANGAROO, 1), (ZEBRA, 1)]);
}

#[test]
fn chameleon_copying_lion_is_bounced() {
    let s = play(&[(MONKEY, 1), (LION, 1)], CHAMELEON, &[1]);
    assert_eq!(cards(s.queue()), [(MONKEY, 1), (LION, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(CHAMELEON, 0)]);
}

#[test]
fn chameleon_copying_skunk_is_a_skunk() {
    // The acting chameleon does not count among the strengths and is not bounced...
    let s = play(&[(SKUNK, 1), (LION, 1)], CHAMELEON, &[0]);
    assert_eq!(cards(s.queue()), [(SKUNK, 1), (CHAMELEON, 0)]);
    // ...but another chameleon in the queue is a strength-5 chameleon.
    let s = play(&[(SKUNK, 1), (CHAMELEON, 1), (LION, 1)], CHAMELEON, &[0]);
    assert_eq!(cards(s.queue()), [(SKUNK, 1), (CHAMELEON, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(CHAMELEON, 1), (LION, 1)]);
}

#[test]
fn chameleon_copying_snake_sorts_as_nine() {
    // Equal strengths keep their order: the snake stays ahead of the chameleon.
    let s = play(&[(SEAL, 1), (SNAKE, 1), (CROCODILE, 1)], CHAMELEON, &[1]);
    assert_eq!(cards(s.queue()), [(CROCODILE, 1), (SNAKE, 1), (CHAMELEON, 0), (SEAL, 1)]);
}

#[test]
fn chameleon_copying_monkey_leads_the_monkeys() {
    let s = play(&[(MONKEY, 0), (HIPPO, 1), (MONKEY, 1)], CHAMELEON, &[0]);
    assert_eq!(cards(s.queue()), [(CHAMELEON, 0), (MONKEY, 1), (MONKEY, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(HIPPO, 1)]);
}

#[test]
fn chameleon_is_a_chameleon_again_before_recurring() {
    // It moves like a hippo, then the crocodile behind it bounces a strength-5 chameleon (golden game 0).
    let s = play(&[(HIPPO, 1), (HIPPO, 0), (CROCODILE, 1)], CHAMELEON, &[0]);
    assert_eq!(cards(s.queue()), [(HIPPO, 1), (HIPPO, 0), (CROCODILE, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(CHAMELEON, 0)]);
}

#[test]
fn monkey_needs_another_monkey() {
    let s = play(&[(HIPPO, 1)], MONKEY, &[]);
    assert_eq!(cards(s.queue()), [(HIPPO, 1), (MONKEY, 0)]);

    let s = play(&[(SEAL, 1), (MONKEY, 1), (CROCODILE, 1), (HIPPO, 1)], MONKEY, &[]);
    assert_eq!(cards(s.queue()), [(MONKEY, 0), (MONKEY, 1), (SEAL, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(CROCODILE, 1), (HIPPO, 1)]);
}

#[test]
fn hippo_stops_behind_zebra_lion_or_hippo() {
    let s = play(&[(SEAL, 1), (ZEBRA, 1), (GIRAFFE, 1)], HIPPO, &[]);
    assert_eq!(cards(s.queue()), [(SEAL, 1), (ZEBRA, 1), (HIPPO, 0), (GIRAFFE, 1)]);

    // It passes the crocodile, which then bounces the seal now ahead of it.
    let s = play(&[(LION, 1), (SEAL, 1), (CROCODILE, 1)], HIPPO, &[]);
    assert_eq!(cards(s.queue()), [(LION, 1), (HIPPO, 0), (CROCODILE, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(SEAL, 1)]);
}

#[test]
fn crocodile_stops_at_zebra() {
    let s = play(&[(SEAL, 1), (ZEBRA, 1), (PARROT, 1)], CROCODILE, &[]);
    assert_eq!(cards(s.queue()), [(SEAL, 1), (ZEBRA, 1), (CROCODILE, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(PARROT, 1)]);
}

#[test]
fn crocodile_played_this_turn_acts_again_after_the_hippo_ahead() {
    // Blocked by the hippo, then the hippo moves to the front and the crocodile eats what it passed,
    // nearest first. (Constructed: in play the hippo would already be at the front.)
    let s = play(&[(SEAL, 1), (PARROT, 1), (HIPPO, 1)], CROCODILE, &[]);
    assert_eq!(cards(s.queue()), [(HIPPO, 1), (CROCODILE, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(PARROT, 1), (SEAL, 1)]);
}

#[test]
fn recurring_animals_act_once() {
    let s = play(&[(KANGAROO, 1), (SEAL, 1), (GIRAFFE, 1)], ZEBRA, &[]);
    assert_eq!(cards(s.queue()), [(KANGAROO, 1), (GIRAFFE, 1), (SEAL, 1), (ZEBRA, 0)]);
}

#[test]
fn giraffe_played_this_turn_jumps_once() {
    let s = play(&[(KANGAROO, 1), (SEAL, 1)], GIRAFFE, &[]);
    assert_eq!(cards(s.queue()), [(KANGAROO, 1), (GIRAFFE, 0), (SEAL, 1)]);
}

#[test]
fn second_lion_is_bounced() {
    let s = play(&[(LION, 1), (MONKEY, 1)], LION, &[]);
    assert_eq!(cards(s.queue()), [(LION, 1), (MONKEY, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(LION, 0)]);
}

#[test]
fn lion_bounces_monkeys_and_takes_the_front() {
    let s = play(&[(SEAL, 1), (MONKEY, 1), (ZEBRA, 1)], LION, &[]);
    assert_eq!(cards(s.queue()), [(LION, 0), (SEAL, 1), (ZEBRA, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(MONKEY, 1)]);
}

#[test]
fn skunk_bounces_the_two_highest_strengths() {
    let s = play(&[(LION, 1), (ZEBRA, 1), (SEAL, 1)], SKUNK, &[]);
    assert_eq!(cards(s.queue()), [(SEAL, 1), (SKUNK, 0)]);

    // Every animal of those strengths goes, front to back.
    let s = play(&[(LION, 0), (SEAL, 1), (LION, 1)], SKUNK, &[]);
    assert_eq!(cards(s.queue()), [(SKUNK, 0)]);
    assert_eq!(cards(s.thats_it_order()), [(LION, 0), (SEAL, 1), (LION, 1)]);

    // Skunks never count.
    let s = play(&[(SKUNK, 1), (SEAL, 1)], SKUNK, &[]);
    assert_eq!(cards(s.queue()), [(SKUNK, 1), (SKUNK, 0)]);
}

#[test]
fn five_animal_check() {
    let s = play(&[(SEAL, 1), (PARROT, 1), (KANGAROO, 1), (SKUNK, 1)], ZEBRA, &[]);
    assert_eq!(cards(s.bar_order()), [(SEAL, 1), (PARROT, 1)]);
    assert_eq!(cards(s.thats_it_order()), [(ZEBRA, 0)]);
    assert_eq!(cards(s.queue()), [(KANGAROO, 1), (SKUNK, 1)]);
    assert_eq!(s.scores, [0, 2 + 4]);
}

#[test]
fn turn_draws_and_passes() {
    let mut s = State::new(1, [1 << SEAL, 1 << ZEBRA], [&[LION, HIPPO], &[GIRAFFE, SNAKE, SKUNK]]);
    s.apply(encode(ZEBRA, &[]).unwrap());
    assert_eq!(s.to_move, 0);
    assert_eq!(s.hands[1], 1 << GIRAFFE);
    assert_eq!(s.deck(1), [SNAKE, SKUNK]);
}

#[test]
fn state_is_small() {
    assert!(size_of::<State>() <= 80, "{} bytes", size_of::<State>());
}

#[test]
fn random_games_keep_invariants() {
    let mut rng = Rng::new(1);
    for i in 0..2000u32 {
        let mut s = State::deal(&mut rng, (i % 2) as u8);
        loop {
            let mut seen = 0u32;
            let mut add = |c: u8| {
                assert_eq!(seen >> c & 1, 0, "card {c} twice");
                seen |= 1 << c;
            };
            for p in 0..2 {
                (0..12).filter(|&a| s.hands[p] >> a & 1 != 0).for_each(|a| add(card(p as u8, a)));
                s.deck(p).iter().for_each(|&a| add(card(p as u8, a)));
            }
            s.queue().iter().chain(s.bar_order()).chain(s.thats_it_order()).for_each(|&c| add(c));
            assert_eq!(seen, (1 << NUM_CARDS) - 1);
            assert!((s.queue_len as usize) < QUEUE_CAP);
            for p in 0..2 {
                let points: u8 = s
                    .bar_order()
                    .iter()
                    .filter(|&&c| owner(c) == p)
                    .map(|&c| POINTS[animal(c) as usize])
                    .sum();
                assert_eq!(s.scores[p as usize], points);
            }

            let legal = s.legal_actions();
            assert_eq!(legal.is_empty(), s.is_over());
            if s.is_over() {
                break;
            }
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
    }
}

#[test]
fn seeded_games_are_reproducible() {
    let game = |seed| {
        let mut rng = Rng::new(seed);
        let mut s = State::deal(&mut rng, 0);
        random_playout(&mut s, &mut rng);
        s
    };
    assert_eq!(game(42), game(42));
    assert_ne!(game(42), game(43));
}
