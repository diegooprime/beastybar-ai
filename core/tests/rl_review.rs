//! Adversarial review of the policy observation (features.rs) at scale.
//!
//! The mover's information set (as the observation is meant to capture it) is: who started, the public move history
//! (player, action, queue before each move) and the mover's current hand. Two positions in the same information set
//! must encode identically (no hidden information leaks), and two positions in different information sets must
//! encode differently (no public information is lost).

use std::collections::HashMap;

use bbcore::features::{History, OBS_DIM, PRIV_DIM, encode_obs, encode_privileged};
use bbcore::{DECK_CAP, Rng, State};

/// The information set relative to the mover (the observation is seat-symmetric): whether I started, my hand,
/// and the history as (by me, action, queue before with cards as mine/theirs).
type Key = (bool, u16, Vec<(bool, u8, Vec<(bool, u8)>)>);

fn key(s: &State, h: &History) -> Key {
    let me = s.to_move;
    let rel = |c: u8| ((c >= 12) as u8 == me, c % 12);
    let plies = h
        .plies()
        .iter()
        .map(|p| (p.player == me, p.action, p.queue[..p.queue_len as usize].iter().map(|&c| rel(c)).collect()))
        .collect();
    (s.start == me, s.hands[me as usize], plies)
}

fn obs(s: &State, h: &History) -> Vec<u32> {
    let mut v = vec![f32::NAN; OBS_DIM];
    encode_obs(s, h, &mut v);
    v.into_iter().map(f32::to_bits).collect()
}

/// A deal that reproduces the same public history for the first `k` actions: the animals each player has not played
/// by ply k are re-assigned among the arrangement positions (4 hand slots + deck order) they occupied. The mover at
/// ply k keeps their current hand (only their undrawn deck is reordered).
fn alternative_deal(deal: &State, actions: &[u8], k: usize, rng: &mut Rng) -> State {
    let mut s = *deal;
    for &a in &actions[..k] {
        s.apply(a);
    }
    let mover = s.to_move as usize;
    let mut hands = [0u16; 2];
    let mut decks = [[0u8; DECK_CAP]; 2];
    let mut lens = [0usize; 2];
    for p in 0..2 {
        let hand: Vec<u8> = (0..12).filter(|&a| deal.hands[p] >> a & 1 != 0).collect();
        let mut arr: Vec<u8> = hand.iter().copied().chain(deal.deck(p).iter().copied()).collect();
        let deck_now: u16 = s.deck(p).iter().fold(0, |m, &a| m | 1 << a);
        let unplayed = s.hands[p] | deck_now;
        let movable = if p == mover { deck_now } else { unplayed };
        let pos: Vec<usize> = (0..arr.len()).filter(|&i| movable >> arr[i] & 1 != 0).collect();
        let mut vals: Vec<u8> = pos.iter().map(|&i| arr[i]).collect();
        rng.shuffle(&mut vals);
        for (&i, &v) in pos.iter().zip(&vals) {
            arr[i] = v;
        }
        hands[p] = arr[..4].iter().fold(0, |m, &a| m | 1 << a);
        lens[p] = arr.len() - 4;
        decks[p][..lens[p]].copy_from_slice(&arr[4..]);
    }
    State::new(deal.start, hands, [&decks[0][..lens[0]], &decks[1][..lens[1]]])
}

#[test]
fn same_information_set_same_observation_across_alternative_deals() {
    let mut rng = Rng::new(42);
    let mut perturbed = 0;
    for g in 0..20_000u64 {
        let deal = State::deal(&mut rng, (g % 2) as u8);
        let mut s = deal;
        let mut actions = Vec::new();
        let mut positions = Vec::new();
        let mut h = History::default();
        while !s.is_over() {
            positions.push((s, h));
            let legal = s.legal_actions();
            let a = legal.nth_action(rng.below(legal.len()));
            h.push(&s, a);
            s.apply(a);
            actions.push(a);
        }
        let k = rng.below(positions.len());
        let (s, h) = positions[k];
        let alt_deal = alternative_deal(&deal, &actions, k, &mut rng);
        // Replay the same actions on the alternative deal: every one must stay legal.
        let mut t = alt_deal;
        let mut th = History::default();
        for &a in &actions[..k] {
            assert!(t.legal_actions().contains(a), "alternative deal must keep the public history legal");
            th.push(&t, a);
            t.apply(a);
        }
        assert_eq!(key(&t, &th), key(&s, &h));
        assert_eq!(obs(&t, &th), obs(&s, &h), "game {g} ply {k}: observation depends on hidden information");
        let (mut p1, mut p2) = (vec![0.0; PRIV_DIM], vec![0.0; PRIV_DIM]);
        encode_privileged(&s, &mut p1);
        encode_privileged(&t, &mut p2);
        perturbed += (p1 != p2) as u32;
    }
    assert!(perturbed > 10_000, "the hidden information must actually change ({perturbed})");
}

#[test]
fn different_information_sets_never_share_an_observation() {
    let mut rng = Rng::new(7);
    let mut seen: HashMap<Vec<u32>, Key> = HashMap::new();
    let mut collisions_same_key = 0;
    for g in 0..30_000u64 {
        let mut s = State::deal(&mut rng, (g % 2) as u8);
        let mut h = History::default();
        while !s.is_over() {
            let o = obs(&s, &h);
            let k = key(&s, &h);
            if let Some(prev) = seen.get(&o) {
                assert_eq!(prev, &k, "two different information sets encode to the same observation");
                collisions_same_key += 1;
            } else {
                seen.insert(o, k);
            }
            let legal = s.legal_actions();
            let a = legal.nth_action(rng.below(legal.len()));
            h.push(&s, a);
            s.apply(a);
        }
    }
    // Early-game information sets repeat across games (e.g. ply 0 with the same hand), so collisions with the
    // same key must happen; they show the map really is compared.
    assert!(collisions_same_key > 1000);
}
