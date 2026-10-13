//! Animals and cards. Animal id = strength - 1; card id = owner * 12 + animal.

pub const SKUNK: u8 = 0;
pub const PARROT: u8 = 1;
pub const KANGAROO: u8 = 2;
pub const MONKEY: u8 = 3;
pub const CHAMELEON: u8 = 4;
pub const SEAL: u8 = 5;
pub const ZEBRA: u8 = 6;
pub const GIRAFFE: u8 = 7;
pub const SNAKE: u8 = 8;
pub const CROCODILE: u8 = 9;
pub const HIPPO: u8 = 10;
pub const LION: u8 = 11;

pub const NUM_ANIMALS: usize = 12;
pub const NUM_CARDS: usize = 24;

/// All animals of one player, as an animal bitmask.
pub const ALL_ANIMALS: u16 = (1 << NUM_ANIMALS) - 1;

pub const NAMES: [&str; NUM_ANIMALS] = [
    "skunk",
    "parrot",
    "kangaroo",
    "monkey",
    "chameleon",
    "seal",
    "zebra",
    "giraffe",
    "snake",
    "crocodile",
    "hippo",
    "lion",
];

pub const POINTS: [u8; NUM_ANIMALS] = [4, 4, 4, 3, 3, 2, 4, 3, 2, 3, 2, 2];

#[inline]
pub const fn card(owner: u8, animal: u8) -> u8 {
    owner * NUM_ANIMALS as u8 + animal
}

#[inline]
pub const fn animal(card: u8) -> u8 {
    card % NUM_ANIMALS as u8
}

#[inline]
pub const fn owner(card: u8) -> u8 {
    (card >= NUM_ANIMALS as u8) as u8
}

#[inline]
pub const fn strength(animal: u8) -> u8 {
    animal + 1
}

pub fn animal_id(name: &str) -> Option<u8> {
    NAMES.iter().position(|&n| n == name).map(|i| i as u8)
}
