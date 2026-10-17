mod common;

use ark_bls12_381::Fr;
use ark_ff::Field;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, Range};

const VALID: [u64; 12] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10];

fn circuit() -> Range {
    Range {
        len: VALID.len(),
        bounds: 10..=20,
    }
}

#[test]
fn accepts_values_inside_bounds_only() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert!(accepts(&flp, &VALID.map(Fr::from)));

    for value in [9, 21] {
        let mut input = VALID;
        input[0] = value;
        assert!(!accepts(&flp, &input.map(Fr::from)), "{value}");
    }
}

#[test]
fn rejects_single_coordinate_tampering() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert_tampering_rejected(&flp, &VALID.map(Fr::from));
}

#[test]
fn binary_rejects_non_bits_including_cancelling_pair() {
    let x2 = (Fr::ONE + (-Fr::from(23u64)).sqrt().unwrap()) / Fr::from(2u64);
    assert_eq!(x2 * (x2 - Fr::ONE), -Fr::from(6u64));

    let flp = Flp::<Fr, _>::new(Range::binary(2)).unwrap();
    assert!(accepts(&flp, &[1u64, 0].map(Fr::from)));
    assert!(!accepts(&flp, &[2u64, 0].map(Fr::from)));
    assert!(!accepts(&flp, &[Fr::from(3u64), x2]));
}
