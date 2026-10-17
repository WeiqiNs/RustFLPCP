mod common;

use ark_bls12_381::Fr;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{BitRange, Circuit, Flp};
use std::ops::RangeInclusive;

const VALID: [u64; 4] = [10, 15, 20, 13];

fn circuit() -> BitRange {
    BitRange {
        len: VALID.len(),
        bounds: 10..=20,
    }
}

#[test]
fn accepts_values_inside_bounds_only() {
    let circuit = circuit();
    assert_eq!(
        Circuit::<Fr>::input_len(&circuit),
        VALID.len() * (1 + 2 * 4)
    );
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert!(accepts(&flp, &circuit.encode(&VALID)));

    for value in [9, 21] {
        let mut measurement = VALID;
        measurement[1] = value;
        assert!(!accepts(&flp, &circuit.encode(&measurement)), "{value}");
    }
}

#[test]
fn accepts_aligned_ranges_with_lower_gap_only() {
    let cases: [(RangeInclusive<u64>, usize, [u64; 2]); 2] =
        [(8..=15, 3, [8, 15]), (0..=u64::MAX, 64, [0, u64::MAX])];
    for (bounds, bits, valid) in cases {
        let circuit = BitRange { len: 2, bounds };
        assert_eq!(Circuit::<Fr>::input_len(&circuit), 2 * (1 + bits));
        let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
        assert!(accepts(&flp, &circuit.encode(&valid)), "{valid:?}");
    }

    let circuit = BitRange {
        len: 2,
        bounds: 8..=15,
    };
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    for invalid in [[7, 15], [8, 16]] {
        assert!(!accepts(&flp, &circuit.encode(&invalid)), "{invalid:?}");
    }
}

#[test]
fn degenerate_single_value_range() {
    let circuit = BitRange {
        len: 2,
        bounds: 7..=7,
    };
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert!(accepts(&flp, &circuit.encode(&[7, 7])));
    assert!(!accepts(&flp, &circuit.encode(&[6, 7])));
    assert!(!accepts(&flp, &circuit.encode(&[7, 8])));
}

#[test]
fn rejects_non_bit_gap_decompositions() {
    let circuit = circuit();
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    let (len, bits) = (VALID.len(), 4);
    let minus_one = -Fr::from(1u64);

    let mut below = VALID;
    below[1] = 9;
    let mut forged = circuit.encode(&below);
    forged[len + bits..][..bits].copy_from_slice(&[
        minus_one,
        Fr::from(0u64),
        Fr::from(0u64),
        Fr::from(0u64),
    ]);
    assert!(!accepts(&flp, &forged));

    let mut above = VALID;
    above[1] = 21;
    let mut forged = circuit.encode(&above);
    forged[len + len * bits + bits..][..bits].copy_from_slice(&[
        minus_one,
        Fr::from(0u64),
        Fr::from(0u64),
        Fr::from(0u64),
    ]);
    assert!(!accepts(&flp, &forged));

    let aligned = BitRange {
        len: 2,
        bounds: 8..=15,
    };
    let flp = Flp::<Fr, _>::new(aligned.clone()).unwrap();
    let mut forged = aligned.encode(&[16, 8]);
    forged[2..5].copy_from_slice(&[Fr::from(0u64), Fr::from(0u64), Fr::from(2u64)]);
    assert!(!accepts(&flp, &forged));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let circuit = circuit();
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert_tampering_rejected(&flp, &circuit.encode(&VALID));
}
