mod common;

use ark_bls12_381::Fr;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, SignedNormBound};
use std::num::NonZeroUsize;

const VALID: [i64; 5] = [-8, 7, -3, 2, -1];

fn circuit(square_chunk: Option<NonZeroUsize>) -> SignedNormBound {
    SignedNormBound {
        len: VALID.len(),
        input_bits: NonZeroUsize::new(4).unwrap(),
        norm_bits: 7,
        square_chunk,
    }
}

#[test]
fn accepts_bounded_norm_only() {
    for square_chunk in [None, NonZeroUsize::new(2)] {
        let circuit = circuit(square_chunk);
        let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
        assert!(accepts(&flp, &circuit.encode(&VALID)), "{square_chunk:?}");

        for invalid in [[-8, 7, -3, 2, -2], [8, 7, -3, 2, -1], [-9, 1, 0, 0, 0]] {
            assert!(
                !accepts(&flp, &circuit.encode(&invalid)),
                "{square_chunk:?} {invalid:?}"
            );
        }
    }
}

#[test]
fn accepts_full_i64_range_at_64_bits() {
    let circuit = SignedNormBound {
        len: 2,
        input_bits: NonZeroUsize::new(64).unwrap(),
        norm_bits: 128,
        square_chunk: None,
    };
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert!(accepts(&flp, &circuit.encode(&[i64::MIN, i64::MAX])));
}

#[test]
fn rejects_non_bit_decompositions() {
    let circuit = circuit(None);
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    let mut forged = circuit.encode(&[8, 7, -3, 2, -1]);
    forged[VALID.len() + 3] = Fr::from(2u64);
    assert!(!accepts(&flp, &forged));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let circuit = circuit(NonZeroUsize::new(2));
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert_tampering_rejected(&flp, &circuit.encode(&VALID));
}
