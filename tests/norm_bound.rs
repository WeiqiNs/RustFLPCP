mod common;

use ark_bls12_381::Fr;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, NormBound};
use std::num::NonZeroUsize;

const VALID: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

fn circuit(square_chunk: Option<NonZeroUsize>) -> NormBound {
    NormBound {
        len: VALID.len(),
        input_bits: 5,
        norm_bits: 9,
        square_chunk,
    }
}

#[test]
fn accepts_bounded_norm_only() {
    let cases = [
        (None, 2),
        (NonZeroUsize::new(3), 5),
        (NonZeroUsize::new(10), 12),
    ];
    for (square_chunk, query_count) in cases {
        let circuit = circuit(square_chunk);
        let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
        assert_eq!(flp.query_count(), query_count, "{square_chunk:?}");

        let max_norm = [22, 5, 1, 1, 0, 0, 0, 0, 0, 0];
        let mut over_norm = max_norm;
        over_norm[4] = 1;
        assert!(
            !accepts(&flp, &circuit.encode(&over_norm)),
            "{square_chunk:?}"
        );

        let mut input = circuit.encode(&max_norm);
        assert!(accepts(&flp, &input), "{square_chunk:?}");
        input[0] = -input[0];
        assert!(!accepts(&flp, &input), "{square_chunk:?}");
    }
}

#[test]
fn accepts_norms_beyond_u128() {
    let circuit = NormBound {
        len: 2,
        input_bits: 64,
        norm_bits: 130,
        square_chunk: None,
    };
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert!(accepts(&flp, &circuit.encode(&[u64::MAX, u64::MAX])));
}

#[test]
fn rejects_non_bit_decompositions() {
    let norm_circuit = circuit(None);
    let flp = Flp::<Fr, _>::new(norm_circuit.clone()).unwrap();
    let mut over_norm = norm_circuit.encode(&[22, 5, 1, 1, 1, 0, 0, 0, 0, 0]);
    let top_norm_bit = over_norm.len() - 1;
    over_norm[top_norm_bit] = Fr::from(2u64);
    assert!(!accepts(&flp, &over_norm));

    let wide_circuit = NormBound {
        len: 1,
        input_bits: 5,
        norm_bits: 11,
        square_chunk: None,
    };
    let flp = Flp::<Fr, _>::new(wide_circuit.clone()).unwrap();
    let mut wide_value = wide_circuit.encode(&[33]);
    wide_value[1 + 4] = Fr::from(2u64);
    assert!(!accepts(&flp, &wide_value));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let circuit = circuit(NonZeroUsize::new(3));
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert_tampering_rejected(&flp, &circuit.encode(&VALID));
}
