mod common;

use ark_bls12_381::Fr;
use ark_ff::Field;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, MeanVariance};

const VALID: [u64; 3] = [0, 7, 15];

fn circuit() -> MeanVariance {
    MeanVariance {
        len: VALID.len(),
        input_bits: 4,
    }
}

#[test]
fn accepts_consistent_squares_only() {
    let circuit = circuit();
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();

    let valid = circuit.encode(&VALID);
    assert!(accepts(&flp, &valid));

    let mut wrong_square = valid;
    wrong_square[VALID.len()] += Fr::ONE;
    assert!(!accepts(&flp, &wrong_square));

    let too_wide = [16, 7, 15];
    assert!(!accepts(&flp, &circuit.encode(&too_wide)));

    let mut forged = circuit.encode(&too_wide);
    forged[2 * VALID.len() + 3] = Fr::from(2u64);
    assert!(!accepts(&flp, &forged));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let circuit = circuit();
    let flp = Flp::<Fr, _>::new(circuit.clone()).unwrap();
    assert_tampering_rejected(&flp, &circuit.encode(&VALID));
}
