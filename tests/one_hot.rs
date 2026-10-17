mod common;

use ark_bls12_381::Fr;
use ark_ff::{AdditiveGroup, Field};
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, OneHot};

const VALID: [u64; 5] = [0, 0, 1, 0, 0];

fn circuit() -> OneHot {
    OneHot { len: VALID.len() }
}

#[test]
fn accepts_one_hot_only() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert!(accepts(&flp, &VALID.map(Fr::from)));

    for invalid in [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]] {
        assert!(!accepts(&flp, &invalid.map(Fr::from)), "{invalid:?}");
    }
    let cancelling = [Fr::from(2u64), -Fr::ONE, Fr::ZERO, Fr::ZERO, Fr::ZERO];
    assert!(!accepts(&flp, &cancelling));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert_tampering_rejected(&flp, &VALID.map(Fr::from));
}
