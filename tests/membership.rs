mod common;

use ark_bls12_381::Fr;
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{Flp, Membership};
use std::collections::BTreeSet;

const VALID: [u64; 4] = [7, 3, 42, 7];

fn circuit() -> Membership {
    Membership {
        len: VALID.len(),
        values: BTreeSet::from([3, 7, 42]),
    }
}

#[test]
fn accepts_members_only() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert!(accepts(&flp, &VALID.map(Fr::from)));

    let mut between_members = VALID;
    between_members[1] = 5;
    assert!(!accepts(&flp, &between_members.map(Fr::from)));
}

#[test]
fn rejects_single_coordinate_tampering() {
    let flp = Flp::<Fr, _>::new(circuit()).unwrap();
    assert_tampering_rejected(&flp, &VALID.map(Fr::from));
}
