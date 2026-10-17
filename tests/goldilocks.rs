mod common;

use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{BitRange, Flp, FlpError, Membership, Range};
use std::collections::BTreeSet;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
struct GoldilocksConfig;

type Goldilocks = Fp64<MontBackend<GoldilocksConfig, 1>>;

#[test]
fn rejects_ranges_wider_than_the_field() {
    let cases = [
        ((1 << 62) - 1, None),
        (
            1 << 62,
            Some(FlpError::ExceedsField {
                bits: 64,
                limit: 63,
            }),
        ),
    ];
    for (end, expected) in cases {
        let circuit = BitRange {
            len: 1,
            bounds: 0..=end,
        };
        assert_eq!(Flp::<Goldilocks, _>::new(circuit).err(), expected, "{end}");
    }
}

#[test]
fn rejects_constants_beyond_the_field() {
    let modulus = 18_446_744_069_414_584_321u64;
    let largest_fitting = (1u64 << 63) - 1;
    let exceeds = Some(FlpError::ExceedsField {
        bits: 64,
        limit: 63,
    });
    let membership = |value: u64| Membership {
        len: 1,
        values: BTreeSet::from([value]),
    };
    let range = |end: u64| Range {
        len: 1,
        bounds: end - 3..=end,
    };
    let bit_range = |end: u64| BitRange {
        len: 1,
        bounds: end - 9..=end,
    };

    assert_eq!(
        Flp::<Goldilocks, _>::new(membership(modulus + 5)).err(),
        exceeds
    );
    assert_eq!(
        Flp::<Goldilocks, _>::new(membership(largest_fitting)).err(),
        None
    );
    assert_eq!(Flp::<Goldilocks, _>::new(range(modulus + 3)).err(), exceeds);
    assert_eq!(
        Flp::<Goldilocks, _>::new(range(largest_fitting)).err(),
        None
    );
    assert_eq!(
        Flp::<Goldilocks, _>::new(bit_range(u64::MAX)).err(),
        exceeds
    );
    assert_eq!(
        Flp::<Goldilocks, _>::new(bit_range(largest_fitting)).err(),
        None
    );
}

#[test]
fn proves_over_goldilocks() {
    let circuit = BitRange {
        len: 2,
        bounds: 10..=20,
    };
    let flp = Flp::<Goldilocks, _>::new(circuit.clone()).unwrap();
    assert_tampering_rejected(&flp, &circuit.encode(&[10, 20]));
    assert!(!accepts(&flp, &circuit.encode(&[9, 20])));
}
