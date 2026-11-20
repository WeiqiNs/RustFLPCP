mod common;

use ark_ff::fields::{Fp64, MontBackend, MontConfig};
use common::{accepts, assert_tampering_rejected};
use rust_flpcp::{
    BitRange, Circuit, Flp, FlpError, Gadget, GadgetCalls, GadgetId, ParallelSum, RangeProduct,
    SetProduct, Wire,
};
use std::collections::BTreeSet;
use std::slice;

#[derive(MontConfig)]
#[modulus = "18446744069414584321"]
#[generator = "7"]
struct GoldilocksConfig;

type Goldilocks = Fp64<MontBackend<GoldilocksConfig, 1>>;

const MODULUS: u64 = 18_446_744_069_414_584_321;

type GadgetFactory = fn() -> Box<dyn Gadget<Goldilocks>>;

struct SingleGadget(GadgetFactory);

impl Circuit<Goldilocks> for SingleGadget {
    fn input_len(&self) -> usize {
        1
    }

    fn gadgets(&self) -> Vec<Box<dyn Gadget<Goldilocks>>> {
        vec![(self.0)()]
    }

    fn eval<W: Wire<Goldilocks>>(&self, input: &[W], calls: &mut impl GadgetCalls<W>) -> Vec<W> {
        vec![calls.call(GadgetId(0), slice::from_ref(&input[0]))]
    }
}

#[test]
fn bit_range_rejects_bounds_beyond_the_field() {
    let largest_fitting = (1u64 << 63) - 1;
    let exceeds = Some(FlpError::ExceedsField {
        bits: 64,
        limit: 63,
    });
    let cases = [
        (0..=(1 << 62) - 1, None),
        (0..=1 << 62, exceeds.clone()),
        (largest_fitting - 9..=largest_fitting, None),
        (u64::MAX - 9..=u64::MAX, exceeds),
    ];
    for (bounds, expected) in cases {
        let circuit = BitRange {
            len: 1,
            bounds: bounds.clone(),
        };
        assert_eq!(
            Flp::<Goldilocks, _>::new(circuit).err(),
            expected,
            "{bounds:?}"
        );
    }
}

#[test]
fn custom_gadgets_reject_constants_beyond_the_field() {
    let exceeds = Some(FlpError::ExceedsField {
        bits: 64,
        limit: 63,
    });
    let cases: [(GadgetFactory, Option<FlpError>); 5] = [
        (
            || {
                Box::new(SetProduct {
                    values: BTreeSet::from([MODULUS]),
                })
            },
            exceeds.clone(),
        ),
        (
            || {
                Box::new(RangeProduct {
                    bounds: MODULUS - 1..=MODULUS + 1,
                })
            },
            exceeds.clone(),
        ),
        (
            || {
                Box::new(ParallelSum {
                    inner: SetProduct {
                        values: BTreeSet::from([MODULUS]),
                    },
                    count: 1,
                })
            },
            exceeds,
        ),
        (
            || {
                Box::new(SetProduct {
                    values: BTreeSet::from([(1 << 63) - 1]),
                })
            },
            None,
        ),
        (
            || {
                Box::new(RangeProduct {
                    bounds: (1 << 63) - 4..=(1 << 63) - 1,
                })
            },
            None,
        ),
    ];
    for (gadget, expected) in cases {
        assert_eq!(
            Flp::<Goldilocks, _>::new(SingleGadget(gadget)).err(),
            expected
        );
    }
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
