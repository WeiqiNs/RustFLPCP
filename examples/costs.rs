use ark_bls12_381::Fr;
use rust_flpcp::{
    BitRange, Circuit, Flp, MeanVariance, Membership, NormBound, OneHot, Range, SignedNormBound,
};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::num::NonZeroUsize;

const LENS: [usize; 6] = [63, 64, 255, 256, 1023, 1024];
const DENOMINATIONS: [u64; 7] = [1, 2, 5, 10, 20, 50, 100];

fn row<C: Circuit<Fr> + Debug>(circuit: C) {
    let label = format!("{circuit:?}");
    let input_len = circuit.input_len();
    let flp = Flp::<Fr, C>::new(circuit).expect("benchmark parameters are valid");
    println!(
        "| {label} | {input_len} | {} | {} | {} |",
        flp.vector_len(),
        flp.proof_len(),
        flp.query_count()
    );
}

fn norm_bound(len: usize, square_chunk: Option<NonZeroUsize>) -> NormBound {
    NormBound {
        len,
        input_bits: 8,
        norm_bits: 27,
        square_chunk,
    }
}

fn main() {
    println!("| circuit | input_len | vector_len | proof_len | queries |");
    println!("|---|---|---|---|---|");
    for len in LENS {
        row(Range::binary(len));
    }
    for len in LENS {
        row(Range {
            len,
            bounds: 0..=15,
        });
    }
    for len in LENS {
        row(BitRange {
            len,
            bounds: 0..=1023,
        });
    }
    row(BitRange {
        len: 1024,
        bounds: 0..=1000,
    });
    for len in LENS {
        row(OneHot { len });
    }
    for len in LENS {
        row(Membership {
            len,
            values: BTreeSet::from(DENOMINATIONS),
        });
    }
    for len in LENS {
        row(norm_bound(len, None));
    }
    for len in LENS {
        row(SignedNormBound {
            len,
            input_bits: NonZeroUsize::new(8).unwrap(),
            norm_bits: 27,
            square_chunk: None,
        });
    }
    for len in LENS {
        row(MeanVariance { len, input_bits: 8 });
    }
    for square_chunk in [
        NonZeroUsize::new(4),
        NonZeroUsize::new(16),
        NonZeroUsize::new(64),
    ] {
        row(norm_bound(1024, square_chunk));
    }
}
