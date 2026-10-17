use ark_ff::FftField;
use rust_flpcp::{Circuit, Flp};

pub fn accepts<F: FftField, C: Circuit<F>>(flp: &Flp<F, C>, input: &[F]) -> bool {
    let mut rng = ark_std::test_rng();
    let vector = flp.prove(input, &mut rng).unwrap();
    let query = flp.query(&mut rng);
    flp.verify(&query, &vector)
}

pub fn assert_tampering_rejected<F: FftField, C: Circuit<F>>(flp: &Flp<F, C>, input: &[F]) {
    let mut rng = ark_std::test_rng();
    let vector = flp.prove(input, &mut rng).unwrap();
    let query = flp.query(&mut rng);
    assert!(flp.verify(&query, &vector));

    for position in [0, input.len(), input.len() + 1, flp.vector_len() - 1] {
        let mut tampered = vector.clone();
        tampered[position] += F::ONE;
        assert!(!flp.verify(&query, &tampered), "position {position}");
    }
}
