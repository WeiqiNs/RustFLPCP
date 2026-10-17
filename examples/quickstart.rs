use ark_bls12_381::Fr;
use rand::rngs::OsRng;
use rust_flpcp::{Flp, FlpError, OneHot};

fn main() -> Result<(), FlpError> {
    let rng = &mut OsRng;
    let flp = Flp::<Fr, _>::new(OneHot { len: 4 })?;

    let vector = flp.prove(&[0u64, 0, 1, 0].map(Fr::from), rng)?;

    let query = flp.query(rng);
    let answers: Vec<Fr> = query
        .linear_queries()
        .iter()
        .map(|linear| {
            let inner: Fr = linear.coeffs.iter().zip(&vector).map(|(c, v)| *c * v).sum();
            inner + linear.constant
        })
        .collect();
    assert!(flp.decide(&query, &answers));
    assert!(flp.verify(&query, &vector));
    Ok(())
}
