use rust_flpcp::NormBoundValidation;

#[test]
fn test_nbv_input_correct() {
    // Set an input parameters.
    let (input_size, norm_bound, input_bound) = (10, 10, 5);

    // Initialize the prover.
    let prover = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    // Initialize the verifier.
    let verifier = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be true.
    assert!(NormBoundValidation::verify(&proof, &query));
}

#[test]
fn test_nbv_input_incorrect() {
    // Set an input parameters.
    let (input_size, norm_bound, input_bound) = (11, 10, 5);

    // Initialize the prover.
    let prover = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    // Initialize the verifier.
    let verifier = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be true.
    assert!(!NormBoundValidation::verify(&proof, &query));
}

#[test]
fn test_nbv_input_mixed() {
    // Set an input parameters.
    let (input_size, norm_bound, input_bound) = (10, 11, 5);

    // Initialize the prover.
    let prover = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate proofs.
    let proof1 = prover.proof_gen(&vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 5]);
    let proof2 = prover.proof_gen(&vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 33]);
    let proof3 = prover.proof_gen(&vec![30, 30, 30, 30, 30, 30, 30, 30, 30, 30]);

    // Initialize the verifier.
    let verifier = NormBoundValidation::new(input_size, norm_bound, input_bound);
    // Generate verification query.
    let query = verifier.query_gen();

    // The first verification should be true.
    assert!(NormBoundValidation::verify(&proof1, &query));
    // The second verification should be false.
    assert!(!NormBoundValidation::verify(&proof2, &query));
    // The third verification should be false.
    assert!(!NormBoundValidation::verify(&proof3, &query));
}
