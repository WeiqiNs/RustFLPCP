use rust_flpcp::RangeValidation;
#[test]
fn test_rv_input_correct() {
    // Set an input parameters.
    let (input_size, lower, upper) = (10, 10, 20);

    // Initialize the prover.
    let prover = RangeValidation::new(input_size, lower, upper);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);

    // Initialize the verifier.
    let verifier = RangeValidation::new(input_size, lower, upper);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be true.
    assert!(RangeValidation::verify(&proof, &query, lower));
}

#[test]
fn test_rv_input_incorrect() {
    // Set an input parameters.
    let (input_size, lower, upper) = (11, 10, 20);

    // Initialize the prover.
    let prover = RangeValidation::new(input_size, lower, upper);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 100]);

    // Initialize the verifier.
    let verifier = RangeValidation::new(input_size, lower, upper);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be true.
    assert!(!RangeValidation::verify(&proof, &query, lower));
}

#[test]
fn test_rv_input_mixed() {
    // Set an input parameters.
    let (input_size, lower, upper) = (12, 10, 20);

    // Initialize the prover.
    let prover = RangeValidation::new(input_size, lower, upper);
    // Generate proofs.
    let proof1 = prover.proof_gen(&vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10]);
    let proof2 = prover.proof_gen(&vec![100, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 10]);

    // Initialize the verifier.
    let verifier = RangeValidation::new(input_size, lower, upper);
    // Generate verification query.
    let query = verifier.query_gen();

    // The first verification should be true.
    assert!(RangeValidation::verify(&proof1, &query, lower));
    // The second verification should be false.
    assert!(!RangeValidation::verify(&proof2, &query, lower));
}
