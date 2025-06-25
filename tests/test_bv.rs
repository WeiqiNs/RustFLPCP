use rust_flpcp::BinaryValidation;

#[test]
fn test_bv_input_correct() {
    // Set an input size.
    let input_size = 10;

    // Initialize the prover.
    let prover = BinaryValidation::new(input_size);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0]);

    // Initialize the verifier.
    let verifier = BinaryValidation::new(input_size);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be true.
    assert!(BinaryValidation::verify(&proof, &query));
}

#[test]
fn test_bv_input_incorrect() {
    // Set an input size.
    let input_size = 11;

    // Initialize the prover.
    let prover = BinaryValidation::new(input_size);
    // Generate a proof.
    let proof = prover.proof_gen(&vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 10]);

    // Initialize the verifier.
    let verifier = BinaryValidation::new(input_size);
    // Generate verification query.
    let query = verifier.query_gen();

    // The verification result should be false.
    assert!(!BinaryValidation::verify(&proof, &query));
}

#[test]
fn test_bv_input_mixed() {
    // Set an input size.
    let input_size = 12;

    // Initialize the prover.
    let prover = BinaryValidation::new(input_size);
    // Generate proofs.
    let proof1 = prover.proof_gen(&vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]);
    let proof2 = prover.proof_gen(&vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 10]);

    // Initialize the verifier.
    let verifier = BinaryValidation::new(input_size);
    // Generate verification query.
    let query = verifier.query_gen();

    // The first verification should be true.
    assert!(BinaryValidation::verify(&proof1, &query));
    // The second verification should be false.
    assert!(!BinaryValidation::verify(&proof2, &query));
}
