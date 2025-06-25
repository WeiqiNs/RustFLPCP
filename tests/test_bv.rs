use rust_flpcp::BinaryValidation;

#[test]
// This test is for a validated circuit, the verification should pass.
fn validate_bv_input() {
    let binary_val = BinaryValidation::new(10);

    let message: Vec<usize> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

    // Generate proof and query.
    let proof = binary_val.proof_gen(message);
    let query = binary_val.query_gen();

    // Perform verification.
    let result = BinaryValidation::verify(proof, query);

    // The result should be true.
    assert!(result);
}

#[test]
// This test is for a validated circuit, the verification should NOT pass.
fn invalidate_bv_input() {
    let binary_val = BinaryValidation::new(12);

    let message: Vec<usize> = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 100];

    // Generate proof and query.
    let proof = binary_val.proof_gen(message);
    let query = binary_val.query_gen();

    // Perform verification.
    let result = BinaryValidation::verify(proof, query);

    assert!(!result);
}
