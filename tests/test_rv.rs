use rust_flpcp::RangeValidation;

#[test]
// This test is for a validated circuit, the verification should pass.
fn validate_rv_input() {
    let range_val = RangeValidation::new(10, 10, 20);

    let message: Vec<usize> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

    // Generate proof and query.
    let proof = range_val.proof_gen(message);
    let query = range_val.query_gen();

    // Perform verification.
    let result = range_val.verify(proof, query);

    // The result should be true.
    assert!(result);
}

#[test]
// This test is for a validated circuit, the verification should NOT pass.
fn invalidate_rv_input() {
    let range_val = RangeValidation::new(12, 10, 20);

    let message: Vec<usize> = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];

    // Generate proof and query.
    let proof = range_val.proof_gen(message);
    let query = range_val.query_gen();

    // Perform verification.
    let result = range_val.verify(proof, query);

    // The result should be true.
    assert!(!result);
}
