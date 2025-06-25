use ark_bls12_381::Fr as F;
use rust_flpcp::utils;

#[test]
fn test_int_to_fixed_binary() {
    let r = utils::int_to_fixed_binary(5, 4);
    assert_eq!(vec![1, 0, 1, 0], r);

    let r = utils::int_to_fixed_binary(1, 5);
    assert_eq!(vec![1, 0, 0, 0, 0], r);

    let r = utils::int_to_fixed_binary(127, 6);
    assert_eq!(vec![1, 1, 1, 1, 1, 1], r);
}

#[test]
fn test_inner_product() {
    let vec_x: Vec<F> = (0..10).map(|i| F::from(i as u64)).collect();
    let vec_y: Vec<F> = (11..21).map(|i| F::from(i as u64)).collect();
    let r = utils::inner_product(&vec_x, &vec_y);
    assert_eq!(r, F::from(780));
}

#[test]
fn test_random_larger_than() {
    let r = utils::random_larger_than(100);
    assert!(r > F::from(100));
}

#[test]
fn test_multiply_polynomials() {
    let p_1: Vec<F> = [1, 2, 3, 4].iter().map(|&i| F::from(i as u64)).collect();
    let p_2: Vec<F> = [4, 5, 6].iter().map(|&i| F::from(i as u64)).collect();
    let r = utils::multiply_polynomials(&p_1, &p_2);
    assert_eq!(r[0], F::from(4));
    assert_eq!(r[3], F::from(43));
    assert_eq!(r[5], F::from(24));
}
