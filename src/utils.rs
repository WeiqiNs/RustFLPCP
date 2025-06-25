use ark_bls12_381::Fr as F;

/// Define some useful types.
pub type Message = Vec<usize>;
pub type Proof = Vec<F>;
pub struct Query {
    pub f_queries: Vec<Vec<F>>,
    pub p_query: Vec<F>,
    pub output_query: Vec<F>,
}

pub mod utils {
    /// Use F from super and import necessary modules.
    use super::F;
    use ark_ff::{One, UniformRand, Zero};


    /// Compute inner product between two vectors.
    ///
    /// # Arguments
    ///
    /// * `x_vec`: a vector of field element.
    /// * `y_vec`: a vector of field element.
    ///
    /// returns: a field element.
    pub fn inner_product(x_vec: &[F], y_vec: &[F]) -> F {
        assert_eq!(
            x_vec.len(),
            y_vec.len(),
            "Vectors must have the same length"
        );
        x_vec.iter().zip(y_vec).map(|(x, y)| *x * *y).sum()
    }

    /// Randomly sample an element from the field larger than integer x.
    ///
    /// # Arguments
    ///
    /// * `x`: an u64 integer.
    ///
    /// returns: a field element.
    pub fn random_larger_than(x: usize) -> F {
        let x_field = F::from(x as u64);
        let mut rng = ark_std::test_rng();
        loop {
            let candidate = F::rand(&mut rng);
            if candidate > x_field {
                return candidate;
            }
        }
    }

    /// Multiply two polynomials with the following assumptions.
    ///
    /// The input coefficients correspond to c_0, c_1 * x, c_2 * x^2, ...
    ///
    /// # Arguments
    ///
    /// * `p1`: the coefficients of the first polynomial.
    /// * `p2`: the coefficients of the second polynomial.
    ///
    /// returns: a vector of field elements: the coefficients of the product of the two inputs.
    pub fn multiply_polynomials(p1: &[F], p2: &[F]) -> Vec<F> {
        // Set the desired length of the result.
        let mut result = vec![F::zero(); p1.len() + p2.len() - 1];
        // Perform the computation.
        for (i, &a) in p1.iter().enumerate() {
            for (j, &b) in p2.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        result
    }

    /// Perform lagrange interpolation on paris of coordinates and returns the coefficients.
    ///
    /// # Arguments
    ///
    /// * `x_vals`: a vector of field elements, represent x coordinates for point evaluations.
    /// * `y_vals`: a vector of field elements, represent y coordinates for point evaluations.
    ///
    /// returns: a vector of field elements: the coefficients as field elements.
    pub fn lagrange_interpolate_coeffs(x_vals: &[F], y_vals: &[F]) -> Vec<F> {
        assert_eq!(x_vals.len(), y_vals.len());
        let n = x_vals.len();
        let mut coeffs = vec![F::zero(); n];

        for i in 0..n {
            let mut term_coeffs = vec![F::one()];

            for j in 0..n {
                if i == j {
                    continue;
                }
                let denom = x_vals[i] - x_vals[j];
                let base = vec![-x_vals[j], F::one()];
                let scaled: Vec<F> = base.iter().map(|&c| c / denom).collect();
                term_coeffs = multiply_polynomials(&term_coeffs, &scaled);
            }

            for (k, &c) in term_coeffs.iter().enumerate() {
                coeffs[k] += c * y_vals[i];
            }
        }

        coeffs
    }

    /// Evaluate all Lagrange basis polynomials L_j(r) for j in [0, ..., input_size].
    ///
    /// # Arguments
    ///
    /// * `r`: A field element at which to evaluate the basis.
    /// * `size`: The number of input positions (basis size = input_size + 1).
    ///
    /// returns: A list of field elements [L_0(r), ..., L_input_size(r)].
    pub fn lagrange_basis_polynomial(r: F, size: usize) -> Vec<F> {
        (0..=size + 1)
            .map(|j| {
                let mut prod = F::one();
                let j_f = F::from(j as u64);
                for m in 0..=size {
                    if m != j {
                        let m_f = F::from(m as u64);
                        prod *= (r - m_f) / (j_f - m_f);
                    }
                }
                prod
            })
            .collect()
    }
}
