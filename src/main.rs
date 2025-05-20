use ark_bls12_381::Fr as F;
use ark_ff::{BigInt, Field, One, PrimeField, UniformRand, Zero};
use ark_std::rand::RngCore;

/// The binary check circuit struct, containing essential fields.
pub struct BC {
    input_size: usize,
    proof_size: usize,
    total_size: usize,
    rng: Box<dyn RngCore>,
}

impl BC {

    /// Create a new BC instance from the input size.
    /// 
    /// # Arguments
    ///
    /// * `input_size`: the expected size of vectors to the circuit.
    ///
    /// returns: BC
    /// 
    pub fn new(input_size: usize) -> Self {
        let proof_size = input_size * 2 + 3;
        let total_size = input_size + proof_size;
        let test_rng = ark_std::test_rng();
        Self {
            input_size,
            proof_size,
            total_size,
            rng: Box::new(test_rng),
        }
    }
    
    /// Perform lagrange interpolation on paris of coordinates and returns the coefficients.
    /// 
    /// # Arguments 
    /// 
    /// * `x_vals`: a vector of field elements, represent x coordinates for point evaluations.
    /// * `y_vals`: a vector of field elements, represent y coordinates for point evaluations.
    /// 
    /// returns: a vector of field elements: the coefficients as field elements.
    ///
    fn lagrange_interpolate_coeffs(x_vals: &Vec<F>, y_vals: &Vec<F>) -> Vec<F> {
        // Set the length of the coeffs.
        let n = x_vals.len();
        
        // Initialize the coeffs to be zeros.
        let mut coeffs = vec![F::zero(); n];

        // Iterate over each Lagrange basis polynomial L_i(x).
        for i in 0..n {
            // Initialize term to be 1 (constant term of L_i(x)).
            let mut term_coeffs = vec![F::one()]; 

            // Construct L_i(x) as the product of terms (x - x_j) / (x_i - x_j).
            for j in 0..n {
                if i != j {
                    // Set the term for (x - x_j).
                    let mut term = vec![-x_vals[j], F::one()]; 
                    let denominator = x_vals[i] - x_vals[j];

                    // Multiply term by (x - x_j) / (x_i - x_j).
                    term = term.iter().map(|&x| x / denominator).collect();

                    // Multiply the terms to get the polynomial representation of L_i(x).
                    term_coeffs = BC::multiply_polynomials(&term_coeffs, &term);
                }
            }

            // Multiply L_i(x) by the corresponding y_i and add it to the total polynomial.
            let y_term = y_vals[i];
            for k in 0..term_coeffs.len() { coeffs[k] += term_coeffs[k] * y_term; }
        }
        
        // Return the coefficients.
        coeffs
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
    /// 
    fn multiply_polynomials(p1: &Vec<F>, p2: &Vec<F>) -> Vec<F> {
        // Reserve space for the multiplied polynomial.
        let mut result = vec![F::zero(); p1.len() + p2.len() - 1];

        // Perform multiplications of the coefficients.
        for i in 0..p1.len() {
            for j in 0..p2.len() {
                result[i + j] += p1[i] * p2[j];
            }
        }

        result
    }

    
    /// Given a value, find the basis of the lagrange interpolation on this value.
    /// 
    /// This is to find the portion in lagrange interpolation formula without y values.
    /// 
    /// # Arguments 
    /// 
    /// * `r`: a field element.
    /// 
    /// returns: a vector of field elements: the desired lagrange basis values.
    /// 
    fn lagrange_basis_polynomial(&self, r: F) -> Vec<F> {
        // Lagrange basis helper function.
        fn lagrange_basis(input_size: usize, j: usize, r: F) -> F {
            let mut base = F::one();
            let j_f = F::from(j as u32);
            for m in 0..=input_size {
                if m != j {
                    let m_f = F::from(m as u32);
                    base *= (r - m_f) / (j_f - m_f);
                }
            }
            base
        }

        // Compute the Lagrange basis values and return them in a vector.
        (0..=self.input_size)
            .map(|j| lagrange_basis(self.input_size, j, r))
            .collect()
    }

    
    /// Randomly sample an element from the field larger than integer x.
    /// 
    /// # Arguments 
    /// 
    /// * `x`: an u32 integer.
    /// 
    /// returns: a field element.
    /// 
    pub fn random_larger_than(&mut self, x: u32) -> F {
        loop {
            let r = F::rand(self.rng.as_mut());
            let val = r.into_bigint();
            if val > BigInt::from(x) {
                return r;
            }
        }
    }

    /// Generate the proof for a message.
    /// 
    /// The proof is format of x || c || p.
    /// 
    /// # Arguments 
    /// 
    /// * `message`: a vector of u64 integers.
    /// 
    /// returns: a vector of field elements: necessary portions for linear access of the proof.
    /// 
    pub fn proof_gen(&mut self, message: &Vec<u64>) -> Vec<F> {
        // Convert the message to field elements.
        let f_message: Vec<F> = message.into_iter().map(|x| F::from(*x)).collect();

        // Sample random values for the constants.
        let f_1c = vec![self.random_larger_than(0)];
        let f_2c = vec![self.random_larger_than(0)];

        // Build the evaluations results.
        let y_1 = [f_1c.as_slice(), f_message.as_slice()].concat();
        let y_2 = [f_2c.as_slice(), f_message.as_slice()].concat();

        // The evaluation result is on [0, ..., n].
        let x = (0..message.len() + 1).map(|i| F::from(i as u32)).collect();

        // Compute the f1 polynomial.
        let f_1 = BC::lagrange_interpolate_coeffs(&x, &y_1);

        // Compute the f2 polynomial. This needs to be mutable as we decrease constant by 1.
        let mut f_2 = BC::lagrange_interpolate_coeffs(&x, &y_2);

        // Perform the minus one.
        f_2[0] -= F::one();

        // Compute the polynomial p.
        let p = BC::multiply_polynomials(&f_1, &f_2);

        // Return message || constants || p coefficients.
        [f_message, f_1c, f_2c, p].concat()
    }

    /// Generate verification queries based on a random value.
    /// 
    /// # Arguments 
    /// 
    /// * `r`: a field element.
    /// 
    /// returns: a tuple of queries for verification. 
    /// Here the size of the tuple is determined by the G-gate.
    /// 
    pub fn query_gen(&mut self, r: F) -> (Vec<F>, Vec<F>) {
        // Compute the lagrange basis polynomials.
        let lagrange_basis = self.lagrange_basis_polynomial(r);

        // Define the f1 vector per the design of the circuit.
        let mut f1_vec: Vec<Vec<F>> = vec![vec![F::zero(); self.total_size]; self.input_size + 1];
        f1_vec[0][self.input_size] = F::one();
        for i in 0..self.input_size {
            f1_vec[i + 1][i] = F::one();
        }

        // Multiply with the Lagrange basis and sum up the vectors.
        for (index, each_vec) in f1_vec.iter_mut().enumerate() {
            for value in each_vec.iter_mut() {
                *value *= lagrange_basis[index];
            }
        }

        // Find the correct summarized vector (sum of columns).
        let f1_vec_sum: Vec<F> = (0..self.total_size)
            .map(|col_index| f1_vec.iter().map(|row| row[col_index]).sum())
            .collect();

        // f2 vectors
        let mut f2_vec: Vec<Vec<F>> = vec![vec![F::zero(); self.total_size]; self.input_size + 1];
        f2_vec[0][self.input_size + 1] = F::one();
        for i in 0..self.input_size {
            f2_vec[i + 1][i] = F::one();
        }

        // Multiply with the Lagrange basis and sum up the vectors.
        for (index, each_vec) in f2_vec.iter_mut().enumerate() {
            for value in each_vec.iter_mut() {
                *value *= lagrange_basis[index];
            }
        }

        // Find the correct summarized vector (sum of columns).
        let f2_vec_sum: Vec<F> = (0..self.total_size)
            .map(|col_index| f2_vec.iter().map(|row| row[col_index]).sum())
            .collect();

        (f1_vec_sum, f2_vec_sum)
    }

    /// Perform verification assuming linear access to the proof vector.
    /// 
    /// # Arguments 
    /// 
    /// * `proof`: the generated proof vector.
    /// 
    /// returns: bool, indicating whether the validation was successful.
    /// 
    pub fn verify(&mut self, proof: Vec<F>) -> bool {
        // Sample a value larger than the input size.
        let r = self.random_larger_than(self.input_size as u32);
        
        // Generate the validation queries.
        let queries = self.query_gen(r);

        // Compute proof polynomial at r.
        let p: F = proof[self.input_size + 2..]
            .iter()
            .enumerate()
            .map(|(index, value)| value * &r.pow(&[index as u64]))
            .sum();

        // Compute verification polynomials at r.
        let f1: F = queries
            .0
            .iter()
            .enumerate()
            .map(|(index, value)| value * &proof[index])
            .sum();

        let f2: F = queries
            .1
            .iter()
            .enumerate()
            .map(|(index, value)| value * &proof[index])
            .sum();

        // Compute the G gate over the verification polynomials.
        let pp = f1 * (f2 - F::one());
        
        // Compute queries to verify the final output.
        let mut queries = Vec::new();
        
        // Find random linear combinations of vector to evaluate p(1), p(2), ...
        for i in 1..=self.input_size {
            let mut temp = Vec::new();
            let temp_r = self.random_larger_than(0);

            for j in 0..self.proof_size {
                temp.push(temp_r * F::from(i as u32).pow(&[j as u64])); // Assuming the exponentiation works with f64
            }

            queries.push(temp);
        }
        
        // Summarize the query to perform folding.
        let query: Vec<F> = (0..self.proof_size)
            .map(|col_index| queries.iter().map(|row| row[col_index]).sum())
            .collect();

        // Compute the final value 'c'
        let c: F = proof[self.input_size + 2..]
            .iter()
            .enumerate()
            .map(|(index, &value)| value * query[index])
            .sum();
        
        // The evaluation at random points should equal and c should be zero.
        p == pp && c == F::zero()
    }
}

/// In file testing for simplicity.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // This test is for a validated circuit, the verification should pass.
    fn validated_circuit() {
        let mut binary_check = BC::new(8);

        let message: Vec<u64> = vec![1, 1, 1, 1, 0, 0, 0, 0];

        let proof = binary_check.proof_gen(&message);

        let result = binary_check.verify(proof);

        assert_eq!(result, true);
    }

    #[test]
    // This test is for a validated circuit, the verification should NOT pass.
    fn invalidated_circuit() {
        let mut binary_check = BC::new(10);

        let message: Vec<u64> = vec![1, 1, 1, 1, 0, 0, 0, 0, 0, 100];

        let proof = binary_check.proof_gen(&message);

        let result = binary_check.verify(proof);

        assert_ne!(result, true);
    }
}

fn main() {}