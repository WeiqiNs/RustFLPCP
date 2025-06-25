use crate::utils::{utils, Message, Proof, Query};
use ark_bls12_381::Fr as F;
use ark_ff::{Field, One, Zero};

/// The binary check circuit struct, containing essential fields.
pub struct RangeValidation {
    lower: usize,
    degree: usize,
    num_gate: usize,
    input_size: usize,
    proof_size: usize,
    total_size: usize,
}

impl RangeValidation {
    /// Create a new RangeValidation instance from the input size.
    ///
    /// # Arguments
    ///
    /// * `input_size`: the expected size of vectors to the circuit.
    ///
    /// returns: RangeValidation
    pub fn new(input_size: usize, lower: usize, upper: usize) -> Self {
        let degree = upper - lower + 1;
        let num_gate = input_size;
        let proof_size = degree * num_gate + 1;
        let total_size = input_size + degree + proof_size;
        Self {
            lower,
            degree,
            num_gate,
            input_size,
            proof_size,
            total_size,
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
    /// returns: a vector of field elements, necessary portions for linear access of the proof.
    pub fn proof_gen(&self, message: Message) -> Proof {
        // Convert the message to field elements.
        let f_message: Vec<F> = message.iter().map(|&x| F::from(x as u64)).collect();

        // Sample `degree` random constants.
        let constants: Vec<F> = (0..self.degree)
            .map(|_| utils::random_larger_than(0))
            .collect();

        // Construct input x points.
        let x: Vec<F> = (0..=f_message.len()).map(|i| F::from(i as u64)).collect();

        // Create attach c at the beginning of message to create y vectors.
        let polynomials: Vec<Vec<F>> = constants
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let mut y = vec![c];
                y.extend_from_slice(&f_message);
                let mut p = utils::lagrange_interpolate_coeffs(&x, &y);
                p[0] -= F::from((self.lower + i) as u64);
                p
            })
            .collect();

        // Compute the resulting polynomial.
        let mut p = polynomials[0].clone();
        for each_p in &polynomials[1..] {
            p = utils::multiply_polynomials(each_p, &p);
        }

        // Return message || constants || polynomial coefficients.
        [f_message, constants, p].concat()
    }

    /// Generate verification queries based on a random value.
    ///
    /// # Arguments
    ///
    /// * `r`: a field element.
    ///
    /// returns: a tuple of queries for verification.
    pub fn query_gen(&self) -> Query {
        // Sample a value larger than the input size.
        let r = utils::random_larger_than(self.input_size);

        // Compute the lagrange basis polynomials.
        let lagrange_basis = utils::lagrange_basis_polynomial(r, self.num_gate);

        // Store f polynomials.
        let mut f_poly_vec: Vec<Vec<F>> = Vec::with_capacity(self.degree);

        for i in 0..self.degree {
            // Build A matrix: (num_gate + 1) x total_size, all zeros.
            let mut a_mat = vec![vec![F::zero(); self.total_size]; self.num_gate + 1];

            // Set the component to multiply with the constant in proof.
            a_mat[0][self.input_size + i] = F::one();

            // Set the component to multiply with the input message in proof.
            for j in 0..self.input_size {
                a_mat[j + 1][j] = F::one();
            }

            // Multiply each row of A by the corresponding Lagrange basis value.
            let f_vec: Vec<Vec<F>> = lagrange_basis
                .iter()
                .zip(a_mat.iter())
                .map(|(&basis, row)| row.iter().map(|&value| basis * value).collect())
                .collect();

            // Sum across rows to get final vector.
            let f_vec_sum: Vec<F> = (0..f_vec[0].len())
                .map(|j| f_vec.iter().map(|row| row[j]).sum::<F>())
                .collect();

            f_poly_vec.push(f_vec_sum);
        }

        // Build p_poly: [0; total_size - proof_size] || [r^0, r^1, ..., r^{proof_size-1}]
        let mut p_poly = vec![F::zero(); self.total_size - self.proof_size];
        p_poly.extend((0..self.proof_size).map(|i| r.pow([i as u64])));

        // Build c_poly
        let mut queries = vec![vec![F::zero(); self.proof_size]; self.num_gate];

        for (i, query_row) in queries.iter_mut().enumerate() {
            let temp_r = utils::random_larger_than(0);
            for j in 0..self.proof_size {
                query_row[j] = temp_r * F::from((i + 1) as u64).pow([j as u64]);
            }
        }

        // Sum over each column and prepend zeros
        let mut c_poly = vec![F::zero(); self.total_size - self.proof_size];
        c_poly.extend((0..self.proof_size).map(|j| queries.iter().map(|row| row[j]).sum::<F>()));

        // Pack the queries.
        Query {
            f_queries: f_poly_vec,
            p_query: p_poly,
            output_query: c_poly,
        }
    }

    /// Perform verification assuming linear access to the proof vector.
    ///
    /// # Arguments
    ///
    /// * `proof`: the generated proof vector.
    /// * `query`: the generated query vectors.
    ///
    /// returns: bool, indicating whether the validation was successful.
    pub fn verify(&self, proof: Proof, query: Query) -> bool {
        // Unpack the query.
        let Query {
            f_queries,
            p_query,
            output_query,
        } = query;

        // Compute a_i = ⟨f_i, proof⟩.
        let a_list: Vec<F> = f_queries
            .iter()
            .map(|f| utils::inner_product(f, &*proof))
            .collect();

        // Compute p_value = a_0 * (a_1 - 1).
        let mut p_value = F::one();
        for (i, a) in a_list.iter().enumerate() {
            p_value *= *a - F::from((self.lower + i) as u64);
        }

        // Evaluate polynomial p: ⟨p_query, proof⟩.
        let p_prime_value = utils::inner_product(&*p_query, &*proof);

        // Evaluate output polynomial: ⟨c_query, proof⟩.
        let c_value = utils::inner_product(&*output_query, &*proof);

        // Accept if both values match and c_value == 0.
        p_value == p_prime_value && c_value.is_zero()
    }
}
