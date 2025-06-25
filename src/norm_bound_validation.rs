use crate::utilities::{Message, Proof, Query, utils};
use ark_bls12_381::Fr as F;
use ark_ff::{Field, One, Zero};

/// Represents the norm bound validation scheme.
pub struct NormBoundValidation {
    degree: usize,
    num_gate: usize,
    input_size: usize,
    proof_size: usize,
    total_size: usize,
    norm_bound: usize,
    input_bound: usize,
}

impl NormBoundValidation {
    /// Create a new `NormBoundValidation` instance given input size and the bounds for the proof.
    ///
    /// # Arguments
    ///
    /// * `input_size`: the number of elements in the input vector.
    /// * `norm_bound`: the bit length of the L2 norm.
    /// * `input_bound`: the bit length on each input.
    ///
    /// returns: a configured `NormBoundValidation` instance.
    pub fn new(input_size: usize, norm_bound: usize, input_bound: usize) -> Self {
        let degree = 2;
        let norm_bound = norm_bound + 1;
        let num_gate = input_size * (input_bound + 1) + norm_bound;
        let proof_size = num_gate * degree + 1;
        let total_size = num_gate + degree + proof_size;
        Self {
            degree,
            num_gate,
            input_size,
            proof_size,
            total_size,
            norm_bound,
            input_bound,
        }
    }

    /// Attach binary representation to the input message and compute the L2 norm.
    ///
    /// # Arguments
    ///
    /// * `message`: a vector of usize integers.
    ///
    /// returns: a vector of usize integers, with attached binary numbers and L2 norm.
    pub fn prepare_message(&self, message: &Message) -> Message {
        let mut result = message.clone();

        // Append binary representations of each element.
        for &x in message {
            result.extend(utils::int_to_fixed_binary(x, self.input_bound));
        }

        // Compute sum of squares and append its binary representation.
        let norm = message.iter().map(|&x| x * x).sum();
        result.extend(utils::int_to_fixed_binary(norm, self.norm_bound));

        result
    }

    /// Generates a proof that the given message vector consists of integers.
    ///
    /// # Arguments
    ///
    /// * `message`: a vector of usize integers.
    ///
    /// returns: a vector of field elements, in format of message || c || p.
    pub fn proof_gen(&self, message: &Message) -> Proof {
        // Attach the desired component to the message.
        let prepared_message = self.prepare_message(message);
        // Convert the message to field elements.
        let f_message: Vec<F> = prepared_message
            .iter()
            .map(|&x| F::from(x as u64))
            .collect();

        // Sample `degree` random constants.
        let constants: Vec<F> = (0..self.degree)
            .map(|_| utils::random_larger_than(0))
            .collect();

        // Construct input x points.
        let x: Vec<F> = (0..=f_message.len()).map(|i| F::from(i as u64)).collect();

        // Evaluate `degree` number of f polynomials.
        let polynomials: Vec<Vec<F>> = constants
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let mut y = vec![c];
                y.extend_from_slice(&f_message);
                let mut p = utils::lagrange_interpolate_coeffs(&x, &y);
                p[0] -= F::from(i as u64);
                p
            })
            .collect();

        // Compute p = poly[0] * (poly[1] - 1).
        let p = utils::multiply_polynomials(&polynomials[0], &polynomials[1]);

        // Return message || constants || polynomial coefficients.
        [f_message, constants, p].concat()
    }

    /// Generates query vectors for validating the proof.
    ///
    /// returns: a `Query` struct containing the f, p, and c query vectors.
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
            a_mat[0][self.num_gate + i] = F::one();

            // Set the component to multiply with the input message in proof.
            for j in 0..self.num_gate {
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

        // Build p_poly: [0; total_size - proof_size] || [r^0, r^1, ..., r^{proof_size-1}].
        let mut p_poly = vec![F::zero(); self.total_size - self.proof_size];
        p_poly.extend((0..self.proof_size).map(|i| r.pow([i as u64])));

        // Build c_poly.
        let mut queries: Vec<Vec<F>> = Vec::new();

        // These are the query to verify each input corresponds to its binary representation.
        for i in 0..self.input_size {
            let temp_r = utils::random_larger_than(0);
            let mut temp_query = vec![F::zero(); self.total_size];

            temp_query[i] = temp_r;

            // Compute message - binary representation.
            for j in 0..self.input_bound {
                let index = self.input_size + i * self.input_bound + j;
                temp_query[index] = -temp_r * F::from(2).pow([j as u64]);
            }

            queries.push(temp_query);
        }

        // This is the query to verify the norm bound binary representation is correct.
        let mut temp_query = vec![F::one(); self.input_size];
        temp_query.extend(vec![F::zero(); self.total_size - self.input_size]);

        // Evaluate the polynomial from p(1) to p(input_size).
        for i in 1..=self.input_size {
            for j in 0..self.proof_size {
                let index = self.total_size - self.proof_size + j;
                temp_query[index] += F::from(i as u64).pow([j as u64]);
            }
        }

        // Minus the binary representation.
        for i in 0..self.norm_bound {
            let index = self.input_size * (self.input_bound + 1) + i;
            temp_query[index] = -F::from(2).pow([i as u64]);
        }

        let temp_r = utils::random_larger_than(0);
        queries.push(temp_query.iter().map(|&x| temp_r * x).collect());

        // These are the query to verify G-gates should output zero.
        for i in (self.input_size + 1)..=self.num_gate {
            let temp_r = utils::random_larger_than(0);
            let mut temp_query = vec![F::zero(); self.total_size - self.proof_size];

            temp_query
                .extend((0..self.proof_size).map(|j| temp_r * F::from(i as u64).pow([j as u64])));

            queries.push(temp_query);
        }

        // Sum the query in columns.
        let c_poly: Vec<F> = (0..queries[0].len())
            .map(|j| queries.iter().map(|row| row[j]).sum::<F>())
            .collect();

        // Pack the queries.
        Query {
            f_queries: f_poly_vec,
            p_query: p_poly,
            output_query: c_poly,
        }
    }

    /// Verifies that a given proof satisfies the validation scheme.
    ///
    /// # Arguments
    ///
    /// * `proof`: the proof vector to be verified.
    /// * `query`: the query vectors generated by `query_gen`.
    ///
    /// returns: bool, indicating whether the validation was successful.
    pub fn verify(proof: &Proof, query: &Query) -> bool {
        // Unpack the query.
        let Query {
            f_queries,
            p_query,
            output_query,
        } = query;

        // Compute a_i = ⟨f_i, proof⟩.
        let a_list: Vec<F> = f_queries
            .iter()
            .map(|f| utils::inner_product(f, proof))
            .collect();

        // Compute p_value = a_0 * (a_1 - 1).
        let p_value = a_list[0] * (a_list[1] - F::one());

        // Evaluate polynomial p: ⟨p_query, proof⟩.
        let p_prime_value = utils::inner_product(p_query, proof);

        // Evaluate output polynomial: ⟨c_query, proof⟩.
        let c_value = utils::inner_product(output_query, proof);

        // Accept if both values match and c_value == 0.
        p_value == p_prime_value && c_value.is_zero()
    }
}
