//! flpcp schemes module.

/// Checks whether each bit of the message is a valid binary value (0 or 1).
pub use binary_validation::BinaryValidation;

/// Ensures each element of the message lies within a specified numeric range.
pub use range_validation::RangeValidation;

/// Verifies that the L2 norm of the message does not exceed a threshold.
pub use norm_bound_validation::NormBoundValidation;

/// Utility functions and helper types used by the flpcp schemes.
pub use utilities::utils;

// Internal module declarations
mod binary_validation;
mod norm_bound_validation;
mod range_validation;
mod utilities;
