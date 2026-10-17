use crate::circuit::GadgetId;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FlpError {
    ZeroArity {
        gadget: GadgetId,
    },
    ZeroDegree {
        gadget: GadgetId,
    },
    InvalidGadget {
        gadget: GadgetId,
    },
    ArityMismatch {
        gadget: GadgetId,
        call: usize,
        expected: usize,
        actual: usize,
    },
    UnusedGadget {
        gadget: GadgetId,
    },
    NoOutputs,
    DomainTooLarge {
        gadget: GadgetId,
    },
    InputLength {
        expected: usize,
        actual: usize,
    },
    EmptyBounds,
    EmptySet,
    ExceedsField {
        bits: usize,
        limit: usize,
    },
    ChunkTooLarge {
        chunk: usize,
        len: usize,
    },
}

impl fmt::Display for FlpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroArity { gadget } => write!(f, "gadget {} has arity zero", gadget.0),
            Self::ZeroDegree { gadget } => write!(f, "gadget {} has degree zero", gadget.0),
            Self::InvalidGadget { gadget } => {
                write!(f, "circuit calls undeclared gadget {}", gadget.0)
            }
            Self::ArityMismatch {
                gadget,
                call,
                expected,
                actual,
            } => write!(
                f,
                "call {call} to gadget {} passes {actual} inputs but the gadget takes {expected}",
                gadget.0
            ),
            Self::UnusedGadget { gadget } => {
                write!(f, "gadget {} is declared but never called", gadget.0)
            }
            Self::NoOutputs => f.write_str("circuit has no outputs"),
            Self::DomainTooLarge { gadget } => write!(
                f,
                "gadget {} needs a larger evaluation domain than the field supports",
                gadget.0
            ),
            Self::InputLength { expected, actual } => {
                write!(
                    f,
                    "input has length {actual} but the circuit expects {expected}"
                )
            }
            Self::EmptyBounds => f.write_str("range bounds are empty"),
            Self::EmptySet => f.write_str("set of allowed values is empty"),
            Self::ExceedsField { bits, limit } => write!(
                f,
                "values need {bits} bits but the field holds at most {limit} without wrapping"
            ),
            Self::ChunkTooLarge { chunk, len } => {
                write!(f, "square chunk {chunk} exceeds the input length {len}")
            }
        }
    }
}

impl std::error::Error for FlpError {}
