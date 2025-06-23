#![recursion_limit = "256"]

use std::fmt;

pub mod geom;
pub mod quadrature;
pub mod polynomial;


pub trait Integral<I, O> {
    fn integrate<F>(&self, func: F, a: Option<I>, b: Option<I>) -> Result<O>
    where
        F: Fn(I) -> O;
}

/// Internal result type
pub type Result<T> = std::result::Result<T, IntegrationError>;

/// Various errors that can occur while integrating
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IntegrationError {
    /// Integrator does not support integralsn $\int_{-\infty}^\infty f(x) dx$
    InfiniteIntegral,
    /// Integrator does not support integralsn $\int_{a}^\infty f(x) dx$ where $a \in \R$.
    SemiInfiniteIntegral,
    /// Value NaN reached
    Nan,
    /// Integral returned infinity
    Infty,
}

impl fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            IntegrationError::InfiniteIntegral => {
                write!(f, "Integrator does not support infinite intervals")
            }
            IntegrationError::SemiInfiniteIntegral => {
                write!(f, "Integrator does not support semi-infinite intervals")
            }
            IntegrationError::Nan => write!(f, "Integration resulted in NaN"),
            IntegrationError::Infty => write!(f, "Integration resulted in infinity"),
        }
    }
}
