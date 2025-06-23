//! Numerical quadrature rules.
//! To use a (generic, non-existent) rule `Rule`, one must first call `Rule::new()`,
//! and then either use `Rule::nint` or `Rule::integrate` depending if rule modifications are
//! desired or not.

pub mod gauss;
pub use gauss::GaussQuadrature;

/// Trait that all quadrature rules implement. All quadratures blanket implement the (`Integral`)[`super::Integral`]
/// trait
pub trait Quadrature<I, O> {
    /// Default number of iterations for quadrature
    const DEFAULTN: usize;
    fn nint<F>(&self, func: F, a: Option<I>, b: Option<I>, n: usize) -> crate::Result<O>
    where
        F: Fn(I) -> O;
}

impl<Q: Quadrature<I, O>, I, O> super::Integral<I, O> for Q {
    fn integrate<F>(&self, func: F, a: Option<I>, b: Option<I>) -> crate::Result<O>
    where
        F: Fn(I) -> O,
    {
        self.nint(func, a, b, Self::DEFAULTN)
    }
}
