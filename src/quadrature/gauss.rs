use std::{f64::consts::PI, iter, ops};
use crate::polynomial;

use super::Quadrature;

/// Standard Gaussian quadrature for intervals [a, b], with q points.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct DynGaussQuad {
    a: f64,
    b: f64,
    q: usize,
    abscissae: Vec<f64>,
    weights: Vec<f64>,
}

/// Constant gaussian quadrature. Gives
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct GaussQuad<const Q: usize> {
    a: f64,
    b: f64,
    q: usize,
    abscissae: [f64; Q],
    weights: [f64; Q],
}

impl DynGaussQuad {

    /// Returns the configured interval of the quadrature
    pub fn interval(&self) -> (f64, f64) {
        (self.a, self.b)
    }

    /// Returns the number of points for the quadrature
    pub fn npoints(&self) -> usize {
        self.q
    }

    /// Returns abscissae (points of evaluation) for the quadrature
    pub fn abscissae(&self) -> &Vec<f64> {
        &self.abscissae
    }
    /// Returns weights for the quadrature
    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }
    /// Build new Gauss-Legendre quadrature for interval [a, b] and size q
    pub fn gausslegendre(a: f64, b: f64, q: usize) -> Self {
        // let q = Q;
        if q == 1 {
            return Self {
                a,
                b,
                q,
                weights: vec![(b - a)],
                abscissae: vec![0.0 + (a + b) / 2.0],
            };
        } else if q == 2 {
            return Self {
                a,
                b,
                q,
                weights: vec![(b - a)/2.0, (b - a)/2.0],
                abscissae: vec![
                    -(b - a) / 2.0 * 0.57735 + (a + b) / 2.0,
                    (b - a) / 2.0 * 0.57735 + (a + b) / 2.0,
                ],
            };
        }
        let mut quad = Self {
            a,
            b,
            q,
            abscissae: vec![0.0; q],
            weights: vec![0.0; q],
        };

        let n = q as f64;

        let lg = polynomial::generate_legendre(q);

        // TODO: Make this use the pre-calculated list
        let (pn, pn1) = (&lg[q], &lg[q - 1]);

        let tol = 1.0e-15;
        for (i, (x, w)) in quad.abscissae.iter_mut().zip(quad.weights.iter_mut()).enumerate() {
            let mut theta = PI * ((n - i as f64) / n) - 0.25;
            let mut xi = f64::cos(theta);
            let mut err = 2.0 * tol;
            while err >= tol {
                // Find Legendre polynomials for current guess
                let prev = f64::cos(theta);
                theta -= f64::sin(theta) * pn.eval(xi) / (n * xi * pn.eval(xi) - n * pn1.eval(xi));
                err = f64::abs(f64::cos(theta) - prev);
                xi = f64::cos(theta);
            }
            *x = (b - a) * 0.5 * xi + (b + a) * 0.5;
            *w = (b - a) * f64::sin(theta).powi(2) / (n * xi * pn.eval(xi) - n * pn1.eval(xi)).powi(2);
        }
        quad
    }
}

impl<
        I: ops::Sub<I, Output = I>
            + ops::Add<I, Output = I>
            + ops::Mul<f64, Output = I>
            + ops::Mul<O, Output = O>
            + ops::MulAssign<I>
            + Copy
            + num::Float,
        O: num::Zero
            + ops::Mul<f64, Output = O>
            + ops::AddAssign<O>
            + ops::Sub<O, Output = O>
            + num_complex::ComplexFloat
            + iter::Sum,
    > Quadrature<I, O> for DynGaussQuad
{
    const DEFAULTN: usize = 24;
    fn nint<F>(&self, func: F, start: Option<I>, end: Option<I>, _: usize) -> crate::Result<O>
    where
        F: Fn(I) -> O,
    {
        let (c, d) = self.interval();
        match (start, end) {
            (Some(a), Some(b)) => {
                let phi = |t| (b - a)*(1.0/(d - c)*(t - c)) + I::one() * a;
                let gq: O = self
                    .abscissae
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(&x, &w): (&f64, &f64)| func(phi(x)) * w)
                    .sum();
                Ok( ((b - a) * gq) * (1.0/(d - c)))
            }
            (None, None) => Err(crate::IntegrationError::InfiniteIntegral),
            _ => Err(crate::IntegrationError::SemiInfiniteIntegral),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use itertools::izip;
    use num_complex::Complex;
    use std::f64::consts;

    use super::*;


    #[test]
    fn gausslegendre_test_11() {
        let anss = vec![
            (vec![2.0], vec![0.0]),
            (vec![1.0, 1.0], vec![-0.57735, 0.57735]),
            (
                vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
                vec![-0.774597, 0.0, 0.774597],
            ),
            (
                vec![0.347855, 0.652145, 0.652145, 0.347855],
                vec![-0.861136, -0.339981, 0.339981, 0.861136],
            ),
            (
                vec![0.236927, 0.478629, 0.56889, 0.478629, 0.236927],
                vec![-0.90618, -0.538469, 0.0, 0.538469, 0.90618],
            ),
        ];
        let testers = (1..anss.len()).map(|x| DynGaussQuad::gausslegendre(-1.0, 1.0, x));

        for (test, ans) in testers.zip(anss) {
            assert_eq!(test.npoints(), test.abscissae().len());
            assert_eq!(test.npoints(), test.weights().len());
            assert_eq!(test.npoints(), ans.0.len());
            for (t, a) in test.weights().iter().zip(ans.0.iter()) {
                assert_approx_eq!(t, a);
            }
            for (t, a) in test.abscissae().iter().zip(ans.1.iter()) {
                assert_approx_eq!(t, a);
            }
        }
    }
    #[test]
    fn gausslegendre_test_01() {
        let anss = vec![
            (vec![1.0], vec![0.5]),
            (vec![0.5, 0.5], vec![0.2113, 0.7887]),
            (
                vec![0.2778, 0.4444, 0.2778],
                vec![0.1127, 0.5, 0.8873],
            ),
            (
                vec![0.1739, 0.3261, 0.3261, 0.1739],
                vec![0.069432, 0.330009, 0.669991, 0.930568],
            ),
            (
                vec![0.1185, 0.2393, 0.2844, 0.2393, 0.1185],
                vec![0.04691, 0.230765, 0.5, 0.769235, 0.95309],
            ),
        ];
        let testers = (1..anss.len()).map(|x| DynGaussQuad::gausslegendre(0.0, 1.0, x));

        for (test, ans) in testers.zip(anss) {
            assert_eq!(test.npoints(), test.abscissae().len());
            assert_eq!(test.npoints(), test.weights().len());
            assert_eq!(test.npoints(), ans.0.len());
            for (t, a) in test.weights().iter().zip(ans.0.iter()) {
                assert_approx_eq!(t, a, 1e-4);
            }
            for (t, a) in test.abscissae().iter().zip(ans.1.iter()) {
                assert_approx_eq!(t, a, 1e-4);
            }
        }
    }

    // Test Gauss-Legendre quadrature generation and basic integration.
    #[test]
    fn test_gauss_legendre_integration_real() {
        let a = 0.0;
        let b = 1.0;
        let q = 7; // Number of abscissas


let functions: &[fn(f64) -> f64] = &[
            |_x| 1.0,
            |x| x.powi(2),
            |x| x.powi(3),
            |x| x.sin(),
            |x| x.cos(),
            |x| x.exp(),
            |x| (-x).exp(),
            |x| (x + 1.0).ln(),
            |x| 1.0 / (x.powi(2) + 1.0),
        ];

        // Intervals: [start, end]
        let intervals: &[(f64, f64)] = &[
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, consts::PI),
            (0.0, consts::FRAC_PI_2),
            (0.0, 1.0),
            (0.0, 5.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ];

        // Expected results for each function
        let expected_results: &[f64] = &[
            1.0,
            1.0 / 3.0,
            1.0 / 4.0,
            2.0,
            1.0,
            consts::E - 1.0,
            1.0 - (-5.0_f64).exp(),
            2.0*consts::LN_2 - 1.0,
            consts::FRAC_PI_4,
        ];



        // Create a Gauss-Legendre quadrature for integrating on the interval [a, b].
        let rule = DynGaussQuad::gausslegendre(a, b, q);

        for (func, (a, b), res) in izip!(functions, intervals, expected_results) {
            let ans = rule.nint(func, Some(*a), Some(*b), 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
    }

    #[test]
    fn test_gauss_legendre_integration_complex() {
        let a = 0.0;
        let b = std::f64::consts::PI;
        let q = 5; // Number of abscissas

        // Create a Gauss-Legendre quadrature for integrating on the interval [a, b].
        let gauss_quad = DynGaussQuad::gausslegendre(a, b, q);

        // Test integration of a function f(x) = e^(i * x) over [0, π].
        let result = gauss_quad
            .nint(|x| Complex::new(0.0, x).exp(), Some(0.0), Some(PI), 0)
            .unwrap();

        // Expected result is the integral of the complex function e^(i * x) from 0 to π,
        // which is 0 + i2 if evaluated correctly.
        let expected = Complex::new(0.0, 2.0);

        // Compare both real and imaginary parts separately
        assert_approx_eq!(result.re, expected.re);
        assert_approx_eq!(result.im, expected.im);
    }
}
