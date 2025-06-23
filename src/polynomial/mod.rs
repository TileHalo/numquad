use itertools::{EitherOrBoth, Itertools};
use std::ops;
/// Generates Legendre polynomials up to (and including) of order Q
pub fn generate_legendre(q: usize) -> Vec<Polynomial<f64>> {
    let mut poly = vec![Polynomial::default(); q + 1];
    let (mut prev, mut pprev) = (Polynomial::default(), Polynomial::default());
    for (i, elem) in poly.iter_mut().enumerate() {
        if i == 0 {
            *elem = Polynomial::new([1.0]);
            pprev = elem.clone();
        } else if i == 1 {
            *elem = Polynomial::new([0.0, 1.0]);
            prev = elem.clone();
        } else {
            *elem = (Polynomial::new([0.0, 1.0]) * prev.clone() * (2.0 * (i as f64 - 1.0) + 1.0)
                - pprev.clone() * (i as f64 - 1.0))
                * (1.0 / (i as f64));

            pprev = prev;
            prev = elem.clone();
        }
    }

    poly
}

/// Generic one-dimensional polynomial, either
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Polynomial<P> {
    pub exponents: Vec<usize>,
    pub coeffs: Vec<P>,
}

impl<P> Polynomial<P> {
    pub fn new<I>(itr: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        let coeffs: Vec<P> = itr.into_iter().collect();
        let exponents: Vec<usize> = (0..coeffs.len()).collect();
        Polynomial { coeffs, exponents }
    }
}

impl <P: ops::Add<P, Output=P> + num::Float + ops::Mul<P, Output=P> + std::iter::Sum> Polynomial<P> {
    pub fn eval(&self, x: P) -> P {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(p, a)| *a * x.powi(p as i32))
            .sum()
    }
}

impl<Q: Copy, P: Copy + ops::Mul<Q, Output = P>> ops::Mul<Q> for Polynomial<P> {
    type Output = Polynomial<P>;

    fn mul(self, rhs: Q) -> Self::Output {
        Polynomial::new(self.coeffs.iter().map(|&a| a * rhs))
    }
}

impl<Q: Copy, P: Copy + ops::Div<Q, Output = P>> ops::Div<Q> for Polynomial<P> {
    type Output = Polynomial<P>;

    fn div(self, rhs: Q) -> Self::Output {
        Polynomial::new(self.coeffs.iter().map(|&a| a/rhs))
    }
}

impl<
        Q: Copy,
        P: num::Zero + Copy + ops::Mul<Q, Output = P> + PartialEq,
    > ops::Mul<Polynomial<P>> for Polynomial<Q>
{
    type Output = Polynomial<P>;

    fn mul(self, rhs: Polynomial<P>) -> Self::Output {
        let mut poly = Polynomial::new(vec![
            <P as num::Zero>::zero();
            self.coeffs.len() * rhs.coeffs.len()
        ]);
        for (i, &f) in self.coeffs.iter().enumerate() {
            for (j, &g) in rhs.coeffs.iter().enumerate() {
                poly.coeffs[i + j] = g * f;
            }
        }
        poly.coeffs = poly
            .coeffs
            .into_iter()
            .rev()
            .skip_while(|&x| x == <P as num::Zero>::zero())
            .collect();
        poly.coeffs.reverse();
        poly
    }
}

impl<
        Q: Copy,
        P: Copy + ops::Sub<Q, Output = P> + PartialEq,
    > ops::Sub<Q> for Polynomial<P>
{
    type Output = Polynomial<P>;

    fn sub(self, rhs: Q) -> Self::Output {
        let mut poly = Polynomial::new(self.coeffs);
        poly.coeffs[0] = poly.coeffs[0] - rhs;
        poly
    }
}

impl<
        Q: Copy,
        P: num::Zero + Copy + ops::Sub<Q, Output = P> + PartialEq,
    > ops::Sub<Polynomial<Q>> for Polynomial<P>
{
    type Output = Polynomial<P>;

    fn sub(self, rhs: Polynomial<Q>) -> Self::Output {
        let v1 = self.coeffs.into_iter();
        let v2 = rhs.coeffs.into_iter();

        Polynomial::new(v1.zip_longest(v2).map(|a| match a {
            EitherOrBoth::Both(x, y) => x - y,
            EitherOrBoth::Left(x) => x,
            EitherOrBoth::Right(x) => <P as num::Zero>::zero() - x,
        }))
    }
}

impl<
        Q: Copy,
        P: num::Zero + Copy + ops::Add<Q, Output = P> + PartialEq,
    > ops::Add<Q> for Polynomial<P>
{
    type Output = Polynomial<P>;

    fn add(self, rhs: Q) -> Self::Output {
        let mut poly = Polynomial::new(self.coeffs);
        poly.coeffs[0] = poly.coeffs[0] + rhs;
        poly
    }
}

impl<
        Q: Copy,
        P: num::Zero + Copy + ops::Add<Q, Output = P> + PartialEq,
    > ops::Add<Polynomial<Q>> for Polynomial<P>
{
    type Output = Polynomial<P>;

    fn add(self, rhs: Polynomial<Q>) -> Self::Output {
        let v1 = self.coeffs.into_iter();
        let v2 = rhs.coeffs.into_iter();

        Polynomial::new(v1.zip_longest(v2).map(|a| match a {
            EitherOrBoth::Both(x, y) => x + y,
            EitherOrBoth::Left(x) => x,
            EitherOrBoth::Right(x) => <P as num::Zero>::zero() + x,
        }))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_eval_test() {
        assert_eq!(Polynomial::new([1.0, 0.0]).eval(0.0), 1.0);
        assert_eq!(Polynomial::new([0.0, 1.0]).eval(0.0), 0.0);
        assert_eq!(Polynomial::new([0.0, 1.0]).eval(1.0), 1.0);
    }

    #[test]
    fn generate_legendre_test() {
        let anss = vec![
            Polynomial::new([1.0]),
            Polynomial::new([0.0, 1.0]),
            Polynomial::new([-0.5, 0.0, 1.5]),
            Polynomial::new([0.0, -1.5, 0.0, 2.5]),
            Polynomial::new([3.0 / 8.0, 0.0, -30.0 / 8.0, 0.0, 35.0 / 8.0]),
            Polynomial::new([0.0, 15.0 / 8.0, 0.0, -70.0 / 8.0, 0.0, 63.0 / 8.0]),
            Polynomial::new([
                -5.0 / 16.0,
                0.0,
                105.0 / 16.0,
                0.0,
                -315.0 / 16.0,
                0.0,
                231.0 / 16.0,
            ]),
        ];

        let testers = (0..(anss.len() - 1)).map(|x| generate_legendre(x).last().unwrap().clone());

        for (test, ans) in testers.zip(anss) {
            assert_eq!(test, ans);
        }
    }

}
