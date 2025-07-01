//! Tanh-Sinh quadrature for Michalski-Mosig

use std::ops;

use super::Quadrature;


/// Michalski-Mosig version of tanh-sinh/exp-sinh/sinh-sinh double ended quadrature with improvements
/// from Dr. Robert  van Engelen from Genivia Labs [qtsh](https://www.genivia.com/files/qthsh.pdf)
/// This quadrature accepts finite, semi-infinite, and infinite intervals, and uses
/// tanh-sinh, exp-sinh, and sinh-sinh rules respectively
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DEQuad {
    eta: usize,
    nmax: usize,
    kappa: f64,
    maxlev: usize,
    eps: f64,
}


impl DEQuad {
    pub fn new() -> Self {
        Self {
            eta: 1,
            nmax: 24,
            kappa: 1.0e-15,
            maxlev: 5,
            eps: 1.0e-7,
        }
    }
    fn tanhsinh<
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
            + num_complex::ComplexFloat,
        F,
    >(
        &self,
        func: F,
        a: I,
        b: I,
        nmax: usize,
    ) -> crate::Result<O>
    where
        F: Fn(I) -> O,
        f64: ops::Mul<
            <O as num_complex::ComplexFloat>::Real,
            Output = <O as num_complex::ComplexFloat>::Real,
        >,
    {
        let nm: usize = if nmax == 0 { 24 } else { nmax };
        let maxlev = if self.maxlev == 0 { 6 } else { self.maxlev };
        let c = (a + b) * 0.5;
        let d = (b - a) * 0.5;
        let mut h = 2.0;

        let mut s = func(c);

        let (mut fp, mut fm) = (O::zero(), O::zero());

        let eps: f64 = if self.eps == 0.0 { 1.0e-9 } else { self.eps };

        for k in 0..maxlev {
            let mut p = O::zero();
            h /= 2.0;
            let mut eh = f64::exp(h);
            let mut t = I::one() * eh;
            if k > 0 {
                eh *= eh;
            }
            let ieh = I::one() * eh; // Change type
            for _ in 0..nm {
                let u = I::exp(I::one() / t - t);
                let r = u / (I::one() + u) * 2.0;
                let w = (t + I::one() / t) * r / (I::one() + u);
                let x = d * r;
                if a + x > a {
                    let y = func(a + x);
                    if y.is_finite() {
                        fp = y;
                    }
                }
                if b + x > b {
                    let y = func(b - x);
                    if y.is_finite() {
                        fm = y;
                    }
                }
                let q = w * (fp + fm);
                p += q;
                t *= ieh;
                if q.abs() <= eps * p.abs() {
                    break;
                }
            }

            let v = s - p;
            s += p;
            if v.abs() <= (s * 10.0 * eps).abs() {
                break;
            }
        }

        Ok(d * h * s)
    }
    fn expsinh<
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
            + ops::MulAssign<O>
            + ops::Sub<O, Output = O>
            + num_complex::ComplexFloat,
        F,
    >(
        &self,
        func: F,
        a: I,
        nmax: usize,
    ) -> crate::Result<O>
    where
        F: Fn(I) -> O,
        f64: ops::Mul<
            <O as num_complex::ComplexFloat>::Real,
            Output = <O as num_complex::ComplexFloat>::Real,
        >,
    {
        let nm: usize = if nmax == 0 { 24 } else { nmax };
        let maxlev = if self.maxlev == 0 { 6 } else { self.maxlev };
        let c = a;
        let d = I::one();
        let mut h = 2.0;

        let mut s = func(a + d);

        let eps: f64 = if self.eps == 0.0 { 1.0e-9 } else { self.eps };

        for k in 0..maxlev {
            let mut q = O::zero();
            let mut p = q;
            h /= 2.0;
            let mut eh = f64::exp(h);
            let mut t = I::one() * 0.5 * eh;
            if k > 0 {
                eh *= eh;
            }
            let ieh = I::one() * eh;
            for _ in 0..nm {
                q = O::zero();
                let r = I::exp(t - I::one() / (t * 4.0));
                let w = r;
                let (x1, x2) = (c + d / r, c + d * r);
                if x1 == c {
                    break;
                }
                let (y1, y2) = (func(x1), func(x2));
                if y1.is_finite() {
                    q += (I::one() / w) * y1;
                }
                if y2.is_finite() {
                    q += w * y2;
                }
                q *= (t + I::one() / (t * 4.0)) * O::one();
                p += q;
                t *= ieh;
                if q.abs() <= eps * p.abs() {
                    break;
                }
            }

            let v = s - p;
            s += p;
            if v.abs() <= (s * 10.0 * eps).abs() {
                break;
            }
        }

        Ok(d * h * s)
    }

    fn sinhsinh<
        I: ops::Sub<I, Output = I>
            + ops::Add<I, Output = I>
            + ops::Mul<f64, Output = I>
            + ops::Mul<O, Output = O>
            + ops::MulAssign<I>
            + Copy
            + num::Float,
        O: ops::Mul<f64, Output = O>
            + ops::AddAssign<O>
            + ops::MulAssign<O>
            + ops::Sub<O, Output = O>
            + num_complex::ComplexFloat,
        F,
    >(
        &self,
        func: F,
        nmax: usize,
    ) -> crate::Result<O>
    where
        F: Fn(I) -> O,
        f64: ops::Mul<
            <O as num_complex::ComplexFloat>::Real,
            Output = <O as num_complex::ComplexFloat>::Real,
        >,
    {
        let nm: usize = if nmax == 0 { 24 } else { nmax };
        let maxlev = if self.maxlev == 0 { 6 } else { self.maxlev };
        let mut h = 2.0;

        let mut s = func(I::zero());

        let eps: f64 = if self.eps == 0.0 { 1.0e-9 } else { self.eps };

        for k in 0..maxlev {
            let mut q = O::zero();
            let mut p = q;
            h /= 2.0;
            let mut eh = f64::exp(h);
            let mut t = I::one() * eh * 0.5;
            if k > 0 {
                eh *= eh;
            }
            let ieh = I::one() * eh;
            for _ in 0..nm {
                q = O::zero();
                let r = (I::exp(t - I::one() / (t * 4.0))
                    - I::one() / I::exp(t - I::one() / (t * 4.0)))
                    * 0.5;
                let w = (I::exp(t - I::one() / (t * 4.0))
                    + I::one() / I::exp(t - I::one() / (t * 4.0)))
                    * 0.5;
                let (x1, x2) = (-r, r);
                let (y1, y2): (O, O) = (func(x1), func(x2));
                if y1.is_finite() {
                    q += w * y1;
                }
                if y2.is_finite() {
                    q += w * y2;
                }
                q *= (t + I::one() / (t * 4.0)) * O::one();
                p += q;
                t *= ieh;
                if q.abs() <= eps * p.abs() {
                    break;
                }
            }

            let v = s - p;
            s += p;
            if v.abs() <= (s * 10.0 * eps).abs() {
                break;
            }
        }

        Ok(s * h)
    }
}

impl Default for DEQuad {
    fn default() -> Self {
        Self::new()
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
            + ops::MulAssign<O>
            + ops::Sub<O, Output = O>
            + num_complex::ComplexFloat,
    > Quadrature<I, O> for DEQuad
where
    f64: ops::Mul<
        <O as num_complex::ComplexFloat>::Real,
        Output = <O as num_complex::ComplexFloat>::Real,
    >,
{
    const DEFAULTN: usize = 24;
    fn nint<F>(&self, func: F, start: Option<I>, end: Option<I>, nmax: usize) -> crate::Result<O>
    where
        F: Fn(I) -> O,
    {
        match (start, end) {
            (Some(a), Some(b)) => self.tanhsinh(func, a, b, nmax),
            (Some(a), None) => self.expsinh(func, a, nmax),
            (None, Some(a)) => Ok(-self.expsinh(func, a, nmax)?),
            (None, None) => self.sinhsinh(func, nmax),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn tanhsinh_test() {
        let rule = DEQuad::new();
        // Test various integrals from 0 to 1
        let ints = [
            |x: f64| 1.0 / (1.0 + x),
            |x| 4.0 / (1.0 + x * x),
            |x| f64::acos(x),
            |x| f64::sin(x) / x,
            |x| f64::sqrt(x / (1.0 - f64::powf(x, 2.0))),
            |x| 1.0 / f64::sqrt(x),
            |x| 1.0 / f64::sqrt(1.0 - x),
            |x| f64::powf(x, -0.8),
            |x| 1.0 / f64::sqrt(f64::sin(std::f64::consts::PI * x)),
            |x| 1.0 / f64::sqrt(-f64::log10(x)),
        ];

        let cints = [|x| 4.0 / (1.0 + x * x)];

        let resc = [2.0 * f64::atan(33.0 / 56.0)];

        let ress: &[f64] = &[
            f64::ln(2.0),
            std::f64::consts::PI,
            1.0,
            0.946083,
            1.19814,
            2.0,
            2.0,
            5.0,
            1.6692537,
            f64::sqrt(std::f64::consts::PI * std::f64::consts::LN_10),
        ];
        for (i1, res) in ints.iter().zip(ress.iter()) {
            let ans = rule.nint(i1, Some(0.0), Some(1.0), 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
        for (i1, res) in cints.iter().zip(resc.iter()) {
            let ans = rule.nint(i1, Some(2.0), Some(5.0), 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
    }

    #[test]
    fn expsinh_test() {
        let rule = DEQuad::new();
        // Test various integrals from 0 to 1
        let ints = [
            |x| 1.0 / (1.0 + x * x),
            |x: f64| f64::exp(-x) / f64::sqrt(x),
            |x| f64::exp(-1.0 - x) / (1.0 + x),
            |x| (x * x) * f64::exp(-4.0 * x),
            |x| f64::powi(f64::sqrt((x * x) + 9.0) - x, 3),
            |x| f64::exp(-3.0 * x),
        ];

        let cints = [|x: f64| f64::exp(-x) / x];

        let ress: &[f64] = &[
            std::f64::consts::FRAC_PI_2,
            1.772453851,
            0.219383934,
            0.03125,
            30.375,
            0.333333333,
        ];

        let resc: &[f64] = &[0.219383934];
        for (i1, res) in ints.iter().zip(ress.iter()) {
            let ans = rule.nint(i1, Some(0.0), None, 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
        for (i1, res) in cints.iter().zip(resc.iter()) {
            let ans = rule.nint(i1, Some(1.0), None, 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
    }
    #[test]
    fn sinhsinh_test() {
        let rule = DEQuad::new();
        // Test various integrals from 0 to 1
        let ints = [|x| x, |x: f64| f64::exp(-(x.powi(2) + 2.0 * x + 1.0))];

        let ress= [0.0, f64::sqrt(std::f64::consts::PI)];

        for (i1, res) in ints.iter().zip(ress.iter()) {
            let ans = rule.nint(i1, None, None, 24).ok().unwrap();
            assert_approx_eq!(ans, res);
        }
    }

}
