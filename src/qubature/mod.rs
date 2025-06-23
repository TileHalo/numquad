use std::cmp;

use itertools::izip;

use crate::{
    geom::{GeomCell, Point, Triangle},
    quadrature::GaussQuadrature,
};

/// General N-dimensional quadrature (N > 1)
pub trait Qubature<A: GeomCell<M, D>, I, O, const D: usize, const M: usize> {
    fn nint<F>(&self, func: F, cell: A) -> crate::Result<O>
    where
        F: Fn(Point<D>) -> O;
}

pub struct GaussTriQuadrature {
    xi: Vec<f64>,
    eta: Vec<f64>,
    nu: Vec<f64>,
}

impl<const D: usize> Qubature<Triangle<D>, f64, f64, D, 2> for GaussTriQuadrature
where
    Triangle<D>: GeomCell<2, D>,
{
    fn nint<F>(&self, func: F, cell: Triangle<D>) -> crate::Result<f64>
    where
        F: Fn(Point<D>) -> f64,
    {
        let jac = cell.jacobian_meas();
        let res: f64 = izip!(self.xi.iter(), self.eta.iter(), self.nu.iter())
            .map(|(xi, eta, nu)| nu * func(cell.map_reference(Point::new([*xi, *eta]))))
            .sum();
        Ok(jac * res)
    }
}

impl GaussTriQuadrature {
    pub fn new(q: usize) -> Self {
        if q == 1 {
            return GaussTriQuadrature {
                xi: vec![1.0 / 3.0],
                eta: vec![1.0 / 3.0],
                nu: vec![0.5],
            };
        }

        let gl = GaussQuadrature::gausslegendre(0.0, 1.0, q);

        let x = gl.abscissae();
        let w1 = gl.weights();

        let mut xi = vec![1.0 - x[0]];
        let mut eta = vec![0.5 * x[0]];
        let mut nu = vec![x[0] * w1[0]];

        for j in 1..q {
            let qj = cmp::max(2, (x[j] / x[q - 1] * (q as f64)).ceil() as usize);

            let gl = GaussQuadrature::gausslegendre(0.0, 1.0, qj);

            let yj = gl.abscissae().iter().map(|&a| x[j] * a);
            let wj = gl.weights().iter().map(|&a| x[j] * a);

            for (y, w) in yj.zip(wj) {
                xi.push(1.0 - x[j]);
                eta.push(y);
                nu.push(w1[j] * w);
            }
        }
        GaussTriQuadrature { xi, eta, nu }
    }

    pub fn abscissae(&self) -> Vec<Point<2>> {
        self.xi
            .iter()
            .zip(self.eta.iter())
            .map(|(&a, &b)| Point::new([a, b]))
            .collect()
    }
    pub fn weights(&self) -> &Vec<f64> {
        &self.nu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn new_gausstri_test() {
        let anss = [
            GaussTriQuadrature {
                xi: vec![1.0 / 3.0],
                eta: vec![1.0 / 3.0],
                nu: vec![0.5],
            },
            GaussTriQuadrature {
                xi: vec![0.7887, 0.2113, 0.2113],
                eta: vec![0.1057, 0.1667, 0.6220],
                nu: vec![0.1057, 0.1972, 0.1972],
            },
            GaussTriQuadrature {
                xi: vec![0.8873, 0.5, 0.5, 0.1127, 0.1127, 0.1127],
                eta: vec![0.056351, 0.1055625, 0.394338, 0.1, 0.443649, 0.787298],
                nu: vec![0.031306, 0.111111, 0.111111, 0.068464, 0.109543],
            },
            GaussTriQuadrature {
                xi: vec![
                    0.930568, 0.66991, 0.66991, 0.330009, 0.330009, 0.330009, 0.069432, 0.069432,
                    0.069432, 0.069432,
                ],
                eta: vec![
                    0.034716, 0.069739, 0.260270, 0.075509, 0.334995, 0.594481, 0.064611, 0.307096,
                    0.623472, 0.865957,
                ],
                nu: vec![
                    0.012076, 0.053804, 0.053804, 0.060685, 0.097096, 0.060685, 0.028150, 0.052775,
                    0.052775, 0.028150,
                ],
            },
        ];

        for (i, ans) in anss.iter().enumerate() {
            let gt = GaussTriQuadrature::new(i + 1);

            for (p, ps) in gt.abscissae().iter().zip(ans.abscissae().iter()) {
                assert_approx_eq!(p[0], ps[0], 1e-4);
                assert_approx_eq!(p[1], ps[1], 1e-4);
            }
            for (p, ps) in gt.weights().iter().zip(ans.weights().iter()) {
                assert_approx_eq!(p, ps, 1e-4);
            }
        }
    }

    #[test]
    fn gauss_tri_eval_test() {
        let quadrature = GaussTriQuadrature::new(5);

        let triangles = vec![
            (
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.0, 1.0]),
            ),
            (
                Point::new([0.0, 0.0]),
                Point::new([2.0, 0.0]),
                Point::new([0.0, 2.0]),
            ),
            (
                Point::new([0.0, 0.0]),
                Point::new([1.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            (
                Point::new([1.0, 1.0]),
                Point::new([2.0, 1.0]),
                Point::new([1.0, 2.0]),
            ),
            (
                Point::new([-1.0, 0.0]),
                Point::new([0.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            (
                Point::new([3.0, 0.0]),
                Point::new([4.0, 0.0]),
                Point::new([3.0, 1.0]),
            ),
            (
                Point::new([0.0, 0.0]),
                Point::new([0.0, 2.0]),
                Point::new([3.0, 0.0]),
            ),
        ];

        let functions = [
            |p: Point<2>| p[0] + p[1],
            |_: Point<2>| 1.0,
            |p: Point<2>| p[0].powi(2),
            |p: Point<2>| p[1].powi(2),
            |p: Point<2>| p[0] * p[1],
            |p: Point<2>| p[0] + p[1].powi(2),
            |p: Point<2>| p[0].powi(2) + p[1],
        ];

        let expected_integrals = [
            1.0 / 3.0,
            2.0,
            0.1667,
            0.9167,
            -0.2083,
            1.75,
            6.4999,
        ];

        for (i, (func, expected_integral)) in
            functions.iter().zip(expected_integrals.iter()).enumerate()
        {
            let integral = quadrature.nint(func, triangles[i]).ok().unwrap();
            assert_approx_eq!(integral, *expected_integral, 1e-4);
        }
    }
}
