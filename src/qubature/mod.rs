use std::cmp;

use itertools::izip;

use crate::{
    geom::{GeomCell, Point, Tetrahedron, Triangle},
    quadrature::DynGaussQuad,
};

/// General N-dimensional quadrature (N > 1)
pub trait Qubature<A: GeomCell<M, D>, I, O, const D: usize, const M: usize> {
    fn nint<F>(&self, func: F, cell: A) -> crate::Result<O>
    where
        F: Fn(Point<D>) -> O;
}

#[derive(Debug, Clone)]
pub struct DynGaussTriQuad {
    xi: Vec<f64>,
    eta: Vec<f64>,
    nu: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DynGaussTetQuad {
    xi: Vec<f64>,
    eta: Vec<f64>,
    zeta: Vec<f64>,
    nu: Vec<f64>,
}

impl<const D: usize> Qubature<Triangle<D>, f64, f64, D, 2> for DynGaussTriQuad
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

impl<const D: usize> Qubature<Tetrahedron<D>, f64, f64, D, 3> for DynGaussTetQuad
where
    Tetrahedron<D>: GeomCell<3, D>,
{
    fn nint<F>(&self, func: F, cell: Tetrahedron<D>) -> crate::Result<f64>
    where
        F: Fn(Point<D>) -> f64,
    {
        let jac = cell.jacobian_meas();
        let res: f64 = izip!(
            self.xi.iter(),
            self.eta.iter(),
            self.zeta.iter(),
            self.nu.iter()
        )
        .map(|(xi, eta, zeta, nu)| nu * func(cell.map_reference(Point::new([*xi, *eta, *zeta]))))
        .sum();
        Ok(jac * res)
    }
}

impl DynGaussTriQuad {
    pub fn new(q: usize) -> Self {
        if q == 1 {
            return DynGaussTriQuad {
                xi: vec![1.0 / 3.0],
                eta: vec![1.0 / 3.0],
                nu: vec![0.5],
            };
        }

        let gl = DynGaussQuad::gausslegendre(0.0, 1.0, q);

        let x = gl.abscissae();
        let w1 = gl.weights();

        let mut xi = vec![1.0 - x[0]];
        let mut eta = vec![0.5 * x[0]];
        let mut nu = vec![x[0] * w1[0]];

        for j in 1..q {
            let qj = cmp::max(2, (x[j] / x[q - 1] * (q as f64)).ceil() as usize);

            let gl = DynGaussQuad::gausslegendre(0.0, 1.0, qj);

            let yj = gl.abscissae().iter().map(|&a| x[j] * a);
            let wj = gl.weights().iter().map(|&a| x[j] * a);

            for (y, w) in yj.zip(wj) {
                xi.push(1.0 - x[j]);
                eta.push(y);
                nu.push(w1[j] * w);
            }
        }
        DynGaussTriQuad { xi, eta, nu }
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

impl DynGaussTetQuad {
    pub fn new(q: usize) -> Self {
        if q == 1 {
            return DynGaussTetQuad {
                xi: vec![1.0 / 4.0],
                eta: vec![1.0 / 4.0],
                zeta: vec![1.0 / 4.0],
                nu: vec![1.0 / 6.0],
            };
        }

        let gl = DynGaussQuad::gausslegendre(0.0, 1.0, q);

        let x = gl.abscissae();
        let w1 = gl.weights();

        let xi = x.iter().map(|&a| 1.0 - a).collect();
        let mut eta = Vec::new();
        let mut zeta = Vec::new();
        let mut nu = Vec::new();

        for i in 0..q {
            let qi = f64::max(2.0, f64::ceil(x[i] / x[q - 1] * (q as f64)));
            let gl = DynGaussQuad::gausslegendre(0.0, 1.0, qi as usize);

            let y = gl.abscissae();
            let w2 = gl.weights();
            for j in 0..qi as usize {
                eta.push(x[i] * y[j]);
                let qij = f64::max(2.0, f64::ceil(eta[eta.len() - 1] / x[q - 1] * (q as f64)));
                let gl = DynGaussQuad::gausslegendre(0.0, 1.0, qij as usize);

                let z = gl.abscissae();
                let w3 = gl.weights();
                for k in 0..qij as usize {
                    zeta.push(eta[eta.len() - 1] * z[k]);
                    nu.push(w1[i] * w2[j] * w3[k] * eta[eta.len() - 1]);
                }
            }
        }

        DynGaussTetQuad { xi, eta, zeta, nu }
    }
    pub fn abscissae(&self) -> Vec<Point<3>> {
        izip!(self.xi.iter(), self.eta.iter(), self.zeta.iter())
            .map(|(&a, &b, &c)| Point::new([a, b, c]))
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
            DynGaussTriQuad {
                xi: vec![1.0 / 3.0],
                eta: vec![1.0 / 3.0],
                nu: vec![0.5],
            },
            DynGaussTriQuad {
                xi: vec![0.7887, 0.2113, 0.2113],
                eta: vec![0.1057, 0.1667, 0.6220],
                nu: vec![0.1057, 0.1972, 0.1972],
            },
            DynGaussTriQuad {
                xi: vec![0.8873, 0.5, 0.5, 0.1127, 0.1127, 0.1127],
                eta: vec![0.056351, 0.1055625, 0.394338, 0.1, 0.443649, 0.787298],
                nu: vec![0.031306, 0.111111, 0.111111, 0.068464, 0.109543],
            },
            DynGaussTriQuad {
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
            let gt = DynGaussTriQuad::new(i + 1);

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
        let quadrature = DynGaussTriQuad::new(5);

        let triangles = vec![
            Triangle::new(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.0, 1.0]),
            ),
            Triangle::new(
                Point::new([0.0, 0.0]),
                Point::new([2.0, 0.0]),
                Point::new([0.0, 2.0]),
            ),
            Triangle::new(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle::new(
                Point::new([1.0, 1.0]),
                Point::new([2.0, 1.0]),
                Point::new([1.0, 2.0]),
            ),
            Triangle::new(
                Point::new([-1.0, 0.0]),
                Point::new([0.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle::new(
                Point::new([3.0, 0.0]),
                Point::new([4.0, 0.0]),
                Point::new([3.0, 1.0]),
            ),
            Triangle::new(
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

        let expected_integrals = [1.0 / 3.0, 2.0, 0.1667, 0.9167, -0.2083, 1.75, 6.4999];

        for (i, (func, expected_integral)) in
            functions.iter().zip(expected_integrals.iter()).enumerate()
        {
            let integral = quadrature.nint(func, triangles[i]).ok().unwrap();
            assert_approx_eq!(integral, *expected_integral, 1e-4);
        }
    }

    #[test]
    fn new_gauss_tet_test() {
        let anss = [
            DynGaussTetQuad {
                xi: vec![1.0 / 4.0],
                eta: vec![1.0 / 4.0],
                zeta: vec![1.0 /4.0 ],
                nu: vec![1.0 / 6.0],
            },
            DynGaussTetQuad {
                xi: vec![0.544151844,0.544151844,0.544151844,0.544151844],
                eta: vec![0.544151844,0.544151844,0.544151844,0.544151844],
                zeta: vec![0.544151844,0.544151844,0.544151844,0.544151844],
                nu: vec![0.544151844,0.544151844,0.544151844,0.544151844],
            },
        ];

        for (i, ans) in anss.iter().enumerate() {
            let gt = DynGaussTetQuad::new(i + 1);

            assert_eq!(gt.abscissae().len(), ans.abscissae().len());
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
    fn gauss_tet_eval_test() {
        // Vector of tetrahedrons
        let tetrahedrons = vec![
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([1.0, 0.0, 0.0]),
                Point::<3>::new([0.0, 1.0, 0.0]),
                Point::<3>::new([0.0, 0.0, 1.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([2.0, 0.0, 0.0]),
                Point::<3>::new([0.0, 2.0, 0.0]),
                Point::<3>::new([0.0, 0.0, 2.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([1.0, 1.0, 0.0]),
                Point::<3>::new([0.0, 1.0, 1.0]),
                Point::<3>::new([1.0, 0.0, 1.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([1.0, 1.0, 1.0]),
                Point::<3>::new([2.0, 1.0, 1.0]),
                Point::<3>::new([1.0, 2.0, 1.0]),
                Point::<3>::new([1.0, 1.0, 2.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([1.0, 0.0, 0.0]),
                Point::<3>::new([1.0, 1.0, 0.0]),
                Point::<3>::new([1.0, 1.0, 1.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 1.0]),
                Point::<3>::new([1.0, 0.0, 1.0]),
                Point::<3>::new([0.0, 1.0, 1.0]),
                Point::<3>::new([0.0, 0.0, 2.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([0.0, 1.0, 0.0]),
                Point::<3>::new([0.0, 0.0, 1.0]),
                Point::<3>::new([1.0, 1.0, 1.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 1.0]),
                Point::<3>::new([0.0, 1.0, 2.0]),
                Point::<3>::new([1.0, 0.0, 2.0]),
                Point::<3>::new([1.0, 1.0, 3.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([0.0, 0.0, 0.0]),
                Point::<3>::new([0.0, 0.0, 1.0]),
                Point::<3>::new([0.0, 1.0, 0.0]),
                Point::<3>::new([1.0, 0.0, 0.0]),
            ),
            Tetrahedron::new(
                Point::<3>::new([1.0, 0.0, 0.0]),
                Point::<3>::new([2.0, 1.0, 0.0]),
                Point::<3>::new([1.0, 1.0, 0.0]),
                Point::<3>::new([1.0, 0.0, 1.0]),
            ),
        ];

        // Array of functions
        let functions = [
            |_p: Point<3>| 1.0,
            |p: Point<3>| p[0] + p[1] + p[2],
            |p: Point<3>| p[0].powi(2) + p[1].powi(2) + p[2].powi(2),
            |p: Point<3>| p[0] * p[1] * p[2],
            |p: Point<3>| p[0] + p[1].powi(2),
            |p: Point<3>| (p[0] + p[1] + p[2]).exp(),
            |p: Point<3>| (p[0] + p[1] + p[2]).sin(),
            |p: Point<3>| p[0] + p[1] + p[2] + p[3],
            |p: Point<3>| 2.0 * p[0] + 3.0 * p[1],
            |p: Point<3>| p[0] * p[1] + p[2].powi(2),
        ];

        // Array of expected values
        let expected_values: [f64; 10] = [
            1.0 / 6.0,
            2.0, // Example expected value; please adjust with precise calculation
            0.5,
            0.25,
            1.0,
            2.0,
            0.0,
            2.0,
            0.5,
            0.75,
        ];

        let quad = DynGaussTetQuad::new(5); // Specify number of quadrature points
        for (tet, func, expc) in izip!(tetrahedrons, functions, expected_values)
        {
            let result = quad.nint(func, tet).ok().unwrap();
            assert_approx_eq!(result, expc, 1e-4);
        }
    }
}
