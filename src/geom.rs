//! This is where all geometry primitives live

use num::Complex;

use std::{iter, ops};

/// An basic geometry shape/cell. Any simplex that can be used as integration domain must implement
/// this
pub trait GeomCell<const N: usize, const M: usize> {
    type REFT;
    /// Reference cell
    fn refcell() -> Self::REFT;
    /// Jacobian measure to reference cell.
    fn jacobian_meas(self) -> f64;
    // /// Jacobian measure from arbitrary cell
    // fn jacobian_meas_arb(self, rf: Self) -> f64;
    /// Map to reference cell
    fn map_reference(self, p: Point<N>) -> Point<M>;
    // /// Map to arbitrary reference cell
    // fn map_reference_arb(self, p: Point<N>, rf: Self) -> Point<M>;
}

/// Basic (slow) vector type. Should not be used anywhere outside simple vector arithmetic in
/// low dimensions (i.e. cross product etc.)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<T, const D: usize>([T; D]);
pub type Point<const D: usize> = Vector<f64, D>;

/// Standhard triangle in D-dimensional space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle<const D: usize>(Point<D>, Point<D>, Point<D>);

/// Standhard tetrahedron in D-dimensional space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tetrahedron<const D: usize>(Point<D>, Point<D>, Point<D>, Point<D>);

/// Reference triangle (0, 0), (1, 0), (0, 1)
pub const REFERENCE_TRIANGLE: Triangle<2> = Triangle(
    Vector::<f64, 2>([0.0, 0.0]),
    Vector::<f64, 2>([1.0, 0.0]),
    Vector::<f64, 2>([0.0, 1.0]),
);

/// Reference tetrahedron
pub const REFERENCE_TETRAHEDRON: Tetrahedron<3> = Tetrahedron(
    Vector::<f64, 3>([0.0, 0.0, 0.0]),
    Vector::<f64, 3>([1.0, 0.0, 0.0]),
    Vector::<f64, 3>([0.0, 1.0, 0.0]),
    Vector::<f64, 3>([0.0, 0.0, 1.0]),
);

impl<T, const D: usize> Vector<T, D> {
    /// Build new vector based on input array
    pub fn new(a: [T; D]) -> Self {
        Vector(a)
    }
}

impl<T, const D: usize> ops::Index<usize> for Vector<T, D> {
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl<T, const D: usize> ops::IndexMut<usize> for Vector<T, D> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.0[i]
    }
}

impl<T: Copy + ops::Add<K>, K: Copy, const D: usize> ops::Add<Vector<K, D>> for Vector<T, D> {
    type Output = Vector<<T as ops::Add<K>>::Output, D>;
    fn add(self, rhs: Vector<K, D>) -> Self::Output {
        Vector(std::array::from_fn(|i| self[i] + rhs[i]))
    }
}

// Vector<T> - Vector<K>
impl<T: Copy + ops::Sub<K>, K: Copy, const D: usize> ops::Sub<Vector<K, D>> for Vector<T, D> {
    type Output = Vector<<T as ops::Sub<K>>::Output, D>;
    fn sub(self, rhs: Vector<K, D>) -> Self::Output {
        Vector(std::array::from_fn(|i| self[i] - rhs[i]))
    }
}

// Vector<T> * Scalar<K>
impl<T: Copy + ops::Mul<K>, K: Copy, const D: usize> ops::Mul<K> for Vector<T, D> {
    type Output = Vector<<T as ops::Mul<K>>::Output, D>;
    fn mul(self, rhs: K) -> Self::Output {
        Vector(std::array::from_fn(|i| self[i] * rhs))
    }
}

// Vector<T> / Scalar<K>
impl<T: Copy + ops::Div<K>, K: Copy, const D: usize> ops::Div<K> for Vector<T, D> {
    type Output = Vector<<T as ops::Div<K>>::Output, D>;
    fn div(self, rhs: K) -> Self::Output {
        Vector(std::array::from_fn(|i| self[i] / rhs))
    }
}

// Assign: +=
impl<T: ops::AddAssign<K> + Copy, K: Copy, const D: usize> ops::AddAssign<Vector<K, D>>
    for Vector<T, D>
{
    fn add_assign(&mut self, rhs: Vector<K, D>) {
        for i in 0..D {
            self[i] += rhs[i];
        }
    }
}

// Assign: -=
impl<T: ops::SubAssign<K> + Copy, K: Copy, const D: usize> ops::SubAssign<Vector<K, D>>
    for Vector<T, D>
{
    fn sub_assign(&mut self, rhs: Vector<K, D>) {
        for i in 0..D {
            self[i] -= rhs[i];
        }
    }
}

// ========== Dot + Norm ==========

// Real dot
impl<T: num_complex::ComplexFloat + num_traits::NumAssign + Copy + iter::Sum, const D: usize>
    Vector<T, D>
{
    pub fn dot(&self, rhs: &Vector<T, D>) -> T {
        self.0.iter().zip(rhs.0.iter()).map(|(a, b)| *a * *b).sum()
    }

    pub fn norm(&self) -> T {
        self.dot(self).sqrt()
    }
}

// Complex dot
impl<T: num::Float + num_traits::FloatConst + num_traits::NumAssign, const D: usize>
    Vector<Complex<T>, D>
{
    pub fn conj(self) -> Vector<Complex<T>, D> {
        let mut v = Vector([<Complex<T> as num::Zero>::zero(); D]);
        for i in 0..D {
            v.0[i] = self.0[i].conj()
        }
        v
    }
    pub fn cdot(&self, rhs: &Self) -> Complex<T> {
        self.conj().dot(rhs)
    }

    pub fn cnorm(&self) -> T {
        self.cdot(self).norm_sqr().sqrt()
    }
}

// ========== Cross ==========

impl<T> Vector<T, 3>
where
    T: num::Float,
{
    pub fn cross(&self, rhs: &Self) -> Self {
        Vector([
            self[1] * rhs[2] - self[2] * rhs[1],
            self[2] * rhs[0] - self[0] * rhs[2],
            self[0] * rhs[1] - self[1] * rhs[0],
        ])
    }
}

impl<const D: usize> Triangle<D> {
    pub fn new(a: Point<D>, b: Point<D>, c: Point<D>) -> Self {
        Triangle(a, b, c)
    }
}

impl<const D: usize> Tetrahedron<D> {
    pub fn new(a: Point<D>, b: Point<D>, c: Point<D>, d: Point<D>) -> Self {
        Tetrahedron(a, b, c, d)
    }
}

impl GeomCell<2, 2> for Triangle<2> {
    /// Reference cell
    type REFT = Triangle<2>;

    fn refcell() -> Self::REFT {
        REFERENCE_TRIANGLE
    }
    /// Jacobian measure to reference cell.
    fn jacobian_meas(self) -> f64 {
        let Triangle(p1, p2, p3) = self;
        // Compute edge vectors
        let v1 = (p2[0] - p1[0], p2[1] - p1[1]);
        let v2 = (p3[0] - p1[0], p3[1] - p1[1]);

        // Compute determinant which gives twice the area
        let det = (v1.0 * v2.1) - (v1.1 * v2.0);

        // Jacobian measure (area measure)
        det.abs()
    }
    /// Map to reference cell
    fn map_reference(self, p: Point<2>) -> Point<2> {
        self.0 + (self.1 - self.0) * p[0] + (self.2 - self.0) * p[1]
    }
}

impl GeomCell<2, 3> for Triangle<3> {
    /// Reference cell
    type REFT = Triangle<2>;

    fn refcell() -> Self::REFT {
        REFERENCE_TRIANGLE
    }
    /// Jacobian measure to reference cell.
    fn jacobian_meas(self) -> f64 {
        let u = self.1 - self.0;
        let v = self.2 - self.0;
        let n = u.cross(&v);
        n.norm()
    }
    /// Map to reference cell
    fn map_reference(self, p: Point<2>) -> Point<3> {
        self.0 + (self.1 - self.0) * p[0] + (self.2 - self.0) * p[1]
    }
}

impl GeomCell<3, 3> for Tetrahedron<3> {
    type REFT = Tetrahedron<3>;

    fn refcell() -> Self::REFT {
        REFERENCE_TETRAHEDRON
    }

    fn jacobian_meas(self) -> f64 {
        // Used AI to expand this
        let Tetrahedron(p1, p2, p3, p4) = self;

        // Calculate the differences for the matrix
        let x_diff_1 = p2.0[0] - p1.0[0];
        let x_diff_2 = p3.0[0] - p1.0[0];
        let x_diff_3 = p4.0[0] - p1.0[0];

        let y_diff_1 = p2.0[1] - p1.0[1];
        let y_diff_2 = p3.0[1] - p1.0[1];
        let y_diff_3 = p4.0[1] - p1.0[1];

        let z_diff_1 = p2.0[2] - p1.0[2];
        let z_diff_2 = p3.0[2] - p1.0[2];
        let z_diff_3 = p4.0[2] - p1.0[2];

        // Compute determinant via expanded formula
        f64::abs(x_diff_1 * (y_diff_2 * z_diff_3 - z_diff_2 * y_diff_3)
            - x_diff_2 * (y_diff_1 * z_diff_3 - z_diff_1 * y_diff_3)
            + x_diff_3 * (y_diff_1 * z_diff_2 - z_diff_1 * y_diff_2))
    }

    fn map_reference(self, p: Point<3>) -> Point<3> {
        self.1 * p[0] + self.2 * p[1] + self.3 * p[2] + self.0 * (1.0 - p[1] - p[2] - p[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn jacobian_meas_tri_2d_test() {
        let triangles_2d = vec![
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.0, 1.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([2.0, 0.0]),
                Point::new([0.0, 2.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle(
                Point::new([1.0, 1.0]),
                Point::new([2.0, 1.0]),
                Point::new([1.0, 2.0]),
            ),
            Triangle(
                Point::new([-1.0, 0.0]),
                Point::new([0.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle(
                Point::new([3.0, 0.0]),
                Point::new([4.0, 0.0]),
                Point::new([3.0, 1.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([0.0, 2.0]),
                Point::new([3.0, 0.0]),
            ),
            Triangle(
                Point::new([1.0, 0.0]),
                Point::new([1.0, 2.0]),
                Point::new([3.0, 0.0]),
            ),
            Triangle(
                Point::new([2.0, 1.0]),
                Point::new([4.0, 1.0]),
                Point::new([2.0, 3.0]),
            ),
            Triangle(
                Point::new([1.0, 2.0]),
                Point::new([3.0, 2.0]),
                Point::new([1.0, 3.0]),
            ),
        ];

        let expected_measures_2d = [0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 3.0, 2.0, 2.0, 1.0];

        for (i, triangle) in triangles_2d.iter().enumerate() {
            let measure = Triangle::<2>::jacobian_meas(*triangle);
            assert_approx_eq!(measure, 2.0 * expected_measures_2d[i], 1e-4);
        }
    }

    #[test]
    fn jacobian_meas_tri_3d_test() {
        let triangles_3d = vec![
            Triangle(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([2.0, 0.0, 0.0]),
                Point::new([0.0, 2.0, 0.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 1.0, 0.0]),
                Point::new([-1.0, 1.0, 0.0]),
            ),
            Triangle(
                Point::new([1.0, 1.0, 0.0]),
                Point::new([2.0, 1.0, 0.0]),
                Point::new([1.0, 2.0, 0.0]),
            ),
            Triangle(
                Point::new([-1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([-1.0, 1.0, 0.0]),
            ),
            Triangle(
                Point::new([3.0, 0.0, 0.0]),
                Point::new([4.0, 0.0, 0.0]),
                Point::new([3.0, 1.0, 0.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([0.0, 2.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
            ),
            Triangle(
                Point::new([1.0, 0.0, 0.0]),
                Point::new([1.0, 2.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
            ),
            Triangle(
                Point::new([2.0, 1.0, 0.0]),
                Point::new([4.0, 1.0, 0.0]),
                Point::new([2.0, 3.0, 0.0]),
            ),
            Triangle(
                Point::new([1.0, 2.0, 0.0]),
                Point::new([3.0, 2.0, 0.0]),
                Point::new([1.0, 3.0, 0.0]),
            ),
        ];

        let expected_measures_3d = [0.5, 2.0, 1.0, 0.5, 0.5, 0.5, 3.0, 2.0, 2.0, 1.0];

        for (i, triangle) in triangles_3d.iter().enumerate() {
            let measure = Triangle::<3>::jacobian_meas(*triangle);
            assert_approx_eq!(measure, 2.0 * expected_measures_3d[i], 1e-4);
        }
    }

    #[test]
    fn reference_map_tri_test() {
        let triangles_2d = [
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 0.0]),
                Point::new([0.0, 1.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([2.0, 0.0]),
                Point::new([0.0, 2.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([1.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle(
                Point::new([1.0, 1.0]),
                Point::new([2.0, 1.0]),
                Point::new([1.0, 2.0]),
            ),
            Triangle(
                Point::new([-1.0, 0.0]),
                Point::new([0.0, 1.0]),
                Point::new([-1.0, 1.0]),
            ),
            Triangle(
                Point::new([3.0, 0.0]),
                Point::new([4.0, 0.0]),
                Point::new([3.0, 1.0]),
            ),
            Triangle(
                Point::new([0.0, 0.0]),
                Point::new([0.0, 2.0]),
                Point::new([3.0, 0.0]),
            ),
            Triangle(
                Point::new([1.0, 0.0]),
                Point::new([1.0, 2.0]),
                Point::new([3.0, 0.0]),
            ),
            Triangle(
                Point::new([2.0, 1.0]),
                Point::new([4.0, 1.0]),
                Point::new([2.0, 3.0]),
            ),
            Triangle(
                Point::new([1.0, 2.0]),
                Point::new([3.0, 2.0]),
                Point::new([1.0, 3.0]),
            ),
        ];

        for tri in triangles_2d {
            assert_eq!(
                tri.0,
                Triangle::<2>::map_reference(tri, Point::new([0.0, 0.0]))
            );
            assert_eq!(
                tri.1,
                Triangle::<2>::map_reference(tri, Point::new([1.0, 0.0]))
            );
            assert_eq!(
                tri.2,
                Triangle::<2>::map_reference(tri, Point::new([0.0, 1.0]))
            );
        }
    }

    #[test]
    fn jacobian_meas_tet_test() {
        let tetrahedrons = [
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 2.0, 0.0]),
                Point::new([0.0, 0.0, 2.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 1.0, 1.0]),
                Point::new([2.0, 1.0, 1.0]),
                Point::new([1.0, 2.0, 1.0]),
                Point::new([1.0, 1.0, 2.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 4.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 2.0, 3.0]),
                Point::new([4.0, 5.0, 6.0]),
                Point::new([7.0, 8.0, 9.0]),
                Point::new([10.0, 11.0, 12.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 2.0]),
                Point::new([0.0, 2.0, 2.0]),
                Point::new([1.0, 2.0, 0.0]),
            ),
            Tetrahedron(
                Point::new([-1.0, -1.0, -1.0]),
                Point::new([1.0, -1.0, -1.0]),
                Point::new([-1.0, 1.0, -1.0]),
                Point::new([-1.0, -1.0, 1.0]),
            ),
            Tetrahedron(
                Point::new([2.0, 3.0, 4.0]),
                Point::new([5.0, 3.0, 4.0]),
                Point::new([2.0, 6.0, 4.0]),
                Point::new([2.0, 3.0, 7.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 1.0, 0.0]),
                Point::new([4.0, 1.0, 0.0]),
                Point::new([1.0, 5.0, 0.0]),
                Point::new([1.0, 1.0, 3.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([2.0, 0.0, 0.0]),
                Point::new([0.0, 3.0, 0.0]),
                Point::new([0.0, 0.0, 5.0]),
            ),
        ];

        let expected = [
            1.0,
            4.0, 1.0,
            12.0,
            0.0,
            8.0,
            8.0,
            27.0,
            36.0,
            30.0,
        ];

        for (i, tetrahedron) in tetrahedrons.iter().enumerate() {
            assert_approx_eq!(tetrahedron.jacobian_meas(), expected[i], 1e-4);
        }
    }

    #[test]
    fn reference_map_tet_test() {
        let tetrahedrons = [
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 1.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 0.0]),
                Point::new([0.0, 2.0, 0.0]),
                Point::new([0.0, 0.0, 2.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 1.0, 1.0]),
                Point::new([2.0, 1.0, 1.0]),
                Point::new([1.0, 2.0, 1.0]),
                Point::new([1.0, 1.0, 2.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([3.0, 0.0, 0.0]),
                Point::new([0.0, 1.0, 0.0]),
                Point::new([0.0, 0.0, 4.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 2.0, 3.0]),
                Point::new([4.0, 5.0, 6.0]),
                Point::new([7.0, 8.0, 9.0]),
                Point::new([10.0, 11.0, 12.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([1.0, 0.0, 2.0]),
                Point::new([0.0, 2.0, 2.0]),
                Point::new([1.0, 2.0, 0.0]),
            ),
            Tetrahedron(
                Point::new([-1.0, -1.0, -1.0]),
                Point::new([1.0, -1.0, -1.0]),
                Point::new([-1.0, 1.0, -1.0]),
                Point::new([-1.0, -1.0, 1.0]),
            ),
            Tetrahedron(
                Point::new([2.0, 3.0, 4.0]),
                Point::new([5.0, 3.0, 4.0]),
                Point::new([2.0, 6.0, 4.0]),
                Point::new([2.0, 3.0, 7.0]),
            ),
            Tetrahedron(
                Point::new([1.0, 1.0, 0.0]),
                Point::new([4.0, 1.0, 0.0]),
                Point::new([1.0, 5.0, 0.0]),
                Point::new([1.0, 1.0, 3.0]),
            ),
            Tetrahedron(
                Point::new([0.0, 0.0, 0.0]),
                Point::new([2.0, 0.0, 0.0]),
                Point::new([0.0, 3.0, 0.0]),
                Point::new([0.0, 0.0, 5.0]),
            ),
        ];


        for tet in tetrahedrons {
            assert_eq!(
                tet.0,
                Tetrahedron::map_reference(tet, Point::new([0.0, 0.0, 0.0]))
            );
            assert_eq!(
                tet.1,
                Tetrahedron::map_reference(tet, Point::new([1.0, 0.0, 0.0]))
            );
            assert_eq!(
                tet.2,
                Tetrahedron::map_reference(tet, Point::new([0.0, 1.0, 0.0]))
            );
            assert_eq!(
                tet.3,
                Tetrahedron::map_reference(tet, Point::new([0.0, 0.0, 1.0]))
            );
        }
    }



}
