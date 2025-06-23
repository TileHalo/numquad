//! This is where all geometry primitives live

use num::Complex;

use std::{iter, ops};

/// An basic geometry shape/cell.
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
pub type Triangle<const D: usize> = (Point<D>, Point<D>, Point<D>);

const REFERENCE_TRIANGLE: Triangle<2> = (
    Vector::<f64, 2>([0.0, 0.0]),
    Vector::<f64, 2>([1.0, 0.0]),
    Vector::<f64, 2>([0.0, 1.0]),
);

impl<T, const D: usize> Vector<T, D> {
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

impl<const D: usize> GeomCell<2, D> for Triangle<D> {
    /// Reference cell
    type REFT = Triangle<2>;

    fn refcell() -> Self::REFT {
        REFERENCE_TRIANGLE
    }
    /// Jacobian measure to reference cell.
    fn jacobian_meas(self) -> f64 {
        match D {
            0..=1 => panic!("This is not possible"),
            2 => 0.0,
            3 => 0.0,
            _ => panic!("Please use feature nönnönnöö"), // This should be translated into macro
                                                         // tomfoolery
        }
    }
    /// Map to reference cell
    fn map_reference(self, p: Point<2>) -> Point<D> {
        self.0 + (self.1 - self.0) * p[0] + (self.1 - self.0) * p[1]
    }
}
