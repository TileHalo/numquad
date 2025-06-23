# Numerical quadrature rules (Numquad) ![build status](https://github.com/TileHalo/numquad/actions/workflows/rust.yml/badge.svg)
Numquad provides various numerical integration algorithms for use.

## Usage
Add the following under `[dependencies]` in your cargo.toml
```toml
numquad = 0.1
```

To use for example the [Michalski-Mosig variant](http://dx.doi.org/10.1080/09205071.2015.1129915)
of $\tanh\sinh$ quadrature include the following:
```rust
use numquad::{Integral, quadrature::{TanhSinh}};
...

let integrator = TanhSinh::new();
let func = |x: f64| f64::powi(x, 2);

let result = integrator.integrate(func, Some(0), Some(1)); // Ok(0.3333...)
```

WARNING: This project is in early stages and until 1.0.0 subject to breaking
chances etc.

## Use cases
This crate can be used to integrate integrals in the form $\int_a^b f(x) dx$
where $a, b \in \R$ and $f: \R \to B \subset \C$, semi-infinite integrals where
either $a$ or $b$ is infinite and infinite intervals (both are infinite).
Furthermore, integration on certain simplicies (triangle, quadrilateral, tetrahedrons,
and hexahedrons) is supported.

Currently used and planned quadratures are:
 - [x] Trapezoidal rule
 - [ ] Gaussian quadratures
    - [x] Gauss-Legendre
    - [ ] Gauss-Jacobi
    - [ ] Chebyshev-Gauss
    - [ ] Gauss-Laguerre
    - [ ] Gauss-Hermite
 - [x] Gauss-Legendre
     - [x] Real line
     - [ ] Triangle
     - [ ] Quadrilateral
     - [ ] Tetrahedron
     - [ ] Hexahedron
 - [x] Michalski-Mosig $\tanh\sinh$
 - [x] Michalski-Mosig $\exp\sinh$ (semi-infinite interval)
 - [x] Michalski-Mosig $\sinh\sinh$ (infinite interval)


## Contributing
Feel free to submit bug reports, feature requests, and the best of all,
pull requests to those issues.
