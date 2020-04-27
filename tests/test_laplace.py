import unittest as u
from common import CommonTest, x, y, z
from sympy import Add, sqrt, S, log, sin, pi
from sympy.vector import (
    CoordSys3D, Vector, VectorZero as VZero, 
    curl, gradient
)

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Grad, Laplace
)


class test_Laplace(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        # gradient of scalar fields
        # TODO: separate scalar fields from constants symbols, numbers, ...
        assert isinstance(Laplace(nabla, x), Laplace)
        assert isinstance(Laplace(x), Laplace)
        assert isinstance(nabla.laplace(x), Laplace)
        assert isinstance(v1.lap, Laplace)
        # laplacian of scalar field is scalar
        assert not Laplace(x).is_Vector
        assert Laplace(x).is_Vector_Scalar
        # laplacian of vector field is vector
        assert Laplace(v1).is_Vector
        assert not Laplace(v1).is_Vector_Scalar
        assert Laplace(vn1).is_Vector
        assert not Laplace(vn1).is_Vector_Scalar

        # laplacian of scalar field
        v = C.x * C.y * C.z
        assert isinstance(Laplace(v), Laplace)
        # laplacian of vector field
        assert isinstance(Laplace(vn1), Laplace)

        # laplacian of nabla
        with self.assertRaises(NotImplementedError) as context:
            Laplace(nabla)
        
        def func(*args):
            with self.assertRaises(TypeError) as context:
                Laplace(*args)

        # first argument is not nabla
        func(x, v1)
        func(v1, v1)
        func(v1, x)
        
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        # NOTE: the following tests come from:
        # https://www.plymouth.ac.uk/uploads/production/document/path/3/3744/PlymouthUniversity_MathsandStats_the_Laplacian.pdf
        # Be careful, a few exercises in the pdf are incorrectly solved.

        # laplacian of a scalar field
        expr = C.x * C.y**2 + C.z**3
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            2 * C.x + 6 * C.z
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = 3 * C.x**3 * C.y**2 * C.z**3
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            18 * C.x**3 * C.y**2 * C.z + 18 * C.x * C.y**2 * C.z**3 + 6 * C.x**3 * C.z**3
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = sqrt(C.x * C.z) + C.y
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            -sqrt(C.x * C.z) / (4 * C.z**2) - sqrt(C.x * C.z) / (4 * C.x**2)
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = sqrt(C.x**2 + C.y**2 + C.z**2)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit().simplify(),
            2 / expr
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = 1 / sqrt(C.x**2 + C.y**2 + C.z**2)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit().simplify(),
            S.Zero
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        # product of scalar fields
        expr = (C.x + C.y + C.z) * (C.x - 2 * C.z)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            S(-2)
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = (2 * C.x - 5 * C.y + C.z) * (C.x - 3 * C.y + C.z)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            S(36)
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = (C.x**2 - C.y) * (C.x + C.z)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            6 * C.x + 2 * C.z
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        expr = (C.y - C.z) * (C.x**2 + C.y**2 + C.z**2)
        assert isinstance(Laplace(expr), Laplace)
        assert self._check_args(
            Laplace(expr).doit(),
            10 * C.y - 10 * C.z
        )
        assert isinstance(Laplace(expr).doit(deep=False), VecDot)

        # laplacian of vector fields
        v = 3 * C.z**2 * C.i + C.x * C.y * C.z * C.j + C.x**2 * C.z**2 * C.k
        assert isinstance(Laplace(v), Laplace)
        assert self._check_args(
            Laplace(v).doit(),
            6 * C.i + (2 * C.x**2 + 2 * C.z**2) * C.k
        )
        assert isinstance(Laplace(v).doit(deep=False), Vector)

        v = C.x**3 * C.y * C.i + log(C.z) * C.j + log(C.x * C.y) * C.k
        assert isinstance(Laplace(v), Laplace)
        assert self._check_args(
            Laplace(v).doit(),
            6 * C.x * C.y * C.i - (1 / C.z**2) * C.j - ((1 / C.y**2) + (1 / C.x**2)) * C.k
        )
        assert isinstance(Laplace(v).doit(deep=False), Vector)

        v = 3 * C.x**2 * C.z * C.i - sin(pi * C.y) * C.j + log(2 * C.x**3) * C.k
        assert isinstance(Laplace(v), Laplace)
        assert self._check_args(
            Laplace(v).doit().subs({C.x: 1, C.y: -2, C.z: 1}),
            6 * C.i - 3 * C.k
        )
        assert isinstance(Laplace(v).doit(deep=False), Vector)

        v = log(C.y) * C.i + C.z**2 * C.j - sin(2 * pi * C.x) * C.k
        assert isinstance(Laplace(v), Laplace)
        assert self._check_args(
            Laplace(v).doit().subs({C.x: 1, C.y: 1, C.z: pi}),
            -C.i + 2 * C.j
        )
        assert isinstance(Laplace(v).doit(deep=False), Vector)

if __name__ == "__main__":
    u.main()