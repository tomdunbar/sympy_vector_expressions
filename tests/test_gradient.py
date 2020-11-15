import unittest as u
from common import CommonTest
from sympy import Add
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
    VecDot, VecCross, Grad
)

class test_Grad(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        # gradient of scalar fields
        # TODO: separate scalar fields from constants symbols, numbers, ...
        assert isinstance(Grad(nabla, x), Grad)
        assert isinstance(Grad(x), Grad)
        assert isinstance(nabla.grad(x), Grad)

        # gradient of scalar field
        v = C.x * C.y * C.z
        assert isinstance(Grad(v), Grad)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                Grad(*args)

        # gradient of vector fields
        func(nabla, v1)
        func(v1)

        # gradient of nabla
        with self.assertRaises(NotImplementedError) as context:
            Grad(nabla)
        
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        v = C.x * C.y * C.z
        assert gradient(v) == Grad(v).doit()

        expr = v * (VecDot(vn1, vn2))
        assert isinstance(Grad(expr).doit(deep=False), Grad)
        assert isinstance(Grad(expr).doit(), Vector)
    
    def test_expand(self):
        x, y, z = self.symbols
        assert Grad(x).expand(gradient=True) == Grad(x)

        expr = Grad(x + y)
        assert self._check_args(
            expr.expand(gradient=False),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True),
            Grad(x) + Grad(y)
        )
        assert self._check_args(
            expr.expand(gradient=True, prod=True),
            Grad(x) + Grad(y)
        )

        expr = Grad(x * y)
        assert self._check_args(
            expr.expand(gradient=False),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True, prod=True),
            y * Grad(x) + x * Grad(y)
        )

        expr = Grad(x * y * z)
        assert self._check_args(
            expr.expand(gradient=False),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True, prod=True),
            z * y * Grad(x) + x * z * Grad(y) + y * x * Grad(z)
        )

        expr = Grad(x * y + z)
        assert self._check_args(
            expr.expand(gradient=False),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True),
            Grad(z) + Grad(x * y) 
        )
        assert self._check_args(
            expr.expand(gradient=True, prod=True),
            Grad(z) + y * Grad(x) + x * Grad(y)
        )

        expr = Grad(x / y)
        assert self._check_args(
            expr.expand(gradient=False),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True),
            expr
        )
        assert self._check_args(
            expr.expand(gradient=True, quotient=True),
            (y * Grad(x) - x * Grad(y)) / y**2
        )

    
if __name__ == "__main__":
    u.main()