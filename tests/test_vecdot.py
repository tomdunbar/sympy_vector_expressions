import unittest as u
from common import CommonTest
from sympy import S, Integer
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude, Grad
)

class test_VecDot(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        assert VecDot(v1, zero) == S.Zero
        assert VecDot(zero, v1) == S.Zero
        assert isinstance(VecDot(v1, v1), VecDot)
        assert isinstance(VecDot(v1, v2), VecDot)
        assert isinstance(VecDot(v1, v1.norm), VecDot)
        assert isinstance(VecDot(v1.norm, v1), VecDot)
        assert isinstance(VecDot(nabla, v1), VecDot)
        assert isinstance(VecDot(v1, vn1), VecDot)
        assert isinstance(VecDot(vn1, v1), VecDot)
        assert isinstance(VecDot(v1 + v2, v1), VecDot)
        assert isinstance(VecDot(v1.mag * v2, v1), VecDot)
        assert isinstance(VecDot(v1 ^ v2, v1), VecDot)
        assert isinstance(VecDot((v1 ^ v2) * v1.mag, v1), VecDot)

        def func(*args):
            with self.assertRaises(TypeError):
                VecDot(*args)
        
        func(x, v1)
        func(v1, x)
        func(v1, v1.mag)
        func(zero, 0)
        func(nabla, nabla)
        func(v1, nabla)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        assert not VecDot(v1, zero).is_Vector
        assert VecDot(v1, zero).is_scalar
        assert not VecDot(v1, v2).is_Vector
        assert not VecDot(vn1, vn2).is_Vector
        assert not VecDot(v1, vn1).is_Vector
        assert not VecDot(v1 ^ v2, v1).is_Vector
    
    def test_reverse(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        assert VecDot(v1, v2).reverse == VecDot(v2, v1)
        assert VecDot(vn1, v2).reverse == VecDot(v2, vn1)
        assert VecDot(v1, vn2).reverse == VecDot(vn2, v1)
        assert VecDot(vn1, vn2).reverse == VecDot(vn2, vn1)
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert VecDot(vn1, vn2).doit() - vn1.dot(vn2) == 0
        assert VecDot(nabla, vn2).doit() - divergence(vn2) == 0
        assert isinstance(VecDot(v1, v1).doit(), VecPow)
        assert isinstance(VecDot(vn1, vn1).doit(), Integer)

        expr = VecDot(v1, v2.norm).doit()
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VecMul), (VecMul, VectorSymbol)]
        
        expr = VecDot(v1, v2.norm).doit(deep=False)
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, Normalize), (Normalize, VectorSymbol)]
        
        expr = VecDot(v1, VecAdd(vn2, vn1)).doit(deep=False)
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VecAdd), (VecAdd, VectorSymbol)]
        expr = VecDot(v1, VecAdd(vn2, vn1)).doit()
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VectorAdd), (VectorAdd, VectorSymbol)]
        
        expr = VecDot(vn1, vn2 + vn1).doit()
        assert expr == (x + 2 * y + 3 * z + 14)
    
    def test_expand(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols
        nabla = Nabla()

        # expand divergence
        expr = nabla & (x * a + b)
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            (nabla & (x * a)) + (nabla & b)
        )

        expr = nabla & (x * a)
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True, prod=True),
            x * (nabla & a) + (Grad(x) & a)
        )

        expr = nabla & (a / x)
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True, quotient=True),
            (x * (nabla & a) - (Grad(x) & a)) / x**2
        )

        # expand additive terms in dot products
        expr = a & (b + c)
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            (a & b) + (a & c)
        )

        expr = (a + b) & c
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            (a & c) + (b & c)
        )

        expr = (a + b) & (c + d)
        assert self._check_args(
            expr.expand(dot=False),
            expr
        )
        assert self._check_args(
            expr.expand(dot=True),
            (a & c) + (b & c) + (b & d) + (a & d)
        )

        expr = a & (b + (c & (d + e) * f.mag) * g)
        assert self._check_args(
            expr.expand(dot=False),
            a & (b + (c & (d * f.mag + e * f.mag)) * g)
        )
        assert self._check_args(
            expr.expand(dot=True),
            (a & b) + f.mag * (a & g) * (c & d) + f.mag *  (a & g) * (c & e)
        )
        

if __name__ == "__main__":
    u.main()