import unittest as u
from common import CommonTest
from sympy import S, Integer, Pow, Symbol
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude, D
)

class test_VecPow(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()

        assert VecPow(1, v1.mag) == S.One
        assert isinstance(VecPow(x, 1), Symbol)
        assert isinstance(VecPow(x, 2), Pow)
        assert isinstance(VecPow(3, 2), Integer)
        assert isinstance(VecPow(3, 2), Integer)
        
        # base is instance of VectorExpr with is_Vector=False
        assert isinstance(VecPow(v1.mag, 1), Magnitude)
        assert isinstance(VecPow(v1.mag, -1), VecPow)
        assert isinstance(VecPow(v1.mag, 2), VecPow)
        assert isinstance(VecPow(v1 & v2, 2), VecPow)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecPow(*args)

        # base is instance of VectorExpr with is_Vector=True
        func(v1, 1)
        func(v1, 2)
        func(v1.norm, 2)
        func(v1 + v2, 2)
        func(v1 + v2, 2)
        func(v1 + v2, 2)
        func(v1 ^ v2, 2)
        func(v1.mag * v2, 2)

        # exponent is instance of VectorExpr with is_Vector=True
        func(2, v1)
        func(2, v1 ^ v2)
        func(2, v1.norm)

        # base is not instance of VectorExpr
        assert isinstance(VecPow(2, v1.mag), VecPow)
        assert isinstance(VecPow(x, v1.mag), VecPow)

        # base and exp are instances of VectorExpr with is_Vector=False
        assert isinstance(VecPow(v1.mag, v1.mag), VecPow)
        assert isinstance(VecPow(v1.mag, (v1 & v2)), VecPow)

    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        expr = VecPow(VecDot(vn1, vn2), v1.mag)
        assert isinstance(expr, VecPow)
        assert isinstance(expr.doit(deep=False), VecPow)
        assert isinstance(expr.doit(), Pow)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert not VecPow(v1.mag, 1).is_Vector
        assert not VecPow(v1.mag, -1).is_Vector
        assert not VecPow(v1.mag, 2).is_Vector
        assert not VecPow(v1 & v2, 2).is_Vector

if __name__ == "__main__":
    u.main()