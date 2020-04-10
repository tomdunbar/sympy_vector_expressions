import unittest as u
from common import CommonTest, x, y, z
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
        assert isinstance(VecPow(x, 1), Symbol)
        assert isinstance(VecPow(x, 2), Pow)
        assert isinstance(VecPow(3, 2), Integer)
        assert isinstance(VecPow(3, 2), Integer)
        
        # base is instance of VectorExpr with is_Vector=False
        assert isinstance(VecPow(self.v1.mag, 1), Magnitude)
        assert isinstance(VecPow(self.v1.mag, -1), VecPow)
        assert isinstance(VecPow(self.v1.mag, 2), VecPow)
        assert isinstance(VecPow(self.v1 & self.v2, 2), VecPow)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecPow(*args)

        # base is instance of VectorExpr with is_Vector=True
        func(self.v1, 1)
        func(self.v1, 2)
        func(self.v1.norm, 2)
        func(self.v1 + self.v2, 2)
        func(self.v1 + self.v2, 2)
        func(self.v1 + self.v2, 2)
        func(self.v1 ^ self.v2, 2)
        func(self.v1.mag * self.v2, 2)

        # exponent is instance of VectorExpr with is_Vector=True
        func(2, self.v1)
        func(2, self.v1 ^ self.v2)
        func(2, self.v1.norm)
    
    def test_doit(self):
        expr = VecPow(VecDot(self.vn1, self.vn2), self.v1.mag)
        assert isinstance(expr, VecPow)
        assert isinstance(expr.doit(deep=False), VecPow)
        assert isinstance(expr.doit(), Pow)
    
    def test_is_vector(self):
        assert not VecPow(self.v1.mag, 1).is_Vector
        assert not VecPow(self.v1.mag, -1).is_Vector
        assert not VecPow(self.v1.mag, 2).is_Vector
        assert not VecPow(self.v1 & self.v2, 2).is_Vector

if __name__ == "__main__":
    u.main()