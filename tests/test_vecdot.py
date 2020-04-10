import unittest as u
from common import CommonTest, x, y, z
from sympy import S, Integer
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude
)

class test_VecDot(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        assert VecDot(self.v1, self.zero) == S.Zero
        assert VecDot(self.zero, self.v1) == S.Zero
        assert isinstance(VecDot(self.v1, self.v1), VecDot)
        assert isinstance(VecDot(self.v1, self.v2), VecDot)
        assert isinstance(VecDot(self.v1, self.v1.norm), VecDot)
        assert isinstance(VecDot(self.v1.norm, self.v1), VecDot)
        assert isinstance(VecDot(self.nabla, self.v1), VecDot)
        assert isinstance(VecDot(self.v1, self.nabla), VecDot)
        assert isinstance(VecDot(self.v1, self.vn1), VecDot)
        assert isinstance(VecDot(self.vn1, self.v1), VecDot)
        assert isinstance(VecDot(self.v1 + self.v2, self.v1), VecDot)
        assert isinstance(VecDot(self.v1.mag * self.v2, self.v1), VecDot)
        assert isinstance(VecDot(self.v1 ^ self.v2, self.v1), VecDot)
        assert isinstance(VecDot((self.v1 ^ self.v2) * self.v1.mag, self.v1), VecDot)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecDot(*args)
        
        func(x, self.v1)
        func(self.v1, x)
        func(self.v1, self.v1.mag)
        func(self.zero, 0)
    
    def test_is_vector(self):
        assert not VecDot(self.v1, self.zero).is_Vector
        assert VecDot(self.v1, self.zero).is_scalar
        assert not VecDot(self.v1, self.v2).is_Vector
        assert VecDot(self.v1, self.v2).is_scalar
        assert not VecDot(self.vn1, self.vn2).is_Vector
        assert VecDot(self.vn1, self.vn2).is_scalar
        assert not VecDot(self.v1, self.vn1).is_Vector
        assert VecDot(self.v1, self.vn1).is_scalar
        assert not VecDot(self.v1 ^ self.v2, self.v1).is_Vector
        assert VecDot(self.v1 ^ self.v2, self.v1).is_scalar
    
    def test_reverse(self):
        assert VecDot(self.v1, self.v2).reverse == VecDot(self.v2, self.v1)
        assert VecDot(self.vn1, self.v2).reverse == VecDot(self.v2, self.vn1)
        assert VecDot(self.v1, self.vn2).reverse == VecDot(self.vn2, self.v1)
        assert VecDot(self.vn1, self.vn2).reverse == VecDot(self.vn2, self.vn1)
    
    def test_doit(self):
        assert VecDot(self.vn1, self.vn2).doit() - self.vn1.dot(self.vn2) == 0
        assert VecDot(self.vn2, self.nabla).doit() - divergence(self.vn2) == 0
        assert VecDot(self.nabla, self.vn2).doit() - divergence(self.vn2) == 0
        assert isinstance(VecDot(self.v1, self.v1).doit(), VecPow)
        assert isinstance(VecDot(self.vn1, self.vn1).doit(), Integer)

        expr = VecDot(self.v1, self.v2.norm).doit()
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VecMul), (VecMul, VectorSymbol)]
        
        expr = VecDot(self.v1, self.v2.norm).doit(deep=False)
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, Normalize), (Normalize, VectorSymbol)]
        
        expr = VecDot(self.v1, VecAdd(self.vn2, self.vn1)).doit(deep=False)
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VecAdd), (VecAdd, VectorSymbol)]
        expr = VecDot(self.v1, VecAdd(self.vn2, self.vn1)).doit()
        assert tuple(type(a) for a in expr.args) in [(VectorSymbol, VectorAdd), (VectorAdd, VectorSymbol)]
        
        expr = VecDot(self.vn1, self.vn2 + self.vn1).doit()
        assert expr == (x + 2 * y + 3 * z + 14)

if __name__ == "__main__":
    u.main()