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
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecDot(v1, zero) == S.Zero
        assert VecDot(zero, v1) == S.Zero
        assert isinstance(VecDot(v1, v1), VecDot)
        assert isinstance(VecDot(v1, v2), VecDot)
        assert isinstance(VecDot(v1, v1.norm), VecDot)
        assert isinstance(VecDot(v1.norm, v1), VecDot)
        assert isinstance(VecDot(nabla, v1), VecDot)
        assert isinstance(VecDot(v1, nabla), VecDot)
        assert isinstance(VecDot(v1, vn1), VecDot)
        assert isinstance(VecDot(vn1, v1), VecDot)
        assert isinstance(VecDot(v1 + v2, v1), VecDot)
        assert isinstance(VecDot(v1.mag * v2, v1), VecDot)
        assert isinstance(VecDot(v1 ^ v2, v1), VecDot)
        assert isinstance(VecDot((v1 ^ v2) * v1.mag, v1), VecDot)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecDot(*args)
        
        func(x, v1)
        func(v1, x)
        func(v1, v1.mag)
        func(zero, 0)
        func(nabla, nabla)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert not VecDot(v1, zero).is_Vector
        assert VecDot(v1, zero).is_scalar
        assert not VecDot(v1, v2).is_Vector
        assert VecDot(v1, v2).is_Vector_Scalar
        assert not VecDot(vn1, vn2).is_Vector
        assert VecDot(vn1, vn2).is_Vector_Scalar
        assert not VecDot(v1, vn1).is_Vector
        assert VecDot(v1, vn1).is_Vector_Scalar
        assert not VecDot(v1 ^ v2, v1).is_Vector
        assert VecDot(v1 ^ v2, v1).is_Vector_Scalar
    
    def test_reverse(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecDot(v1, v2).reverse == VecDot(v2, v1)
        assert VecDot(vn1, v2).reverse == VecDot(v2, vn1)
        assert VecDot(v1, vn2).reverse == VecDot(vn2, v1)
        assert VecDot(vn1, vn2).reverse == VecDot(vn2, vn1)
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert VecDot(vn1, vn2).doit() - vn1.dot(vn2) == 0
        assert VecDot(vn2, nabla).doit() - divergence(vn2) == 0
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

if __name__ == "__main__":
    u.main()