import unittest as u
from common import CommonTest
from sympy import S, Integer, Number
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude, D
)

class test_Dclass(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert isinstance(D(2), Number)
        assert isinstance(D(v1), VectorSymbol)
        assert isinstance(D(v1 + v2), VecAdd)
        assert isinstance(D(v1.mag * v2), VecMul)
        assert isinstance(D(v1 & v2), VecDot)
        assert isinstance(D(v1 ^ v2), VecCross)
        
        assert isinstance(D(v1.diff(x)), D)
        assert isinstance((v1 + v2).diff(x, evaluate=False), D)
        assert isinstance((v1.mag * v2).diff(x, evaluate=False), D)
        assert isinstance((v1 & v2).diff(x, evaluate=False), D)
        assert isinstance((v1 ^ v2).diff(x, evaluate=False), D)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert (v1 + v2).diff(x, evaluate=False).is_Vector
        assert not (v1 + v2).diff(x, evaluate=False).is_scalar
        assert (v1.mag * v2).diff(x, evaluate=False).is_Vector
        assert not (v1.mag * v2).diff(x, evaluate=False).is_scalar
        assert not (v1 & v2).diff(x, evaluate=False).is_Vector
        assert (v1 & v2).diff(x, evaluate=False).is_scalar
        assert (v1 ^ v2).diff(x, evaluate=False).is_Vector
        assert not (v1 ^ v2).diff(x, evaluate=False).is_scalar
    

if __name__ == "__main__":
    u.main()