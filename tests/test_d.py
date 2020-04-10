import unittest as u
from common import CommonTest, x, y, z
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

class test_VecDot(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        assert isinstance(D(2), Number)
        assert isinstance(D(self.v1), VectorSymbol)
        assert isinstance(D(self.v1 + self.v2), VecAdd)
        assert isinstance(D(self.v1.mag * self.v2), VecMul)
        assert isinstance(D(self.v1 & self.v2), VecDot)
        assert isinstance(D(self.v1 ^ self.v2), VecCross)
        
        assert isinstance(D(self.v1.diff(x)), D)
        assert isinstance((self.v1 + self.v2).diff(x, evaluate=False), D)
        assert isinstance((self.v1.mag * self.v2).diff(x, evaluate=False), D)
        assert isinstance((self.v1 & self.v2).diff(x, evaluate=False), D)
        assert isinstance((self.v1 ^ self.v2).diff(x, evaluate=False), D)
    
    def test_is_vector(self):
        assert (self.v1 + self.v2).diff(x, evaluate=False).is_Vector
        assert not (self.v1 + self.v2).diff(x, evaluate=False).is_scalar
        assert (self.v1.mag * self.v2).diff(x, evaluate=False).is_Vector
        assert not (self.v1.mag * self.v2).diff(x, evaluate=False).is_scalar
        assert not (self.v1 & self.v2).diff(x, evaluate=False).is_Vector
        assert (self.v1 & self.v2).diff(x, evaluate=False).is_scalar
        assert (self.v1 ^ self.v2).diff(x, evaluate=False).is_Vector
        assert not (self.v1 ^ self.v2).diff(x, evaluate=False).is_scalar
    

if __name__ == "__main__":
    u.main()