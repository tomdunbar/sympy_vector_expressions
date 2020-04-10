import unittest as u
from common import CommonTest, x, y, z
from sympy import Abs, Pow, sqrt, Number, S
from sympy.vector import CoordSys3D, Vector

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    Magnitude, VecCross
)

class test_Magnitude(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        assert isinstance(Magnitude(self.v1), Magnitude)
        assert isinstance(Magnitude(self.v1.mag), Magnitude)
        assert isinstance(Magnitude(self.v1 + self.v2), Magnitude)
        assert isinstance(Magnitude(-2), Number)
        assert isinstance(Magnitude(x), Abs)
        assert isinstance(Magnitude(self.v1 & self.v2), Abs)
        assert isinstance(Magnitude(self.vn1), Magnitude)
        assert isinstance(Magnitude(self.vn1.dot(self.vn2)), Abs)

    def test_vectorsymbol_magnitudes(self):
        assert self.zero.mag == S.Zero

        with self.assertRaises(TypeError) as context:
            self.nabla.mag
        self.assertTrue("nabla operator doesn't have magnitude." in str(context.exception))

        assert isinstance(self.v1.mag, Magnitude)
        assert isinstance(self.one.mag, Magnitude)
    
    def test_is_vector(self):
        assert not self.v1.mag.is_Vector
        assert not self.one.mag.is_Vector
        assert not self.zero.mag.is_Vector
        assert not Magnitude(self.v1 + self.v2).is_Vector
        assert not Magnitude(self.v1 ^ self.v2).is_Vector
    

    def test_doit(self):
        assert Magnitude(VecAdd(self.v1, self.zero, evaluate=False)).doit() == self.v1.mag
        assert Magnitude(self.vn1).doit() == sqrt(14)
        assert Magnitude(self.vn2).doit() == sqrt(x**2 + y**2 + z**2)
        assert isinstance(Magnitude(VecCross(self.vn1, self.vn2)).doit(deep=False), Magnitude)
        assert isinstance(Magnitude(VecCross(self.vn1, self.vn2)).doit(), Pow)

if __name__ == "__main__":
    u.main()