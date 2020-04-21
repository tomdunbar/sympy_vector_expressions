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
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert isinstance(Magnitude(v1), Magnitude)
        assert isinstance(Magnitude(v1.mag), Magnitude)
        assert isinstance(Magnitude(v1 + v2), Magnitude)
        assert isinstance(Magnitude(-2), Number)
        assert isinstance(Magnitude(x), Abs)
        assert isinstance(Magnitude(v1 & v2), Abs)
        assert isinstance(Magnitude(vn1), Magnitude)
        assert isinstance(Magnitude(vn1.dot(vn2)), Abs)

    def test_vectorsymbol_magnitudes(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert zero.mag == S.Zero

        with self.assertRaises(TypeError) as context:
            nabla.mag
        self.assertTrue("nabla operator doesn't have magnitude." in str(context.exception))

        assert isinstance(v1.mag, Magnitude)
        assert isinstance(one.mag, Magnitude)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert not v1.mag.is_Vector
        assert not one.mag.is_Vector
        assert not zero.mag.is_Vector
        assert not Magnitude(v1 + v2).is_Vector
        assert not Magnitude(v1 ^ v2).is_Vector
    

    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert Magnitude(VecAdd(v1, zero, evaluate=False)).doit() == v1.mag
        assert Magnitude(vn1).doit() == sqrt(14)
        assert Magnitude(vn2).doit() == sqrt(x**2 + y**2 + z**2)
        assert isinstance(Magnitude(VecCross(vn1, vn2)).doit(deep=False), Magnitude)
        assert isinstance(Magnitude(VecCross(vn1, vn2)).doit(), Pow)

if __name__ == "__main__":
    u.main()