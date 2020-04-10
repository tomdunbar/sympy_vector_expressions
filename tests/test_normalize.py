import unittest as u
from common import CommonTest, x, y, z
from sympy.vector import CoordSys3D, Vector

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    Magnitude, Normalize, VecDot
)

class test_Normalize(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        assert isinstance(Normalize(self.v1), Normalize)
        assert isinstance(Normalize(self.one), Normalize)
        assert Normalize(self.zero) == self.zero

        def func(arg):
            with self.assertRaises(TypeError) as context:
                Normalize(arg)

        assert isinstance(Normalize(self.v1 + self.v2), Normalize)
        assert isinstance(Normalize(self.vn1), Normalize)

        args = [self.nabla, 1, self.v1.mag, x * self.v1, VecDot(self.v1, self.v2)]
        for a in args:
            func(a)

        assert isinstance(Normalize(self.v1.norm), Normalize)
    
    def test_doit(self):
        # return a VecMul object (a fraction, vector/magnitude)
        assert isinstance(Normalize(self.v1).doit(), VecMul)
        assert isinstance(Normalize(self.one).doit(), VecMul)
        assert isinstance(Normalize(self.zero).doit(), VectorZero)
        # symbolic Vector
        assert Normalize(self.vn1).doit() - self.vn1.normalize() == Vector.zero
        assert Normalize(self.vn2).doit() - self.vn2.normalize() == Vector.zero
        assert Normalize(VecAdd(self.vn1, self.vn2)).doit() - (self.vn1 + self.vn2).normalize() == Vector.zero
        assert isinstance(Normalize(VecAdd(self.vn1, self.vn2)).doit(deep=False), VecMul)
    
        # test to check if the result is a vector
        assert Normalize(self.v1).doit().is_Vector
        assert Normalize(self.v1 + self.v2).doit().is_Vector
        assert Normalize(self.one).doit().is_Vector
        assert Normalize(self.zero).doit().is_Vector
        assert Normalize(self.vn1).doit().is_Vector
        assert Normalize(VecAdd(self.vn1, self.vn2)).doit().is_Vector
        assert Normalize(VecAdd(self.vn1, self.vn2)).doit(deep=False).is_Vector

if __name__ == "__main__":
    u.main()