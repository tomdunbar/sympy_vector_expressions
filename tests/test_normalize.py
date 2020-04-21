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
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert isinstance(Normalize(v1), Normalize)
        assert isinstance(Normalize(one), Normalize)
        assert Normalize(zero) == zero
        assert isinstance(Normalize(x * v1), Normalize)

        def func(arg):
            with self.assertRaises(TypeError) as context:
                Normalize(arg)

        assert isinstance(Normalize(v1 + v2), Normalize)
        assert isinstance(Normalize(vn1), Normalize)

        args = [nabla, 1, v1.mag, VecDot(v1, v2)]
        for a in args:
            func(a)

        assert isinstance(Normalize(v1.norm), Normalize)
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        # return a VecMul object (a fraction, vector/magnitude)
        assert isinstance(Normalize(v1).doit(), VecMul)
        assert isinstance(Normalize(one).doit(), VecMul)
        assert isinstance(Normalize(zero).doit(), VectorZero)
        # symbolic Vector
        assert Normalize(vn1).doit() - vn1.normalize() == Vector.zero
        assert Normalize(vn2).doit() - vn2.normalize() == Vector.zero
        assert Normalize(VecAdd(vn1, vn2)).doit() - (vn1 + vn2).normalize() == Vector.zero
        assert isinstance(Normalize(VecAdd(vn1, vn2)).doit(deep=False), VecMul)
    
        # test to check if the result is a vector
        assert Normalize(v1).doit().is_Vector
        assert Normalize(v1 + v2).doit().is_Vector
        assert Normalize(one).doit().is_Vector
        assert Normalize(zero).doit().is_Vector
        assert Normalize(vn1).doit().is_Vector
        assert Normalize(VecAdd(vn1, vn2)).doit().is_Vector
        assert Normalize(VecAdd(vn1, vn2)).doit(deep=False).is_Vector

if __name__ == "__main__":
    u.main()