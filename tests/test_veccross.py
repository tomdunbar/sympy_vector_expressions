import unittest as u
from common import CommonTest, x, y, z
from sympy import Add
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero, curl

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross
)

class test_VecCross(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        assert VecCross(self.v1, self.zero) == self.zero
        assert VecCross(self.zero, self.v1) == self.zero
        assert VecCross(self.v1, self.v1) == self.zero
        assert VecCross(self.vn1, self.vn1) == Vector.zero
        assert VecCross(self.v1, self.v1.norm)
        assert VecCross(self.vn1, self.v1.norm)
        assert VecCross(self.nabla, self.v1)
        assert VecCross(self.v1, self.nabla)
        assert VecCross(self.v1, self.vn1)
        assert VecCross(self.vn1, self.v1)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecCross(*args)
        
        func(x, self.v1)
        func(self.v1, x)
        func(self.v1, self.v1.mag)
        func(self.v1 & self.v2, self.v1.mag)
        func(self.v1 & self.v2, self.v1)
        func(0, self.v1)
    
    def test_is_vector(self):
        assert VecCross(self.v1, self.zero).is_Vector
        assert VecCross(self.v1, self.v2).is_Vector
        assert VecCross(self.vn1, self.vn2).is_Vector
        assert VecCross(self.v1, self.vn1).is_Vector
        assert VecCross((self.v1 & self.v2) * self.one, self.v2).is_Vector
    
    def test_reverse(self):
        assert VecCross(self.v1, self.v2).reverse == -VecCross(self.v2, self.v1)
        assert VecCross(self.vn1, self.v2).reverse == -VecCross(self.v2, self.vn1)
        assert VecCross(self.v1, self.vn2).reverse == -VecCross(self.vn2, self.v1)
        assert VecCross(self.vn1, self.vn2).reverse == -VecCross(self.vn2, self.vn1)
    
    def test_doit(self):
        assert VecCross(self.vn1, self.vn2).doit() - self.vn1.cross(self.vn2) == Vector.zero
        assert VecCross(self.vn2, self.nabla).doit() - curl(self.vn2) == Vector.zero
        assert VecCross(self.nabla, self.vn2).doit() - curl(self.vn2) == Vector.zero

        C = self.C
        vn3 = C.x * C.y * C.z * C.i + C.x * C.y * C.z * C.j + C.x * C.y * C.z * C.k
        assert VecCross(self.nabla, vn3).doit() - curl(vn3) == Vector.zero
        assert VecCross(vn3, self.nabla).doit() + curl(vn3) == Vector.zero
        
        expr = VecCross(self.v2 * VecDot(self.vn1, self.vn2), self.v1).doit()
        assert isinstance(expr.args[0].args[0], Add)
        expr = VecCross(self.v2 * VecDot(self.vn1, self.vn2), self.v1).doit(deep=False)
        assert isinstance(expr.args[0], VecMul)

if __name__ == "__main__":
    u.main()