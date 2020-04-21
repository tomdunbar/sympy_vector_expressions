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
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecCross(v1, zero) == zero
        assert VecCross(zero, v1) == zero
        assert VecCross(v1, v1) == zero
        assert VecCross(vn1, vn1) == Vector.zero
        assert VecCross(v1, v1.norm)
        assert VecCross(vn1, v1.norm)
        assert VecCross(nabla, v1)
        assert VecCross(v1, nabla)
        assert VecCross(v1, vn1)
        assert VecCross(vn1, v1)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                VecCross(*args)
        
        func(x, v1)
        func(v1, x)
        func(v1, v1.mag)
        func(v1 & v2, v1.mag)
        func(v1 & v2, v1)
        func(0, v1)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecCross(v1, zero).is_Vector
        assert VecCross(v1, v2).is_Vector
        assert VecCross(vn1, vn2).is_Vector
        assert VecCross(v1, vn1).is_Vector
        assert VecCross((v1 & v2) * one, v2).is_Vector
    
    def test_reverse(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecCross(v1, v2).reverse == -VecCross(v2, v1)
        assert VecCross(vn1, v2).reverse == -VecCross(v2, vn1)
        assert VecCross(v1, vn2).reverse == -VecCross(vn2, v1)
        assert VecCross(vn1, vn2).reverse == -VecCross(vn2, vn1)
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert VecCross(vn1, vn2).doit() - vn1.cross(vn2) == Vector.zero
        assert VecCross(vn2, nabla).doit() - curl(vn2) == Vector.zero
        assert VecCross(nabla, vn2).doit() - curl(vn2) == Vector.zero

        vn3 = C.x * C.y * C.z * C.i + C.x * C.y * C.z * C.j + C.x * C.y * C.z * C.k
        assert VecCross(nabla, vn3).doit() - curl(vn3) == Vector.zero
        assert VecCross(vn3, nabla).doit() + curl(vn3) == Vector.zero
        
        expr = VecCross(v2 * VecDot(vn1, vn2), v1).doit()
        assert isinstance(expr.args[0].args[0], Add)
        expr = VecCross(v2 * VecDot(vn1, vn2), v1).doit(deep=False)
        assert isinstance(expr.args[0], VecMul)

if __name__ == "__main__":
    u.main()