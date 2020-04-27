import unittest as u
from common import CommonTest, x, y, z
from sympy import Add, log
from sympy.vector import (
    CoordSys3D, Vector, VectorZero as VZero, 
    curl, gradient, divergence
)

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Grad, DotNablaOp
)

class test_DotNablaOp(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        # scalar field
        assert isinstance(DotNablaOp(v1, x), DotNablaOp)
        # vector fields
        assert isinstance(DotNablaOp(v1, v1), DotNablaOp)
        assert isinstance(DotNablaOp(vn1, v1), DotNablaOp)
        assert isinstance(DotNablaOp(v1, vn1), DotNablaOp)

        def func(*args):
            with self.assertRaises(TypeError) as context:
                DotNablaOp(*args)

        func(nabla, v1)
        func(v1, nabla)
        func(v1, v2, x)
        func(v1, v2, v1)
        
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert isinstance(DotNablaOp(v1, v2).doit(), DotNablaOp)

        # identity test
        a = C.x * C.i + C.y * C.j + C.z * C.k
        b = C.y * C.z * C.i + C.x * C.z * C.j + C.x * C.y * C.k
        assert self._check_args(
            (nabla ^ (a ^ b)).doit(),
            (divergence(b) * a) + DotNablaOp(b, a).doit() - (divergence(a) * b) - DotNablaOp(a, b).doit()
        )

        assert isinstance(DotNablaOp(VecAdd(a, b), C.x).doit(deep=False), DotNablaOp)
        assert isinstance(DotNablaOp(VecAdd(a, b), VecAdd(a, b)).doit(deep=False), DotNablaOp)

    
if __name__ == "__main__":
    u.main()