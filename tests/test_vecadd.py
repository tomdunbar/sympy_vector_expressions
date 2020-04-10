import unittest as u
from common import CommonTest, x, y, z
from sympy import Add, Mul, S
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecDot, VecCross, VectorSymbol, VectorZero, VectorOne, Nabla
)

# TODO: should Nabla be allowed as an argument of VecAdd?

class test_VecAdd(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)

    def test_identity(self):
        assert VecAdd.identity == S.Zero

    def test_creation(self):
        # no args
        assert VecAdd() == VectorZero()
        # 1 arg
        assert VecAdd(self.v1.mag) == self.v1.mag
        assert VecAdd(self.v2) == self.v2
        assert VecAdd(self.zero) == self.zero
        assert VecAdd(0) == S.Zero
        # one argument is 0 or VectorZero()
        assert VecAdd(self.one, self.zero) == self.one
        assert VecAdd(0, x) == x
        assert VecAdd(self.zero, self.v1) == self.v1
        assert VecAdd(self.v1.mag, 0) == self.v1.mag
        # no VectorExpr in arguments -> return Add
        assert isinstance(VecAdd(2, x, y), Add)
        # VectorExpr in args -> return VecAdd
        assert isinstance(VecAdd(self.v1, self.v2), VecAdd)
        assert isinstance(VecAdd(self.vn1, self.vn2), VecAdd)
        assert isinstance(VecAdd(self.v1.mag, self.v2.mag), VecAdd)
        assert isinstance(VecAdd(1, self.v2.mag), VecAdd)
        assert isinstance(VecAdd(self.one, self.v2 ^ self.v1), VecAdd)
        assert isinstance(VecAdd(1, self.v2 & self.v1), VecAdd)
        assert isinstance(VecAdd(self.one, self.v1, self.v2), VecAdd)

    def test_mix_scalar_vector(self):
        def func(args, **kwargs):
            with self.assertRaises(TypeError) as context:
                VecAdd(*args, **kwargs)
            self.assertTrue('Mix of Vector and Scalar symbols' in str(context.exception))

        func([self.v1, 1])
        func([self.v1.mag, self.one])
        func([self.vn1, 1])
        func([self.vn1.magnitude(), self.one])
        func([self.v1, 0], evaluate=False)
        func([self.v1.mag, self.zero], evaluate=False)
        func([self.v1 & self.v2, self.one])
        func([self.v1 & self.v2, self.one], evaluate=False)
        func([self.v1 ^ self.v2, 1])
        func([self.v1 ^ self.v2, 1], evaluate=False)

    def test_flatten(self):
        # test sympy.strategies.flatten applied to VecAdd
        assert self._check_args(
            VecAdd(self.v1, VecAdd(self.v2, self.one)),
            VecAdd(self.one, self.v1, self.v2)
        )
        assert self._check_args(
            VecAdd(self.v1, VecAdd(self.one), VecAdd(self.zero, self.v2)),
            VecAdd(self.one, self.v1, self.v2)
        )
        assert self._check_args(
            VecAdd(self.v1, VecAdd(self.one, VecAdd(self.zero, self.v2))),
            VecAdd(self.one, self.v1, self.v2)
        )
        assert self._check_args(
            VecAdd(self.v1 ^ self.v2, VecAdd(self.one, VecAdd(2 * self.v1, (self.v2 & self.v1) * self.one))),
            VecAdd(self.v1 ^ self.v2, self.one, 2 * self.v1, (self.v2 & self.v1) * self.one)
        )

    def test_doit(self):
        expr = VecAdd(self.v1, self.v2, self.v1, evaluate=False)
        assert not self._check_args(
            expr,
            VecAdd(VecMul(2, self.v1), self.v2)
        )
        assert self._check_args(
            expr.doit(),
            VecAdd(VecMul(2, self.v1), self.v2)
        )

        expr = VecAdd(self.v1, self.v2, VecAdd(self.v1, self.v2, evaluate=False), evaluate=False)
        assert  not self._check_args(
            expr,
            VecAdd(VecMul(2, self.v1), VecMul(2, self.v2))
        )
        assert self._check_args(
            expr.doit(), 
            VecAdd(VecMul(2, self.v1), VecMul(2, self.v2))
        )

        r = self.vn1 + self.vn2
        assert VecAdd(self.vn1, self.vn2).doit() == r

        expr = VecAdd(self.v1.mag, VecDot(self.vn1, self.vn2))
        assert self._check_args(
            expr.doit(deep=False),
            expr
        )
        assert self._check_args(
            expr.doit(),
            VecAdd(self.v1.mag, self.vn1 & self.vn2)
        )

        # test the rule default_sort_key used into VecAdd.doit
        assert VecAdd(self.v1, self.one, self.v2).doit().args == (self.one, self.v1, self.v2)
        assert VecAdd(self.v1, VecCross(self.v1, self.v2), self.one).doit().args == (VecCross(self.v1, self.v2), self.one, self.v1)
        # test the rule merge_explicit used into VecAdd.doit
        assert VecAdd(self.vn1, self.vn2).doit() == self.vn1 + self.vn2
        assert self._check_args(
            VecAdd(self.vn1, self.vn2, self.one).doit(),
            VecAdd(self.one, self.vn1 + self.vn2)
        )

    def test_is_vector(self):
        assert VecAdd(self.v1, self.v2).is_Vector
        assert not VecAdd(self.v1, self.v2).is_scalar
        assert not VecAdd(self.v1.mag, self.v2.mag).is_Vector
        assert VecAdd(self.v1.mag, self.v2.mag).is_scalar
        # with dot product and nested mul/dot
        assert not VecAdd(2, self.v2 & self.v1).is_Vector
        assert VecAdd(2, self.v2 & self.v1).is_scalar
        assert not VecAdd(2, VecMul(x, self.v2 & self.v1)).is_Vector
        assert VecAdd(2,  VecMul(x, self.v2 & self.v1)).is_scalar
        # with cross product and nested mul/cross
        assert VecAdd(self.v1, self.v2 ^ self.v1).is_Vector
        assert not VecAdd(self.v1, self.v2 ^ self.v1).is_scalar
        assert VecAdd(self.v1, VecMul(x, self.v2 ^ self.v1)).is_Vector
        assert not VecAdd(self.v1,  VecMul(x, self.v2 ^ self.v1)).is_scalar

if __name__ == "__main__":
    u.main()