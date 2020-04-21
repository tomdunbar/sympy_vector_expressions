import unittest as u
from common import CommonTest, x, y, z
from sympy import Add, Mul, Symbol, S
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, VecPow
)

class test_VecMul(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_identity(self):
        assert VecMul.identity == S.One

    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        # no args
        assert VecMul() == S.One
        # assert VecMul() == VectorOne()
        # 1 arg
        assert VecMul(one) == VectorOne()
        assert VecMul(1) == S.One
        assert VecMul(zero) == VectorZero()
        assert VecMul(0) == S.Zero
        assert VecMul(x) == x
        # one argument is 0 or VectorZero()
        assert VecMul(0, v1.mag) == S.Zero
        assert VecMul(0, v1) == VectorZero()
        assert VecMul(zero, v1.mag) == VectorZero()
        assert VecMul(zero, 0) == VectorZero()
        # one argument is number 1
        assert VecMul(1, 1) == S.One
        assert VecMul(1, one) == one
        assert VecMul(1, 2) == S(2)
        assert VecMul(1, x) == x
        # no VectorExpr in arguments -> return Add
        assert isinstance(VecMul(2, x), Mul)
        # VectorExpr in args -> return VecAdd
        assert isinstance(VecMul(one, x), VecMul)
        assert isinstance(VecMul(v1.mag, v2), VecMul)
        assert isinstance(VecMul(v1, v2.mag), VecMul)
        assert isinstance(VecMul(v1.mag, v2.mag), VecMul)
        assert isinstance(VecMul(2, v2.mag), VecMul)
        assert isinstance(VecMul(2, v2, v1.mag), VecMul)

    def test_flatten(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        # test sympy.strategies.flatten applied to VecAdd
        assert self._check_args(
            VecMul(3, VecMul(v1, 4)),
            VecMul(12, v1)
        )
        assert self._check_args(
            VecMul(1, VecMul(v1, 4)),
            VecMul(4, v1)
        )
        assert self._check_args(
            VecMul(VecMul(v1, 4), VecMul(3, x, v1.mag, VecMul(x, v2.mag))),
            VecMul(12, x**2, v1.mag, v2.mag, v1)
        )

    def test_mul_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        def func(args, **kwargs):
            with self.assertRaises(TypeError) as context:
                VecMul(*args, **kwargs)
            self.assertTrue('Multiplication of vector quantities not supported' in str(context.exception))
        func([v1, v2])
        func([v1, one], evaluate=False)
        func([v1, one])
        func([v1, zero], evaluate=False)
        func([v1, zero])
        func([v1, zero, zero])
        func([v1, vn1])
        func([v1, vn1])
        func([v1, v1 + v2])
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        assert isinstance(VecMul(v1.mag, v1.mag, evaluate=False).doit(), VecPow)
        assert isinstance(VecMul(3, VecDot(vn1, vn2)).doit(), Add)
        assert isinstance(VecMul(3, VecDot(vn1, vn2)).doit(deep=False), VecMul)
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        # one argument
        assert not VecMul(1).is_Vector
        assert not VecMul(x).is_Vector
        assert VecMul(one).is_Vector
        assert VecMul(v1).is_Vector
        # two arguments
        assert VecMul(one, x).is_Vector
        assert VecMul(0, one).is_Vector
        assert not VecMul(0, x).is_Vector
        assert VecMul(v1, 2).is_Vector
        assert not VecMul(v1.mag, 2).is_Vector
        assert VecMul(v1.mag, v2).is_Vector
        assert VecMul(v1 + v2, v2.mag).is_Vector
        assert VecMul(v1 + v2, 3).is_Vector
        assert VecMul(v1 + v2, (v1 + v2).mag).is_Vector
        # with dot product and nested mul/dot
        assert not VecMul(3, VecDot(v1, v2)).is_Vector
        assert VecMul(3, VecMul(v1, VecDot(v1, v2))).is_Vector
        assert VecMul(3, v1 + v2, VecMul(VecDot(v1, v2), x)).is_Vector
        # with cross product and nested mul/cross
        assert VecMul(3, VecCross(v1, v2)).is_Vector
        assert VecMul(3, v1.mag, VecMul(x, VecCross(v1, v2))).is_Vector
        # with cross and dot products
        assert VecMul(3, VecDot(v1, v2), VecMul(x, VecCross(v1, v2))).is_Vector
        assert not VecMul(3, VecDot(v1, v2), VecDot(VecMul(x, VecCross(v1, v2)), v1)).is_Vector

    def test_is_commutative(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()

        # one argument
        assert VecMul(1).is_commutative
        assert VecMul(one).is_commutative
        assert VecMul(zero).is_commutative
        assert VecMul(0).is_commutative
        assert VecMul(x).is_commutative
        # two arguments
        assert VecMul(x, v1).is_commutative
        assert VecMul(x, v1.mag).is_commutative
        assert VecMul(x, v1 + v2).is_commutative
        assert not VecMul(x, v1 ^ v2).is_commutative
        assert VecMul(x, v1 & v2).is_commutative
        
        t = Symbol("t", commutative=False)
        assert not VecMul(t, v1).is_commutative
        assert not VecMul(t, v1 ^ v2).is_commutative

        assert not VecMul(x, v1 ^ v2, evaluate=False).is_commutative
        assert VecMul(x, v1 & v2, evaluate=False).is_commutative
        assert not VecMul(t, v1, evaluate=False).is_commutative
        assert not VecMul(t, v1 ^ v2, evaluate=False).is_commutative

if __name__ == "__main__":
    u.main()