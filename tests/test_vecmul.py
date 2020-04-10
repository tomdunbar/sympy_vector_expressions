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
        # no args
        assert VecMul() == VectorOne()
        # 1 arg
        assert VecMul(self.one) == VectorOne()
        assert VecMul(1) == S.One
        assert VecMul(self.zero) == VectorZero()
        assert VecMul(0) == S.Zero
        assert VecMul(x) == x
        # one argument is 0 or VectorZero()
        assert VecMul(0, self.v1.mag) == S.Zero
        assert VecMul(0, self.v1) == VectorZero()
        assert VecMul(self.zero, self.v1.mag) == VectorZero()
        assert VecMul(self.zero, 0) == VectorZero()
        # one argument is number 1
        assert VecMul(1, 1) == S.One
        assert VecMul(1, self.one) == self.one
        assert VecMul(1, 2) == S(2)
        assert VecMul(1, x) == x
        # no VectorExpr in arguments -> return Add
        assert isinstance(VecMul(2, x), Mul)
        # VectorExpr in args -> return VecAdd
        assert isinstance(VecMul(self.one, x), VecMul)
        assert isinstance(VecMul(self.v1.mag, self.v2), VecMul)
        assert isinstance(VecMul(self.v1, self.v2.mag), VecMul)
        assert isinstance(VecMul(self.v1.mag, self.v2.mag), VecMul)
        assert isinstance(VecMul(2, self.v2.mag), VecMul)
        assert isinstance(VecMul(2, self.v2, self.v1.mag), VecMul)

    def test_flatten(self):
        # test sympy.strategies.flatten applied to VecAdd
        assert self._check_args(
            VecMul(3, VecMul(self.v1, 4)),
            VecMul(12, self.v1)
        )
        assert self._check_args(
            VecMul(1, VecMul(self.v1, 4)),
            VecMul(4, self.v1)
        )
        assert self._check_args(
            VecMul(VecMul(self.v1, 4), VecMul(3, x, self.v1.mag, VecMul(x, self.v2.mag))),
            VecMul(12, x**2, self.v1.mag, self.v2.mag, self.v1)
        )

    def test_mul_vector(self):
        def func(args, **kwargs):
            with self.assertRaises(TypeError) as context:
                VecMul(*args, **kwargs)
            self.assertTrue('Multiplication of vector quantities not supported' in str(context.exception))
        func([self.v1, self.v2])
        func([self.v1, self.one], evaluate=False)
        func([self.v1, self.one])
        func([self.v1, self.zero], evaluate=False)
        func([self.v1, self.zero])
        func([self.v1, self.zero, self.zero])
        func([self.v1, self.vn1])
        func([self.v1, self.vn1])
        func([self.v1, self.v1 + self.v2])
    
    def test_doit(self):
        assert isinstance(VecMul(self.v1.mag, self.v1.mag, evaluate=False).doit(), VecPow)
        assert isinstance(VecMul(3, VecDot(self.vn1, self.vn2)).doit(), Add)
        assert isinstance(VecMul(3, VecDot(self.vn1, self.vn2)).doit(deep=False), VecMul)
    
    def test_is_vector(self):
        # one argument
        assert not VecMul(1).is_Vector
        assert not VecMul(x).is_Vector
        assert VecMul(self.one).is_Vector
        assert VecMul(self.v1).is_Vector
        # two arguments
        assert VecMul(self.one, x).is_Vector
        assert VecMul(0, self.one).is_Vector
        assert not VecMul(0, x).is_Vector
        assert VecMul(self.v1, 2).is_Vector
        assert not VecMul(self.v1.mag, 2).is_Vector
        assert VecMul(self.v1.mag, self.v2).is_Vector
        assert VecMul(self.v1 + self.v2, self.v2.mag).is_Vector
        assert VecMul(self.v1 + self.v2, 3).is_Vector
        assert VecMul(self.v1 + self.v2, (self.v1 + self.v2).mag).is_Vector
        # with dot product and nested mul/dot
        assert not VecMul(3, VecDot(self.v1, self.v2)).is_Vector
        assert VecMul(3, VecMul(self.v1, VecDot(self.v1, self.v2))).is_Vector
        assert VecMul(3, self.v1 + self.v2, VecMul(VecDot(self.v1, self.v2), x)).is_Vector
        # with cross product and nested mul/cross
        assert VecMul(3, VecCross(self.v1, self.v2)).is_Vector
        assert VecMul(3, self.v1.mag, VecMul(x, VecCross(self.v1, self.v2))).is_Vector
        # with cross and dot products
        assert VecMul(3, VecDot(self.v1, self.v2), VecMul(x, VecCross(self.v1, self.v2))).is_Vector
        assert not VecMul(3, VecDot(self.v1, self.v2), VecDot(VecMul(x, VecCross(self.v1, self.v2)), self.v1)).is_Vector

    def test_is_commutative(self):
        # one argument
        assert VecMul(1).is_commutative
        assert VecMul(self.one).is_commutative
        assert VecMul(self.zero).is_commutative
        assert VecMul(0).is_commutative
        assert VecMul(x).is_commutative
        # two arguments
        assert VecMul(x, self.v1).is_commutative
        assert VecMul(x, self.v1.mag).is_commutative
        assert VecMul(x, self.v1 + self.v2).is_commutative
        assert not VecMul(x, self.v1 ^ self.v2).is_commutative
        assert VecMul(x, self.v1 & self.v2).is_commutative
        
        t = Symbol("t", commutative=False)
        assert not VecMul(t, self.v1).is_commutative
        assert not VecMul(t, self.v1 ^ self.v2).is_commutative

        assert not VecMul(x, self.v1 ^ self.v2, evaluate=False).is_commutative
        assert VecMul(x, self.v1 & self.v2, evaluate=False).is_commutative
        assert not VecMul(t, self.v1, evaluate=False).is_commutative
        assert not VecMul(t, self.v1 ^ self.v2, evaluate=False).is_commutative

if __name__ == "__main__":
    u.main()