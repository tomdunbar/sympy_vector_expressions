# import unittest as u
# from common import CommonTest
# from sympy import Add, Mul, Symbol, S
# from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 

# from vector_expr import (
#     VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
#     VecDot, VecCross, VecPow
# )

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_exprII import (
    VecAdd, VecMul, VecPow, VecDot, VecCross, VectorSymbol, 
    VectorZero, VectorOne, Nabla
)
from sympy import S, Add, Mul, Symbol, symbols, Pow
from pytest import raises
from common import get_vars, check_args


def test_identity():
    assert VecMul.identity == S.One

def test_creation():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()
    
    # no args
    assert VecMul().doit() == S.One
    # assert VecMul() == VectorOne()
    # 1 arg
    assert VecMul(one).doit() == VectorOne()
    assert VecMul(1).doit() == S.One
    assert VecMul(zero).doit() == VectorZero()
    assert VecMul(0).doit() == S.Zero
    assert VecMul(x).doit() == x
    # one argument is 0 or VectorZero()
    assert VecMul(0, v1.mag).doit() == S.Zero
    assert VecMul(0, v1).doit() == VectorZero()
    assert VecMul(zero, v1.mag).doit() == VectorZero()
    assert VecMul(zero, 0).doit() == VectorZero()
    # one argument is number 1
    assert VecMul(1, 1).doit() == S.One
    assert VecMul(1, one).doit() == one
    assert VecMul(1, 2).doit() == S(2)
    assert VecMul(1, x).doit() == x
    # no VectorExpr in arguments -> return Add
    assert isinstance(VecMul(2, x).doit(), Mul)
    # VectorExpr in args -> return VecAdd
    assert isinstance(VecMul(one, x).doit(), VecMul)
    assert isinstance(VecMul(v1.mag, v2).doit(), VecMul)
    assert isinstance(VecMul(v1, v2.mag).doit(), VecMul)
    # TODO: in an ideal world, I'd like the following to test to return
    # objects of type VecMul
    assert isinstance(VecMul(v1.mag, v2.mag).doit(), Mul)
    assert isinstance(VecMul(2, v2.mag).doit(), Mul)
    assert isinstance(VecMul(2, v2, v1.mag).doit(), VecMul)

def test_flatten():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()

    # test sympy.strategies.flatten applied to VecAdd
    assert check_args(
        VecMul(3, VecMul(v1, 4)).doit(),
        VecMul(12, v1).doit()
    )
    assert check_args(
        VecMul(1, VecMul(v1, 4)).doit(),
        VecMul(4, v1).doit()
    )
    assert check_args(
        VecMul(VecMul(v1, 4), VecMul(3, x, v1.mag, VecMul(x, v2.mag))).doit(),
        VecMul(12, x**2, v1.mag, v2.mag, v1).doit()
    )

def test_mul_vector():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()

    def func(args, **kwargs):
        raises(ValueError, lambda: VecMul(*args, **kwargs).doit())
    
    func([v1, v2])
    func([v1, one], evaluate=False)
    func([v1, one])
    func([v1, zero], evaluate=False)
    func([v1, zero])
    func([v1, zero, zero])
    func([v1, vn1])
    func([v1, vn1])
    func([v1, v1 + v2])

def test_doit():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()

    # TODO: in an ideal world, I'd like the following to test to return
    # object of type VecPow
    assert isinstance(VecMul(v1.mag, v1.mag).doit(), Pow)
    assert isinstance(VecMul(3, VecDot(vn1, vn2)).doit(), Add)
    # TODO: in an ideal world, I'd like the following to test to return
    # objects of type VecMul
    assert isinstance(VecMul(3, VecDot(vn1, vn2)).doit(deep=False), Mul)

def test_is_vector():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()

    # one argument
    assert not VecMul(1).doit().is_Vector
    assert not VecMul(x).doit().is_Vector
    assert VecMul(one).doit().is_Vector
    assert VecMul(v1).doit().is_Vector
    # two arguments
    assert VecMul(one, x).doit().is_Vector
    assert VecMul(0, one).doit().is_Vector
    assert not VecMul(0, x).doit().is_Vector
    assert VecMul(v1, 2).doit().is_Vector
    assert not VecMul(v1.mag, 2).doit().is_Vector
    assert VecMul(v1.mag, v2).doit().is_Vector
    assert VecMul(v1 + v2, v2.mag).doit().is_Vector
    assert VecMul(v1 + v2, 3).doit().is_Vector
    assert VecMul(v1 + v2, (v1 + v2).mag).doit().is_Vector
    # with dot product and nested mul/dot
    assert not VecMul(3, VecDot(v1, v2)).doit().is_Vector
    assert VecMul(3, VecMul(v1, VecDot(v1, v2))).doit().is_Vector
    assert VecMul(3, v1 + v2, VecMul(VecDot(v1, v2), x)).doit().is_Vector
    # with cross product and nested mul/cross
    assert VecMul(3, VecCross(v1, v2)).doit().is_Vector
    assert VecMul(3, v1.mag, VecMul(x, VecCross(v1, v2))).doit().is_Vector
    # with cross and dot products
    assert VecMul(3, VecDot(v1, v2), VecMul(x, VecCross(v1, v2))).doit().is_Vector
    assert not VecMul(3, VecDot(v1, v2), VecDot(VecMul(x, VecCross(v1, v2)), v1)).doit().is_Vector

# def test_is_commutative():
#     v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = get_vars()

#     # one argument
#     assert VecMul(1).is_commutative
#     assert VecMul(one).is_commutative
#     assert VecMul(zero).is_commutative
#     assert VecMul(0).is_commutative
#     assert VecMul(x).is_commutative
#     # two arguments
#     assert VecMul(x, v1).is_commutative
#     assert VecMul(x, v1.mag).is_commutative
#     assert VecMul(x, v1 + v2).is_commutative
#     assert not VecMul(x, v1 ^ v2).is_commutative
#     assert VecMul(x, v1 & v2).is_commutative
    
#     t = Symbol("t", commutative=False)
#     assert not VecMul(t, v1).is_commutative
#     assert not VecMul(t, v1 ^ v2).is_commutative

#     assert not VecMul(x, v1 ^ v2, evaluate=False).is_commutative
#     assert VecMul(x, v1 & v2, evaluate=False).is_commutative
#     assert not VecMul(t, v1, evaluate=False).is_commutative
#     assert not VecMul(t, v1 ^ v2, evaluate=False).is_commutative
