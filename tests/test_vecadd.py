import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_exprII import (
    VecAdd, VecMul, VecDot, VecCross, VectorSymbol, 
    VectorZero, VectorOne, Nabla
)
from sympy import S, Add, symbols
from pytest import raises
from common import get_vars, check_args

def test_identity():
    assert VecAdd.identity == S.Zero

def test_creation():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = _get_vars()

    # no args
    assert VecAdd() == VectorZero()
    # 1 arg
    assert VecAdd(v1.mag).doit() == v1.mag
    assert VecAdd(v2).doit() == v2
    assert VecAdd(zero).doit() == zero
    assert VecAdd(0).doit() == S.Zero
    # one argument is 0 or VectorZero()
    r = VecAdd(one, zero).doit()
    assert r == one and r.is_Vector
    r = VecAdd(VectorOne(), VectorZero()).doit()
    assert r == one and r.is_Vector
    r = VecAdd(v1, VectorOne(), VectorZero()).doit()
    assert isinstance(r, VecAdd) and r.is_Vector
    assert VecAdd(0, x).doit() == x
    assert VecAdd(zero, v1).doit() == v1
    assert VecAdd(v1.mag, 0).doit() == v1.mag
    # all scalars-> return VecAdd
    assert isinstance(VecAdd(2, x, y).doit(), VecAdd)
    # VectorExpr in args -> return VecAdd
    assert isinstance(VecAdd(v1, v2).doit(), VecAdd)
    assert isinstance(VecAdd(vn1, vn2), VecAdd)
    assert isinstance(VecAdd(vn1, vn2).doit(), VectorAdd)
    assert isinstance(VecAdd(v1.mag, v2.mag).doit(), VecAdd)
    assert isinstance(VecAdd(1, v2.mag).doit(), VecAdd)
    assert isinstance(VecAdd(one, v2 ^ v1).doit(), VecAdd)
    assert isinstance(VecAdd(1, v2 & v1).doit(), VecAdd)
    assert isinstance(VecAdd(one, v1, v2).doit(), VecAdd)
    assert isinstance(VecAdd(v1, v1).doit(), VecMul)
    assert isinstance(VecAdd(v1, v1), VecAdd)

def test_mix_scalar_vector():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = _get_vars()

    def func(args, **kwargs):
        raises(ValueError, lambda: VecAdd(*args, **kwargs).doit())

    func([v1, 1])
    func([v1.mag, one])
    func([vn1, 1])
    func([vn1.magnitude(), one])
    func([v1 & v2, one])
    func([v1 & v2, one], evaluate=False)
    func([v1 ^ v2, 1])
    func([v1 ^ v2, 1], evaluate=False)

def test_flatten():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = _get_vars()

    # test sympy.strategies.flatten applied to VecAdd
    assert _check_args(
        VecAdd(v1, VecAdd(v2, one)).doit(),
        VecAdd(one, v1, v2).doit()
    )
    assert _check_args(
        VecAdd(v1, VecAdd(one), VecAdd(zero, v2)).doit(),
        VecAdd(one, v1, v2).doit()
    )
    assert _check_args(
        VecAdd(v1, VecAdd(one, VecAdd(zero, v2))).doit(),
        VecAdd(one, v1, v2).doit()
    )
    assert _check_args(
        VecAdd(v1 ^ v2, VecAdd(one, VecAdd(2 * v1, (v2 & v1) * one))).doit(),
        VecAdd(v1 ^ v2, one, 2 * v1, (v2 & v1) * one).doit()
    )

def test_doit():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = _get_vars()

    expr = VecAdd(v1, v2, v1, evaluate=False)
    assert not _check_args(
        expr,
        VecAdd(VecMul(2, v1), v2)
    )
    assert _check_args(
        expr.doit(),
        VecAdd(VecMul(2, v1), v2)
    )

    expr = VecAdd(v1, v2, VecAdd(v1, v2, evaluate=False), evaluate=False)
    assert  not _check_args(
        expr,
        VecAdd(VecMul(2, v1), VecMul(2, v2))
    )
    assert _check_args(
        expr.doit(), 
        VecAdd(VecMul(2, v1), VecMul(2, v2))
    )

    r = vn1 + vn2
    assert VecAdd(vn1, vn2).doit() == r

    expr = VecAdd(v1.mag, VecDot(vn1, vn2))
    assert _check_args(
        expr.doit(deep=False),
        expr
    )
    assert _check_args(
        expr.doit(),
        VecAdd(v1.mag, vn1 & vn2)
    )

    # test the rule default_sort_key used into VecAdd.doit
    assert VecAdd(v1, one, v2).doit().args == (one, v1, v2)
    assert VecAdd(v1, VecCross(v1, v2), one).doit().args == (VecCross(v1, v2), one, v1)
    # test the rule merge_explicit used into VecAdd.doit
    assert VecAdd(vn1, vn2).doit() == vn1 + vn2
    assert _check_args(
        VecAdd(vn1, vn2, one).doit(),
        VecAdd(one, vn1 + vn2)
    )

def test_is_vector():
    v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = _get_vars()
    
    assert VecAdd(v1, v2).is_Vector
    # assert not VecAdd(v1, v2).is_Vector_Scalar
    assert not VecAdd(v1.mag, v2.mag).is_Vector
    # assert VecAdd(v1.mag, v2.mag).is_Vector_Scalar
    # with dot product and nested mul/dot
    assert not VecAdd(2, v2 & v1).is_Vector
    # assert VecAdd(2, v2 & v1).is_Vector_Scalar
    assert not VecAdd(2, VecMul(x, v2 & v1)).is_Vector
    # assert VecAdd(2,  VecMul(x, v2 & v1)).is_Vector_Scalar
    # with cross product and nested mul/cross
    assert VecAdd(v1, v2 ^ v1).is_Vector
    # assert not VecAdd(v1, v2 ^ v1).is_Vector_Scalar
    assert VecAdd(v1, VecMul(x, v2 ^ v1)).is_Vector
    # assert not VecAdd(v1, VecMul(x, v2 ^ v1)).is_Vector_Scalar
