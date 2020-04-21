import unittest as u
from common import CommonTest, x, y, z
from sympy import S, Integer
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude, D
)
import sympy as sp
class test_Derivatives(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_vector_symbols(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert isinstance(v1.diff(x), D)
        assert isinstance(v1.diff(x, 2), D)
        assert v1.diff(x, 2).args[0].variable_count[0][1] == 2
        assert isinstance(one.diff(x), VectorZero)
        assert isinstance(zero.diff(x), VectorZero)

        with self.assertRaises(NotImplementedError) as context:
            nabla.diff(x)
        
        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            v1.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            one.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            zero.diff(x, y)

    def test_magnitude(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = v1.mag
        assert isinstance(expr.diff(x), D)
        assert isinstance(expr.diff(x, evaluate=False), D)

        # derivatives of magnitude of vector one
        assert isinstance(one.diff(x, evaluate=False), D)
        assert one.diff(x) == VectorZero()

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            v1.mag.diff(x, y)

    def test_vecadd(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = v1 + v2
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecAdd)
        assert self._check_args(
            dexpr,
            v1.diff(x) + v2.diff(x)
        )
        assert isinstance(expr.diff(x, evaluate=False), D)
        dexpr = expr.diff(x, 3)
        assert isinstance(dexpr, VecAdd)
        assert dexpr == v1.diff(x, 3) + v2.diff(x, 3)
        dexpr = expr.diff(x, 3, evaluate=False)
        assert isinstance(dexpr, D)
        assert dexpr.args[0].variable_count[0][1] == 3

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)
    
    def test_vecmul(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = 2 * v1
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecMul)
        assert isinstance(expr.diff(x, evaluate=False), D)
        assert self._check_args(
            dexpr,
            2 * v1.diff(x)
        )
        assert self._check_args(
            dexpr.diff(x),
            2 * v1.diff(x, 2)
        )

        expr = v1.mag * v2
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecAdd)
        assert self._check_args(
            dexpr,
            v2 * v1.mag.diff(x) + v1.mag * v2.diff(x)
        )

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)
    
    def test_vecpow(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = VecPow(v1.mag, 3)
        assert isinstance(expr.diff(x), VecMul)
        assert isinstance(expr.diff(x, evaluate=False), D)
        
        expr = VecPow(v1.mag, -1)
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecMul)
        assert self._check_args(
            dexpr,
            VecMul(-1, VecPow(VecPow(v1.mag, -1), 2), v1.mag.diff(x))
        )
        assert not dexpr.is_Vector
        assert dexpr.is_Vector_Scalar

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)

    def test_normalize(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert isinstance(v1.norm.diff(x), D)
        assert isinstance(v1.norm.doit().diff(x, evaluate=False), D)
        assert isinstance(v1.norm.doit().diff(x), VecAdd)
        
        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            v1.norm.diff(x, y)

        # TODO: v1.norm.diff(x).doit() produces wrong result

    def test_vecdot(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        assert isinstance(VecDot(v1, v2).diff(x), VecAdd)
        assert isinstance(VecDot(v1, v2).diff(x, evaluate=False), D)

        expr = VecDot(v1, v2).diff(x, 2)
        assert isinstance(expr, VecAdd) and len(expr.args) == 3

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            VecDot(v1, v2).diff(x, y)
    
    def test_veccross(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = VecCross(v1, v2)
        assert isinstance(expr.diff(x), VecAdd)
        assert isinstance(expr.diff(x, evaluate=False), D)
        assert self._check_args(
            expr.diff(x),
            (v1.diff(x) ^ v2) + (v1 ^ v2.diff(x))
        )

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            VecCross(v1, v2).diff(x, y)
        
    def test_d(self):
        v1, v2, zero, one, nabla, C, vn1, vn2 = self._get_vars()
        
        expr = v1.diff(x)
        assert expr.diff(x, 3).args[0].variable_count[0][1] == 4
        expr = (v1 + v2).diff(x, 2, evaluate=False)
        assert expr.diff(x, 3).args[0].variable_count[0][1] == 5

if __name__ == "__main__":
    u.main()