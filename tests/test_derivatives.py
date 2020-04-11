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
        assert isinstance(self.v1.diff(x), D)
        assert isinstance(self.v1.diff(x, 2), D)
        assert self.v1.diff(x, 2).args[0].variable_count[0][1] == 2
        assert isinstance(self.one.diff(x), VectorZero)
        assert isinstance(self.zero.diff(x), VectorZero)

        with self.assertRaises(NotImplementedError) as context:
            self.nabla.diff(x)
        
        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            self.v1.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            self.one.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            self.zero.diff(x, y)

    def test_magnitude(self):
        expr = self.v1.mag
        assert isinstance(expr.diff(x), D)
        assert isinstance(expr.diff(x, evaluate=False), D)

        # derivatives of magnitude of vector one
        assert isinstance(self.one.diff(x, evaluate=False), D)
        assert self.one.diff(x) == VectorZero()

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            self.v1.mag.diff(x, y)

    def test_vecadd(self):
        expr = self.v1 + self.v2
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecAdd)
        assert self._check_args(
            dexpr,
            self.v1.diff(x) + self.v2.diff(x)
        )
        assert isinstance(expr.diff(x, evaluate=False), D)
        dexpr = expr.diff(x, 3)
        assert isinstance(dexpr, VecAdd)
        assert dexpr == self.v1.diff(x, 3) + self.v2.diff(x, 3)
        dexpr = expr.diff(x, 3, evaluate=False)
        assert isinstance(dexpr, D)
        assert dexpr.args[0].variable_count[0][1] == 3

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)
    
    def test_vecmul(self):
        expr = 2 * self.v1
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecMul)
        assert isinstance(expr.diff(x, evaluate=False), D)
        assert self._check_args(
            dexpr,
            2 * self.v1.diff(x)
        )
        assert self._check_args(
            dexpr.diff(x),
            2 * self.v1.diff(x, 2)
        )

        expr = self.v1.mag * self.v2
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecAdd)
        assert self._check_args(
            dexpr,
            self.v2 * self.v1.mag.diff(x) + self.v1.mag * self.v2.diff(x)
        )

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)
    
    def test_vecpow(self):
        expr = VecPow(self.v1.mag, 3)
        assert isinstance(expr.diff(x), VecMul)
        assert isinstance(expr.diff(x, evaluate=False), D)
        
        expr = VecPow(self.v1.mag, -1)
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecMul)
        assert self._check_args(
            dexpr,
            VecMul(-1, VecPow(VecPow(self.v1.mag, -1), 2), self.v1.mag.diff(x))
        )
        assert not dexpr.is_Vector
        assert dexpr.is_Vector_Scalar

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr.diff(x, y)

    def test_normalize(self):
        assert isinstance(self.v1.norm.diff(x), D)
        assert isinstance(self.v1.norm.doit().diff(x, evaluate=False), D)
        assert isinstance(self.v1.norm.doit().diff(x), VecAdd)
        
        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            self.v1.norm.diff(x, y)

        # TODO: self.v1.norm.diff(x).doit() produces wrong result

    def test_vecdot(self):
        assert isinstance(VecDot(self.v1, self.v2).diff(x), VecAdd)
        assert isinstance(VecDot(self.v1, self.v2).diff(x, evaluate=False), D)

        expr = VecDot(self.v1, self.v2).diff(x, 2)
        assert isinstance(expr, VecAdd) and len(expr.args) == 3

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            VecDot(self.v1, self.v2).diff(x, y)
    
    # def test_veccross(self):
    #     expr = VecCross(self.v1, self.v2)
    #     assert isinstance(expr.diff(x), VecAdd)
    #     assert isinstance(expr.diff(x, evaluate=False), D)
    #     assert self._check_args(
    #         expr.diff(x),
    #         (self.v1.diff(x) ^ self.v2) + (self.v1 ^ self.v2.diff(x))
    #     )

    #     # partial differentiation not implemented
    #     with self.assertRaises(NotImplementedError) as context:
    #         VecCross(self.v1, self.v2).diff(x, y)
        
    # def test_d(self):
    #     expr = self.v1.diff(x)
    #     assert expr.diff(x, 3).args[0].variable_count[0][1] == 4
    #     expr = (self.v1 + self.v2).diff(x, 2, evaluate=False)
    #     assert expr.diff(x, 3).args[0].variable_count[0][1] == 5

if __name__ == "__main__":
    u.main()