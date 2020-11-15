import unittest as u
from common import CommonTest
from sympy import Add, Mul, Symbol, S, Wild
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, VecPow, WildVectorSymbol, Grad
)

class test_VecMul(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        a = WildVectorSymbol("a")
        a2 = WildVectorSymbol("a_2")

        assert a.is_Vector
    
    def test_find(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        a, b, c, d, e, f, g, h = self.vector_symbols
        w1, w2, w3, w4 = [WildVectorSymbol(t) for t in ["w_1", "w_2", "w_3", "w_4"]]
        w = Wild("w")

        expr = (a & b) + (c & d)
        assert len(expr.find(w1 & w2) - set([a & b, c & d])) == 0

        expr = (a & b) + ((x * d) & (2 * (e & f) * g)) + a.mag
        assert len(expr.find(w1 & w2) - set([a & b, e & f, ((x * d) & (2 * (e & f) * g))])) == 0

        expr = (a ^ (b ^ c)) + (d ^ e) * f.mag
        assert len(expr.find(w1 & w2) - set([a ^ (b ^ c), b ^ c, d ^ e])) == 0

        expr = (nabla & ((a + b) ^ (c.mag * d))) + (e & f)
        assert len(expr.find(nabla & w1) - set([nabla & ((a + b) ^ (c.mag * d))])) == 0
        assert len(expr.find(w1 & w2) - set([nabla & ((a + b) ^ (c.mag * d)), e & f])) == 0

        expr = (nabla ^ ((a + b) ^ (c.mag * d))) + (e ^ f)
        assert len(expr.find(nabla ^ w1) - set([nabla ^ ((a + b) ^ (c.mag * d))])) == 0
        assert len(expr.find(w1 ^ w2) - set([nabla ^ ((a + b) ^ (c.mag * d)), e ^ f, (a + b) ^ (c.mag * d)])) == 0

        expr = Grad((a + b) & (c + d)) * e.mag + f
        assert len(expr.find(Grad(w1 & w2)) - set([Grad((a + b) & (c + d))])) == 0

        expr = (a & a) + ((a + b) & (a + b)) + ((a ^ b) & (a ^ b)) + (Grad(x) & Grad(x)) + (a & b) + ((a + b) & (c + d))
        assert len(expr.find(w1 & w1) - set([(a & a), ((a + b) & (a + b)), ((a ^ b) & (a ^ b)), (Grad(x) & Grad(x))])) == 0

        expr = (nabla ^ (nabla ^ (a + b))) + (nabla ^ c)
        assert len(expr.find(nabla ^ (nabla ^ w1)) - set([nabla ^ (nabla ^ (a + b))])) == 0

        expr = (nabla & (nabla ^ (a + b))) + (nabla & c)
        assert len(expr.find(nabla & (nabla ^ w1)) - set([nabla & (nabla ^ (a + b))])) == 0

        expr = (nabla ^ Grad(x)) + (a + b + (nabla ^ Grad((a + b) & c))) * 3 
        assert len(expr.find(nabla ^ Grad(w)) - set([nabla ^ Grad(x), nabla ^ Grad((a + b) & c)])) == 0


if __name__ == "__main__":
    u.main()