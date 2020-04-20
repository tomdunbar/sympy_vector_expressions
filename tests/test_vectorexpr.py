import unittest as u
from common import CommonTest, x, y, z
from sympy import Add, Mul, S, srepr, symbols
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecDot, VecCross, VectorSymbol, VectorZero, VectorOne, Nabla
)

# TODO: should Nabla be allowed as an argument of VecAdd?

def d(obj):
    print("WTF", type(obj), srepr(obj))

class test_VectorExpr(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)

    def test_expressions(self):
        # Testing expressions that had caused problems with their properties
        # is_Vector and is_Vector_Scalar. Specifically, incompatibilieties 
        # between the values of expr and its arguments
        
        a, b, c, d, e, f, g = [VectorSymbol(t) for t in 
            ["a", "b", "c", "d", "e", "f", "g"]]
        x, y, z = symbols("x:z")

        def _func(expr, expected_is_Vector=True):
            assert expr.is_Vector == expected_is_Vector
            assert expr.is_Vector_Scalar == (not expected_is_Vector)

        def _assert(expr, expected_is_Vector=True):
            _func(expr)
            for a in expr.args:
                _func(a)

        expr = b * (a & c) - c * (a & b)
        _assert(expr)
        
        expr = (b * (a & c)) - (c * (a & b)) + 2 * x * y * (d * (e & f)) - (2 * x * y * f * (d & e))
        _assert(expr)
        
        expr = (((b * (c & d)) - (d * (b & c))) * (a & c)) - (c * (a & ((b * (c & d)) - (d * (b & c)))))
        _assert(expr)
        _func(expr.args[0].args[0])
        _func(expr.args[1].args[1].args[1])

        

if __name__ == "__main__":
    u.main()