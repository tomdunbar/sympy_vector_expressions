# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:34:52 2021

@author: Gamer
"""

# import unittest as u
# from common import CommonTest
# from sympy import Add, Mul, S, srepr, symbols
# from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 

# from vector_expr import (
#     VecAdd, VecMul, VecDot, VecCross, VectorSymbol, VectorZero, VectorOne, Nabla, D
# )


# def setUp():
#         CommonTest.setUp()

# # def test_expressions():
#     # Testing expressions that had caused problems with their properties
#     # is_Vector and is_Vector_Scalar. Specifically, incompatibilieties 
#     # between the values of expr and its arguments
    
# a, b, c, d, e, f, g = [VectorSymbol(t) for t in 
#     ["a", "b", "c", "d", "e", "f", "g"]]
# x, y, z = symbols("x:z")

# #print("I'm here")

# def _func(expr, expected_is_Vector=True):
#     assert expr.is_Vector == expected_is_Vector
#     assert expr.is_scalar == (not expected_is_Vector)

# def _assert(expr, expected_is_Vector=True):
#     _func(expr)
#     for a in expr.args:
#         _func(a)

# expr = b * (a & c) - c * (a & b)
# _assert(expr)

# print("I'm here")
# expr = (b * (a & c)) - (c * (a & b)) + 2 * x * y * (d * (e & f)) - (2 * x * y * f * (d & e))
# _assert(expr)

# expr = (((b * (c & d)) - (d * (b & c))) * (a & c)) - (c * (a & ((b * (c & d)) - (d * (b & c)))))
# _assert(expr)
# #     _func(expr.args[0].args[0])
# #     _func(expr.args[1].args[1].args[1])

# def test_equals():
#     a, b, c, d, e, f, g = [VectorSymbol(t) for t in 
#         ["a", "b", "c", "d", "e", "f", "g"]]
#     x, y, z = symbols("x:z")

#     assert (a + b) == (a + b)
#     assert not ((a + b) == (a + b * 2))

# #test_expressions()
# #test_equals()

# def printAttri(obj):
#     temp = vars(obj)
#     for item in temp:
#         print(item, ':', temp[item])


"""Test Derivatives"""

import unittest as u
from common import CommonTest
from sympy import S, Integer, symbols, Function
from sympy.vector import CoordSys3D, Vector, VectorAdd, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecPow, VectorSymbol, VectorZero, VectorOne, Nabla,
    VecDot, VecCross, Normalize, Magnitude, D, wtf
)


x, y,z = symbols("x:z")

v1 = VectorSymbol("v1")
v2 = VectorSymbol("v2")

# type(D)
# type(v1.diff(x))

# assert isinstance(v1.diff(x), D)

# DotProd = v1 & v2
# DotDiff = (v1 & v2).diff(x, evaluate=False)

# print(DotProd.is_scalar)
# print(DotDiff.is_scalar)

# expr = v1.diff(x)
# thing = expr.diff(x, 3)

# print(thing.args[0])

f = symbols('f', cls= Function)

print(y.diff(x))
print(f.diff(x))