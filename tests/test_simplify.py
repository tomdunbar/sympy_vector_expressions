import unittest as u
from common import CommonTest
from sympy import Add, Mul, S, srepr, symbols, Pow
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VecDot, VecCross, VectorSymbol, 
    VectorZero, VectorOne, Nabla, Grad, Laplace, Advection
)

from vector_simplify import (
    bac_cab_backward, bac_cab_forward, find_bac_cab, bac_cab,
    dot_cross, collect, collect_cross_dot, simplify, identities
)

class test_Simplify(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
        self.vector_symbols = [VectorSymbol(t) for t in 
            ["a", "b", "c", "d", "e", "f", "g", "h"]]
        self.symbols = symbols("x:z")

    def test_bac_cab_forward(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        # basic
        expr = a ^ (b ^ c)
        sexpr = bac_cab_forward(expr)
        assert self._check_args(
            sexpr, 
            (b * (a & c) - c * (a & b))
        )
        
        expr = a ^ (b ^ (c ^ d))
        sexpr = bac_cab_forward(expr)
        assert isinstance(sexpr, VecCross)
        assert self._check_args(
            sexpr.args[1], 
            (c * (b & d) - d * (b & c))
        )

        expr = (a ^ (b ^ c)) + (d ^ (e ^ f))
        sexpr = bac_cab_forward(expr)
        assert self._check_args(
            sexpr, 
            (b * (a & c) - c * (a & b)) + (e * (d & f) - f * (d & e))
        )
        
        # arguments are VectorExpr
        expr = (a - b) ^ (3 * b ^ 4 * (c + d))
        sexpr = bac_cab_forward(expr)
        assert self._check_args(
            sexpr, 
            VecMul(-1, (4 * c + 4 * d), evaluate=False) * ((a - b) & (3 * b)) + (3 * ((a - b) & (4 * c + 4 * d)) * b)
        )

        # nested cross-products
        expr = (a ^ (b ^ (c ^ (d ^ e))))
        sexpr = bac_cab_forward(expr)
        assert self._check_args(
            sexpr, 
            VecMul(-1, d * (c & e) - e * (c & d), (a & b), evaluate=False) + (a & (d * (c & e) - e * (c & d))) * b
        )
    
    def test_dot_cross(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        expr = a & (b ^ c)
        sexpr = dot_cross(expr)
        assert sexpr == (a ^ b) & c

        expr = (a & (b ^ c)) + (d & (e ^ f))
        sexpr = dot_cross(expr)
        assert sexpr == ((a ^ b) & c) + ((d ^ e) & f)

        # nested 
        expr = (a & (b ^ (c + d * (e & (g ^ h)))))
        sexpr = dot_cross(expr)
        assert self._check_args(
            sexpr,
            (a ^ b) & (((e ^ g) & h) * d + c)
        )

    def test_find_bac_cab(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        expr = (b * (a & c)) - (c * (a & b))
        matches = find_bac_cab(expr)
        assert len(matches) == 1
        assert self._check_args(
            matches[0],
            expr
        )

        # with scalar coefficients
        expr = (b * (a & c)) - (c * (a & b)) + 2 * x * y * (d * (e & f)) - (2 * x * y * f * (d & e))
        matches = find_bac_cab(expr)
        assert len(matches) == 2
        assert self._check_args(
            matches[1],
            -2 * x * y * (d & e) * f + 2 * x * y * (e & f) * d
        )
        assert self._check_args(
            matches[0],
            (b * (a & c)) - (c * (a & b))
        )

        # nested
        expr = (((b * (c & d)) - (d * (b & c))) * (a & c)) - (c * (a & ((b * (c & d)) - (d * (b & c)))))
        matches = find_bac_cab(expr)
        assert len(matches) == 2
        assert self._check_args(
            matches[1],
            expr
        )
        assert self._check_args(
            matches[0],
            ((c & d) * b) - (b & c) * d
        )

        # embedded into arguments
        expr = (((b * (c & d)) - (d * (b & c))) * (a & c))
        matches = find_bac_cab(expr)
        assert len(matches) == 1
        assert self._check_args(
            matches[0],
            (b * (c & d)) - (d * (b & c))
        )
    
    def test_bac_cab_backward(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        expr = (b * (a & c)) - (c * (a & b))
        sexpr = bac_cab_backward(expr)
        assert self._check_args(
            sexpr,
            a ^ (b ^ c)
        )

        expr = (b * (a & c)) - (c * (a & b)) + 2 * x * y * (d * (e & f)) - (2 * x * y * f * (d & e))
        sexpr = bac_cab_backward(expr)
        assert self._check_args(
            sexpr,
            2 * x * y * (e ^ (d ^ f)) + (a ^ (b ^ c))
        )

        expr = (((b * (c & d)) - (d * (b & c))) * (a & c)) - (c * (a & ((b * (c & d)) - (d * (b & c)))))
        sexpr = bac_cab_backward(expr)
        assert self._check_args(
            sexpr,
            a ^ ((c ^ (b ^ d)) ^ c)
        )

        expr = (((b * (c & d)) - (d * (b & c))) * (a & c))
        sexpr = bac_cab_backward(expr)
        assert self._check_args(
            sexpr,
            (a & c) * (c ^ (b ^ d))
        )
    
    def test_bac_cab(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        # forward
        expr = (a - b) ^ (3 * b ^ 4 * (c + d))
        sexpr = bac_cab(expr)
        assert self._check_args(
            sexpr, 
            VecMul(-1, (4 * c + 4 * d), evaluate=False) * ((a - b) & (3 * b)) + (3 * ((a - b) & (4 * c + 4 * d)) * b)
        )

        # nested cross-products
        expr = (a ^ (b ^ (c ^ (d ^ e))))
        sexpr = bac_cab(expr)
        assert self._check_args(
            sexpr, 
            VecMul(-1, d * (c & e) - e * (c & d), (a & b), evaluate=False) + (a & (d * (c & e) - e * (c & d))) * b
        )

        # backward
        expr = (b * (a & c)) - (c * (a & b))
        sexpr = bac_cab(expr, False)
        assert self._check_args(
            sexpr,
            a ^ (b ^ c)
        )

        expr = (b * (a & c)) - (c * (a & b)) + 2 * x * y * (d * (e & f)) - (2 * x * y * f * (d & e))
        sexpr = bac_cab(expr, False)
        assert self._check_args(
            sexpr,
            2 * x * y * (e ^ (d ^ f)) + (a ^ (b ^ c))
        )

        expr = (((b * (c & d)) - (d * (b & c))) * (a & c)) - (c * (a & ((b * (c & d)) - (d * (b & c)))))
        sexpr = bac_cab(expr, False)
        assert self._check_args(
            sexpr,
            a ^ ((c ^ (b ^ d)) ^ c)
        )

        expr = (((b * (c & d)) - (d * (b & c))) * (a & c))
        sexpr = bac_cab(expr, False)
        assert self._check_args(
            sexpr,
            (a & c) * (c ^ (b ^ d))
        )

    # def test_collect(self):
    #     a, b, c, d, e, f, g, h = self.vector_symbols
    #     x, y, z = self.symbols

    #     # collecting dot products
    #     expr = (a * (a & b)**3) + (b * (a & b)**2) + self.one
    #     sexpr = collect(expr, (a & b))
    #     assert self._check_args(
    #         sexpr,
    #         self.one + (a & b) * ((a & b)**2 * a + (a & b) * b)
    #     )

    #     sexpr = collect(expr, (a & b)**2)
    #     print("ASD")
    #     print(expr)
    #     print(sexpr)
    #     assert self._check_args(
    #         sexpr,
    #         (((a & b) * a + b) * (a & b)**2) + self.one
    #     )

    #     sexpr = collect(expr, (a & b)**3)
    #     assert self._check_args(
    #         sexpr,
    #         expr
    #     )

    #     expr = a * (a & b)**3 + self.one + b * (a & b)**2 + (a & b) * c
    #     sexpr = collect(expr, (a & b))
    #     assert self._check_args(
    #         sexpr,
    #         ((a & b)**2 * a + (a & b) * b + c) * (a & b) + self.one
    #     )

    #     sexpr = collect(expr, (a & b)**2)
    #     assert self._check_args(
    #         sexpr,
    #         ((a & b) * a + b) * (a & b)**2 + (a & b) * c + self.one
    #     )

    #     # collect cross product
    #     expr = (a ^ b) * c.mag**2 + (a ^ b)
    #     sexpr = collect(expr, a ^ b)
    #     assert self._check_args(
    #         sexpr,
    #         (c.mag**2 + 1) * (a ^ b)
    #     )

    #     expr = x**2 * (a ^ b) * c.mag**2 + (a ^ b) * y
    #     sexpr = collect(expr, a ^ b)
    #     assert self._check_args(
    #         sexpr,
    #         (x**2 * c.mag**2 + y) * (a ^ b)
    #     )

    #     # fall back to expr.collect(match)
    #     expr = c.mag * self.one + c.mag * a + c.mag**2 * b + c.mag**3 * c
    #     sexpr = collect(expr, c.mag)
    #     # TODO: wtf is going on here????????
    #     dc = "VecAdd(VecMul(VecAdd(VectorOne(Symbol('1')), VectorSymbol(Symbol('a'))), Magnitude(VectorSymbol(Symbol('c')))), VecMul(VecPow(Magnitude(VectorSymbol(Symbol('c'))), Integer(3)), VectorSymbol(Symbol('c'))), VecMul(VecPow(Magnitude(VectorSymbol(Symbol('c'))), Integer(2)), VectorSymbol(Symbol('b'))))"
    #     assert srepr(sexpr) == dc
    #     # assert self._check_args(
    #     #     sexpr,
    #     #     ((self.one + a) * c.mag) + (Pow(c.mag, 2) * b) + (Pow(c.mag, 3) * c)
    #     # #     VecAdd(VecMul(VecAdd(self.one, a), c.mag), VecMul(Pow(c.mag, 3), c), VecMul(Pow(c.mag, 2), b))
    #     # )

    #     expr = (a + b) * a.mag + (a + b) * b.mag
    #     sexpr = collect(expr, a + b)
    #     assert self._check_args(
    #         sexpr,
    #         (a + b) * (a.mag + b.mag)
    #     )

    #     expr = x * a + y * a + 4 * a
    #     sexpr = collect(expr, a)
    #     assert self._check_args(
    #         sexpr,
    #         (x + y + 4) * a
    #     )
    
    def test_collect_cross_dot(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        # Cross products
        expr = (a ^ b) + (a ^ c) + (b ^ c)
        assert self._check_args(
            collect_cross_dot(expr),
            (a ^ (b + c)) + (b ^ c)
        )

        assert self._check_args(
            collect_cross_dot(expr, VecCross, False),
            ((a + b) ^ c) + (a ^ b)
        )

        expr = (a ^ (b ^ c)) + (a ^ (b ^ d)) + (a ^ (b ^ e)) + (a ^ (e ^ d)) + (c ^ (d ^ e)) + (c ^ (d ^ f)) + (b ^ (c ^ d)) + (b ^ (c ^ e))
        assert self._check_args(
            collect_cross_dot(expr),
            (a ^ ((b ^ (c + d + e)) + (e ^ d))) + (b ^ (c ^ (d + e))) + (c ^ (d ^ (e + f)))
        )
        assert self._check_args(
            collect_cross_dot(expr, VecCross, False),
            expr
        )

        # Dot products
        expr = (a & b) + (a & c) + (b & c)
        assert self._check_args(
            collect_cross_dot(expr, VecDot),
            (a & (b + c)) + (b & c)
        )
        assert self._check_args(
            collect_cross_dot(expr, VecDot, False),
            (a & b) + ((a + b) & c)
        )

    def test_expand(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols
        nabla = Nabla()

        # expand general expressions
        expr = self.one + 2 * (a ^ (b + c))
        assert self._check_args(
            expr.expand(cross=False),
            expr
        )
        assert self._check_args(
            expr.expand(cross=True),
            self.one + 2 * (a ^ b) + 2 * (a ^ c)
        )

        expr = (x + y) * a + Grad(x + y * z)
        assert self._check_args(
            expr.expand(),
            x * a + y * a + Grad(x + y * z)
        )
        assert self._check_args(
            expr.expand(gradient=True),
            x * a + y * a + Grad(x) + Grad(y * z)
        )
        assert self._check_args(
            expr.expand(gradient=True, prod=True),
            x * a + y * a + Grad(x) + y * Grad(z) + z * Grad(y)
        )

        expr = self.one + (c ^ ((d + e) ^ f))
        assert self._check_args(
            expr.expand(cross=True),
            self.one + (c ^ (d ^ f)) + (c ^ (e ^ f))
        )

        expr = x * ((a & (b + (c & (d + e) * f.mag) * g)) + (((a + b) ^ c) & d))
        assert self._check_args(
            expr.expand(dot=True, cross=True),
            (x * (a & b) + x * f.mag * (a & g) * (c & d) + 
                x * f.mag *  (a & g) * (c & e) + x * ((a ^ c) & d) + 
                x * ((b ^ c) & d))
        )

        expr = Laplace(a + b) + ((c + d) ^ Grad(x * y))
        assert self._check_args(
            expr.expand(cross=True),
            Laplace(a + b) + (c ^ Grad(x * y)) + (d ^ Grad(x * y))
        )
        assert self._check_args(
            expr.expand(cross=True, laplacian=True),
            Laplace(a) +  Laplace(b) + (c ^ Grad(x * y)) + (d ^ Grad(x * y))
        )
        assert self._check_args(
            expr.expand(cross=True, laplacian=True, gradient=True, prod=True),
            Laplace(a) +  Laplace(b) + (x * (c ^ Grad(y))) + (y * (c ^ Grad(x))) + (x * (d ^ Grad(y))) + (y * (d ^ Grad(x)))
        )

    def test_simplify(self):
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        # RULE 1
        expr = (a & a)
        assert self._check_args(
            simplify(expr),
            a.mag**2
        )

        expr = (a & a) + (b & b) + (a & b)
        assert self._check_args(
            simplify(expr),
            a.mag**2 + b.mag**2 + (a & b)
        )

        

        expr = ((a & a) + (b & b) + (a & b)) * x
        assert self._check_args(
            simplify(expr),
            x * (a.mag**2 + b.mag**2 + (a & b))
        )

        # RULE 2
        expr = b * (a & c) - c * (a & b) + (2 * d)
        assert self._check_args(
            simplify(expr),
            (a ^ (b ^ c)) + (2 * d)
        )

        expr = b * (a & c) - c * (a & b) + 2 * x * e * (d & f) - 2 * x * f * (e & d)
        assert self._check_args(
            simplify(expr),
            (a ^ (b ^ c)) + 2 * x * (d ^ (e ^ f))
        )

        expr = d ^ (b * (a & c) - c * (a & b))
        assert self._check_args(
            simplify(expr),
            d ^ (a ^ (b ^ c))
        )

        # RULE 3
        expr = 5 * a & (x * b)
        assert self._check_args(
            simplify(expr),
            5 * x * (a & b)
        )

        expr = (4 * a) & (((3 * (b + c)) & (2 * d)) * e)
        assert self._check_args(
            simplify(expr),
            24 * ((b + c) & d) * (a & e)
        )

        # RULE 4
        expr = 5 * a ^ (x * b)
        assert self._check_args(
            simplify(expr),
            5 * x * (a ^ b)
        )

        expr = 4 * a ^ ((3 * (b + c)) ^ 2 * d)
        assert self._check_args(
            simplify(expr),
            24 * (a ^ ((b + c) ^ d))
        )

        expr = 4 * a ^ ((3 * (b + c)) ^ (2 * (3 * d ^ (x * e))))
        assert self._check_args(
            simplify(expr),
            72 * x * (a ^ ((b + c) ^ (d ^ e)))
        )
        
        expr = (2 * a) ^ ((((3 * b) & (4 * c)) * d) ^ (5 * e))
        assert self._check_args(
            simplify(expr),
            120 * (b & c) * (a ^ (d ^ e))
        )

        # RULE 5
        expr = (a ^ b) + (a ^ c) + (d ^ e) + (f ^ e)
        assert self._check_args(
            simplify(expr),
            ((d + f) ^ e) + (a ^ (b + c))
        )

        expr = a.mag * ((a ^ (b ^ c)) + (a ^ (b ^ d))) + (d ^ e) + (f ^ e)
        assert self._check_args(
            simplify(expr),
            a.mag * (a ^ (b ^ (c + d))) + ((d + f) ^ e)
        )

        # RULE 6
        expr = (a & b) + (a & c)
        assert self._check_args(
            simplify(expr),
            a & (b + c)
        )

        expr = (a & b) + (a & c) + (d & e) + (f & e)
        assert self._check_args(
            simplify(expr),
            (a & (b + c)) + ((d + f) & e)
        )   
    
    def test_identities(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        a, b, c, d, e, f, g, h = self.vector_symbols
        x, y, z = self.symbols

        # Identity C
        expr = (nabla & (x * a))
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, prod_div=True),
            (Grad(x) & a) + (x * (nabla & a))
        )

        expr = (nabla & (x * a)) + 4 * (nabla & (y * a))
        assert self._check_args(
            identities(expr, prod_div=True),
            # (Grad(x) & a) + (x * (nabla & a)) + 4 * (Grad(y) & a) + Mul(4, y) * (nabla & a)
            (Grad(x) & a) + (x * (nabla & a)) + 4 * (Grad(y) & a) + 4 * y * (nabla & a)
        )

        # Identity D
        expr = (nabla ^ (x * a))
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, prod_curl=True),
            (Grad(x) ^ a) + (x * (nabla ^ a))
        )

        expr = (nabla ^ (x * a)) + 4 * (nabla ^ (y * a))
        assert self._check_args(
            identities(expr, prod_curl=True),
            (Grad(x) ^ a) + (x * (nabla ^ a)) + 4 * ((Grad(y) ^ a) + (y * (nabla ^ a)))
        )

        # Identity E
        expr = (nabla & (a ^ b))
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, div_of_cross=True),
            ((nabla ^ a) & b) - ((nabla ^ b) & a)
        )

        expr = (nabla & (a ^ b)) + 4 * (nabla & (c ^ d))
        assert self._check_args(
            identities(expr, div_of_cross=True),
            ((nabla ^ a) & b) - ((nabla ^ b) & a) + 4 * (((nabla ^ c) & d) - ((nabla ^ d) & c))
        )

        # Identity F
        expr = (nabla ^ (a ^ b))
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, curl_of_cross=True),
            ((nabla & b) * a) + Advection(b, a) - ((nabla & a) * b) - Advection(a, b)
        )

        expr = (nabla ^ (a ^ b)) + 4 * (nabla ^ (c ^ d))
        assert self._check_args(
            identities(expr, curl_of_cross=True),
            (((nabla & b) * a) + Advection(b, a) - ((nabla & a) * b) - Advection(a, b)
            + 4 * (((nabla & d) * c) + Advection(d, c) - ((nabla & c) * d) - Advection(c, d)))
        )

        # Identity G
        expr = Grad(a & b)
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, grad_of_dot=True),
            Advection(a, b) + Advection(b, a) + (a ^ (nabla ^ b)) + (b ^ (nabla ^ a))
        )

        expr = Grad(a & b) + 4 * Grad(c & d)
        assert self._check_args(
            identities(expr, grad_of_dot=True),
            (Advection(a, b) + Advection(b, a) + (a ^ (nabla ^ b)) + (b ^ (nabla ^ a))
            + 4 * (Advection(c, d) + Advection(d, c) + (c ^ (nabla ^ d)) + (d ^ (nabla ^ c))))
        )

        # Identity H
        expr = nabla ^ Grad(x)
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, curl_of_grad=True),
            VectorZero()
        )

        expr = (nabla ^ Grad(x)) + 4 * (nabla ^ Grad(y))
        assert self._check_args(
            identities(expr, curl_of_grad=True),
            VectorZero()
        )

        # Identity I
        expr = nabla & (nabla ^ a)
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, div_of_curl=True),
            S.Zero
        )

        expr = (nabla & (nabla ^ a)) + 4 * (nabla & (nabla ^ b))
        assert self._check_args(
            identities(expr, div_of_curl=True),
            S.Zero
        )

        # Identity J
        expr = nabla ^ (nabla ^ a)
        assert self._check_args(
            expr,
            identities(expr)
        )
        assert self._check_args(
            identities(expr, curl_of_curl=True),
            Grad(nabla & a) - Laplace(a)
        )

        expr = (nabla ^ (nabla ^ a)) + 4 * (nabla ^ (nabla ^ b))
        assert self._check_args(
            identities(expr, curl_of_curl=True),
            Grad(nabla & a) - Laplace(a) + 4 * (Grad(nabla & b) - Laplace(b))
        )
        
    

if __name__ == "__main__":
    u.main()