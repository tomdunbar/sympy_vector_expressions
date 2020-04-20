from sympy import symbols, preorder_traversal
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero, curl, divergence

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VectorSymbol, VectorZero, VectorOne, Nabla
)

x, y, z = symbols("x:z")

class CommonTest:
    def setUp(self):
        self.v1 = VectorSymbol("v1")
        self.v2 = VectorSymbol("v2")
        self.zero = VectorZero()
        self.one = VectorOne()
        self.nabla = Nabla()
        C = CoordSys3D("C")
        self.C = C
        self.vn1 = C.i + 2 * C.j + 3 * C.k
        self.vn2 = x * C.i + y * C.j + z * C.k
    
    def _check_args(self, expr1, expr2):
        """Return True if the arguments of expr1 are the same
        as the ones in expr2 (even if stored in different order)
        """
        args1 = set(expr1.args)
        args2 = set(expr2.args)
        if len(args1 - args2) == 0:
            return True
        return False
    
    def _assert_vecadd(self, expr):
        """ 
        """
        assert expr.is_Vector
        assert not expr.is_Vector_Scalar
        for a in expr.args:
            assert a.is_Vector
            assert not a.is_Vector_Scalar
    
    def _assert_expr_tree(self, expr, is_Vector=True):
        """ Parse the expression tree, select the arguments of type VecAdd, 
        VecMul, VecPow, VecCross, D and look for consistency in the is_Vector 
        and is_Vector_Scalar properties.

        Parameters
        ----------
            expr : the expression to parse
            is_Vector : the expected value of the expr.is_Vector
        """
        assert expr.is_Vector == is_Vector
        assert expr.is_Vector_Scalar == (not is_Vector)

        if isinstance(expr, VecAdd):
            
            for arg in expr.args:
                pass


        for arg in preorder_traversal(expr):
            pass
    
    def d(self, expr):
        print("DEBUG", type(expr), expr)