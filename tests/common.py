from sympy import symbols
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