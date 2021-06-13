from vector_exprII import (
    VectorSymbol, VectorZero, VectorOne, Nabla
)
from sympy import symbols
from sympy.vector import CoordSys3D, VectorAdd

def get_vars():
    x, y, z = symbols("x:z")
    v1 = VectorSymbol("v1")
    v2 = VectorSymbol("v2")
    zero = VectorZero()
    one = VectorOne()
    nabla = Nabla()
    C = CoordSys3D("C")
    C = C
    vn1 = C.i + 2 * C.j + 3 * C.k
    vn2 = x * C.i + y * C.j + z * C.k
    
    return [v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z]

def check_args(expr1, expr2):
    """Return True if the arguments of expr1 are the same
    as the ones in expr2 (even if stored in different order)
    """
    args1 = set(expr1.args)
    args2 = set(expr2.args)
    if len(args1 - args2) == 0:
        return True
    return False


# from sympy import symbols, preorder_traversal
# from sympy.vector import CoordSys3D, Vector, VectorZero as VZero, curl, divergence

# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir) 

# from vector_exprII import (
#     VectorSymbol, VectorZero, VectorOne, Nabla
# )

# class CommonTest:
#     def setUp(self):
#         self.x, self.y, self.z = symbols("x:z")
#         self.v1 = VectorSymbol("v1")
#         self.v2 = VectorSymbol("v2")
#         self.zero = VectorZero()
#         self.one = VectorOne()
#         self.nabla = Nabla()
#         C = CoordSys3D("C")
#         self.C = C
#         self.vn1 = C.i + 2 * C.j + 3 * C.k
#         self.vn2 = self.x * C.i + self.y * C.j + self.z * C.k

#         self.vector_symbols = [VectorSymbol(t) for t in 
#             ["a", "b", "c", "d", "e", "f", "g", "h"]]
#         self.symbols = [self.x, self.y, self.z]
    
#     def _get_vars(self):
#         return [self.v1, self.v2, self.zero, self.one, self.nabla, 
#             self.C, self.vn1, self.vn2, self.x, self.y, self.z]
    
#     def _check_args(self, expr1, expr2):
#         """Return True if the arguments of expr1 are the same
#         as the ones in expr2 (even if stored in different order)
#         """
#         args1 = set(expr1.args)
#         args2 = set(expr2.args)
#         if len(args1 - args2) == 0:
#             return True
#         return False
    
#     def d(self, expr):
#         print("DEBUG", type(expr), expr)


# def _get_vars():
#     x, y, z = symbols("x:z")
#     v1 = VectorSymbol("v1")
#     v2 = VectorSymbol("v2")
#     zero = VectorZero()
#     one = VectorOne()
#     nabla = Nabla()
#     C = CoordSys3D("C")
#     C = C
#     vn1 = C.i + 2 * C.j + 3 * C.k
#     vn2 = x * C.i + y * C.j + z * C.k

#     # vector_symbols = [VectorSymbol(t) for t in 
#     #     ["a", "b", "c", "d", "e", "f", "g", "h"]]
#     # symbols = [x, y, z]
    
#     return [v1, v2, zero, one, nabla, 
#         C, vn1, vn2, x, y, z]