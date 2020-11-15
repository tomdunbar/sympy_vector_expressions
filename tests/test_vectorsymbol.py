import unittest as u
from common import CommonTest
from sympy import Symbol, S
from sympy.vector import CoordSys3D, Vector, VectorZero as VZero

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from vector_expr import (
    VecAdd, VecMul, VectorSymbol, VectorZero, VectorOne, 
    Nabla, Magnitude, Normalize
)

class test_VectorSymbol(u.TestCase, CommonTest):
    def setUp(self):
        CommonTest.setUp(self)
    
    def test_creation(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert isinstance(VectorSymbol("v1"), VectorSymbol)
        assert isinstance(VectorSymbol(Symbol("x")), VectorSymbol)
        assert VectorSymbol(v1) == v1

        def func(s):
            with self.assertRaises(TypeError):
                VectorSymbol(s)

        func(x + y)
        func(VecAdd(v1, v2))
        func(Symbol("x") + Symbol("y"))
    
    def test_attributes(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert hasattr(v1, "_vec_symbol")
        assert hasattr(v1, "_unit_vec_symbol")
        assert hasattr(v1, "_bold")
        assert hasattr(v1, "_italic")
    
    def test_is_vector(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert v1.is_Vector
        assert v2.is_Vector
        assert one.is_Vector
        assert zero.is_Vector
        assert nabla.is_Vector
    
    def test_zero_one_nabla(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        # magnitudes
        assert zero.mag == S.Zero
        assert isinstance(one.mag, Magnitude)
        # unit vector
        assert isinstance(zero.norm, VectorZero)
        assert isinstance(one.norm, Normalize)

        with self.assertRaises(TypeError):
            nabla.mag
        with self.assertRaises(TypeError):
            nabla.norm
    
    def test_instances(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert isinstance(zero, VectorSymbol)
        assert isinstance(one, VectorSymbol)
        assert isinstance(nabla, VectorSymbol)
    
    def test_doit(self):
        v1, v2, zero, one, nabla, C, vn1, vn2, x, y, z = self._get_vars()
        
        assert v1.doit() == v1
        assert one.doit() == one
        assert zero.doit() == zero
        assert nabla.doit() == nabla

if __name__ == "__main__":
    u.main()