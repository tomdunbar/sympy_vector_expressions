import unittest as u
from common import CommonTest, x, y, z
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
        assert VectorSymbol("v1")
        assert VectorSymbol(Symbol("x"))
        assert VectorSymbol(self.v1)

        def func(s):
            with self.assertRaises(TypeError) as context:
                VectorSymbol(s)
            self.assertTrue("'name' must be a string or a Symbol or a VectorSymbol" in str(context.exception))
        
        func(VecAdd(self.v1, self.v2))
        func(Symbol("x") + Symbol("y"))
    
    def test_attributes(self):
        assert hasattr(self.v1, "_vec_symbol")
        assert hasattr(self.v1, "_unit_vec_symbol")
        assert hasattr(self.v1, "_bold")
        assert hasattr(self.v1, "_italic")
    
    def test_is_vector(self):
        assert self.v1.is_Vector
        assert self.v2.is_Vector
        assert self.one.is_Vector
        assert self.zero.is_Vector
        assert self.nabla.is_Vector
    
    def test_zero_one_nabla(self):
        # magnitudes
        assert self.zero.mag == S.Zero
        assert isinstance(self.one.mag, Magnitude)
        # unit vector
        assert isinstance(self.zero.norm, VectorZero)
        assert isinstance(self.one.norm, Normalize)

        with self.assertRaises(TypeError):
            self.nabla.mag
        with self.assertRaises(TypeError):
            self.nabla.norm
    
    def test_instances(self):
        assert isinstance(self.zero, VectorSymbol)
        assert isinstance(self.one, VectorSymbol)
        assert isinstance(self.nabla, VectorSymbol)
    
    def test_doit(self):
        assert self.v1.doit() == self.v1
        assert self.one.doit() == self.one
        assert self.zero.doit() == self.zero
        assert self.nabla.doit() == self.nabla

if __name__ == "__main__":
    u.main()