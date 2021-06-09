# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 20:04:32 2021

@author: Gamer
"""

from sympy import Function, Symbol, pi, Matrix, srepr
from sympy.core.kind import Kind


class _VectorKind(Kind):
    """
    Kind for all Vector or Vector like objects (such as vector functions and vector operators)
    """
    def __new__(cls):
        return super().__new__(cls)

    def __repr__(self):
        return "VectorKind"

VectorKind = _VectorKind()

class VectorFunction(Function):
    
    kind = VectorKind
    
    def __new__(cls, name, **kwargs):
        return super().__new__(cls)


class VectorSymbol(Symbol):
    
    kind = VectorKind
    
    def __new__(cls, name, **kwargs):
        return super().__new__(cls,name)

x = Symbol('x')
F = Function('F')(x)

Fprime = F.diff(x)

TwoPi = pi + pi

V = VectorFunction('V')
W = VectorFunction('W')

Z = V+W

#Vprime = V.diff(x)

M = Matrix()
K = Matrix()
#N = Matrix([0, 1, 1])
#X = M*N
Y = M+K

A = V + x

#print("\nFunction class: ", type(F).__name__,"\nFunction Kind: ",F.kind, "\n\n" )
#print("\nFprime class: ", type(Fprime).__name__,"\nFprime Kind: ",Fprime.kind, "\n\n" )
#print("\nTwoPi class: ", type(TwoPi).__name__,"\nTwoPi Kind: ",TwoPi.kind, "\n\n" )
print("\nVector class: ", type(V).__name__,"\nVector Kind: ",V.kind, "\n V sprep: ", srepr(V),"\n\n" )
print("\nVectorAdd class: ", type(Z).__name__,"\nVectorAdd Kind: ",Z.kind, "\n Z sprep: ", srepr(Z),"\n\n" )
#print("\nVprime class: ", type(Vprime).__name__,"\nVprime Kind: ",Vprime.kind, "\n\n" )

#print("\nX class: ", type(Y).__name__,"\nX Kind: ",Y.kind, "\n X sprep: ", srepr(Y), "\n\n" )

