# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 19:52:21 2021

@author: Tom Dunbar

This is a scratch pad for ideas for vector tests
"""

"""Test these Algebraic Vector Identities:

    https://en.wikipedia.org/wiki/Vector_algebra_relations

Commutativity of addition: {\displaystyle \mathbf {A} +\mathbf {B} =\mathbf {B} +\mathbf {A} }{\displaystyle \mathbf {A} +\mathbf {B} =\mathbf {B} +\mathbf {A} }.
Commutativity of scalar product: {\displaystyle \mathbf {A} \cdot \mathbf {B} =\mathbf {B} \cdot \mathbf {A} }{\displaystyle \mathbf {A} \cdot \mathbf {B} =\mathbf {B} \cdot \mathbf {A} }.
Anticommutativity of cross product: {\displaystyle \mathbf {A} \times \mathbf {B} =\mathbf {-B} \times \mathbf {A} }{\displaystyle \mathbf {A} \times \mathbf {B} =\mathbf {-B} \times \mathbf {A} }.
Distributivity of multiplication by a scalar over addition: {\displaystyle c(\mathbf {A} +\mathbf {B} )=c\mathbf {A} +c\mathbf {B} }{\displaystyle c(\mathbf {A} +\mathbf {B} )=c\mathbf {A} +c\mathbf {B} }.
Distributivity of scalar product over addition: {\displaystyle \left(\mathbf {A} +\mathbf {B} \right)\cdot \mathbf {C} =\mathbf {A} \cdot \mathbf {C} +\mathbf {B} \cdot \mathbf {C} }{\displaystyle \left(\mathbf {A} +\mathbf {B} \right)\cdot \mathbf {C} =\mathbf {A} \cdot \mathbf {C} +\mathbf {B} \cdot \mathbf {C} }.
Distributivity of vector product over addition: {\displaystyle (\mathbf {A} +\mathbf {B} )\times \mathbf {C} =\mathbf {A} \times \mathbf {C} +\mathbf {B} \times \mathbf {C} }{\displaystyle (\mathbf {A} +\mathbf {B} )\times \mathbf {C} =\mathbf {A} \times \mathbf {C} +\mathbf {B} \times \mathbf {C} }.
Scalar triple product: {\displaystyle \mathbf {A} \cdot (\mathbf {B} \times \mathbf {C} )=\mathbf {B} \cdot (\mathbf {C} \times \mathbf {A} )=\mathbf {C} \cdot (\mathbf {A} \times \mathbf {B} )}{\displaystyle \mathbf {A} \cdot (\mathbf {B} \times \mathbf {C} )=\mathbf {B} \cdot (\mathbf {C} \times \mathbf {A} )=\mathbf {C} \cdot (\mathbf {A} \times \mathbf {B} )}{\displaystyle =|\mathbf {A} \,\mathbf {B} \,\mathbf {C} |=\left|{\begin{array}{ccc}A_{x}&B_{x}&C_{x}\\A_{y}&B_{y}&C_{y}\\A_{z}&B_{z}&C_{z}\end{array}}\right|}{\displaystyle =|\mathbf {A} \,\mathbf {B} \,\mathbf {C} |}.
Vector triple product: {\displaystyle \mathbf {A} \times (\mathbf {B} \times \mathbf {C} )=(\mathbf {A} \cdot \mathbf {C} )\mathbf {B} -(\mathbf {A} \cdot \mathbf {B} )\mathbf {C} }{\displaystyle \mathbf {A} \times (\mathbf {B} \times \mathbf {C} )=(\mathbf {A} \cdot \mathbf {C} )\mathbf {B} -(\mathbf {A} \cdot \mathbf {B} )\mathbf {C} }.
Jacobi identity: {\displaystyle \mathbf {A} \times (\mathbf {B} \times \mathbf {C} )+\mathbf {C} \times (\mathbf {A} \times \mathbf {B} )+\mathbf {B} \times (\mathbf {C} \times \mathbf {A} )=\mathbf {0} }{\displaystyle \mathbf {A} \times (\mathbf {B} \times \mathbf {C} )+\mathbf {C} \times (\mathbf {A} \times \mathbf {B} )+\mathbf {B} \times (\mathbf {C} \times \mathbf {A} )=\mathbf {0} }.
Binet-Cauchy identity: {\displaystyle \mathbf {\left(A\times B\right)\cdot } \left(\mathbf {C} \times \mathbf {D} \right)=\left(\mathbf {A} \cdot \mathbf {C} \right)\left(\mathbf {B} \cdot \mathbf {D} \right)-\left(\mathbf {B} \cdot \mathbf {C} \right)\left(\mathbf {A} \cdot \mathbf {D} \right)} \mathbf{\left(A\times B\right)\cdot}\left(\mathbf{C}\times\mathbf{D}\right)=\left(\mathbf{A}\cdot\mathbf{C}\right)\left(\mathbf{B}\cdot\mathbf{D}\right)-\left(\mathbf{B}\cdot\mathbf{C}\right)\left(\mathbf{A}\cdot\mathbf{D}\right) .
Lagrange's identity: {\displaystyle |\mathbf {A} \times \mathbf {B} |^{2}=(\mathbf {A} \cdot \mathbf {A} )(\mathbf {B} \cdot \mathbf {B} )-(\mathbf {A} \cdot \mathbf {B} )^{2}}{\displaystyle |\mathbf {A} \times \mathbf {B} |^{2}=(\mathbf {A} \cdot \mathbf {A} )(\mathbf {B} \cdot \mathbf {B} )-(\mathbf {A} \cdot \mathbf {B} )^{2}}.
Vector quadruple product:[4][5] {\displaystyle (\mathbf {A} \times \mathbf {B} )\times (\mathbf {C} \times \mathbf {D} )\ =\ |\mathbf {A} \,\mathbf {B} \,\mathbf {D} |\,\mathbf {C} \,-\,|\mathbf {A} \,\mathbf {B} \,\mathbf {C} |\,\mathbf {D} \ =\ |\mathbf {A} \,\mathbf {C} \,\mathbf {D} |\,\mathbf {B} \,-\,|\mathbf {B} \,\mathbf {C} \,\mathbf {D} |\,\mathbf {A} }{\displaystyle (\mathbf {A} \times \mathbf {B} )\times (\mathbf {C} \times \mathbf {D} )\ =\ |\mathbf {A} \,\mathbf {B} \,\mathbf {D} |\,\mathbf {C} \,-\,|\mathbf {A} \,\mathbf {B} \,\mathbf {C} |\,\mathbf {D} \ =\ |\mathbf {A} \,\mathbf {C} \,\mathbf {D} |\,\mathbf {B} \,-\,|\mathbf {B} \,\mathbf {C} \,\mathbf {D} |\,\mathbf {A} }.

"""

assert A.cross(A) == 0
assert A.dot(A.cross(B)) == 0


""" Del Tests
All the following from:
    
    https://en.wikipedia.org/wiki/Del
"""

"""Test these First Dirivatives

\nabla (fg)&=f\nabla g+g\nabla f
\nabla ({\vec {u}}\cdot {\vec {v}})&={\vec {u}}\times (\nabla \times {\vec {v}})+{\vec {v}}\times (\nabla \times {\vec {u}})+({\vec {u}}\cdot \nabla ){\vec {v}}+({\vec {v}}\cdot \nabla ){\vec {u}}
\nabla \cdot (f{\vec {v}})&=f(\nabla \cdot {\vec {v}})+{\vec {v}}\cdot (\nabla f)
\nabla \cdot ({\vec {u}}\times {\vec {v}})&={\vec {v}}\cdot (\nabla \times {\vec {u}})-{\vec {u}}\cdot (\nabla \times {\vec {v}})
\nabla \times (f{\vec {v}})&=(\nabla f)\times {\vec {v}}+f(\nabla \times {\vec {v}})\nabla \times ({\vec {u}}\times {\vec {v}})&={\vec {u}}\,(\nabla \cdot {\vec {v}})-{\vec {v}}\,(\nabla \cdot {\vec {u}})+({\vec {v}}\cdot \nabla )\,{\vec {u}}-({\vec {u}}\cdot \nabla )\,{\vec {v}}\end{aligned}}}

"""

"""Test Second dirvatives:"""

assert del.cross(grad(f))== 0  #curl of a gradient =0

assert del.dot(del.cross(F)) == 0  # divergence of a curl =0

assert del.dot(del*f) == Laplacian(f)  #divergence of a gradient = laplacian

""" 
\operatorname {div} (\operatorname {grad} f)&=\nabla \cdot (\nabla f) = \nabla ^{2}f = Laplacian
   
\operatorname {grad} (\operatorname {div} {\vec {v}})&=\nabla (\nabla \cdot {\vec {v}})
\operatorname {curl} (\operatorname {curl} {\vec {v}})&=\nabla \times (\nabla \times {\vec {v}})

The 3 above vector second derivatives are related by the equation:

\nabla \times \left(\nabla \times {\vec {v}}\right)=\nabla (\nabla \cdot {\vec {v}})-\nabla ^{2}{\vec {v}}}{\displaystyle \nabla \times \left(\nabla \times {\vec {v}}\right)=\nabla (\nabla \cdot {\vec {v}})-\nabla ^{2}{\vec {v}}}

    
    Also, Verify the laplacian of a vector function = a vector function
"""

"""Verify a tensor product,
 if the functions are well-behaved:
{\displaystyle \nabla (\nabla \cdot {\vec {v}})=\nabla \cdot (\nabla \otimes {\vec {v}})}{\displaystyle \nabla (\nabla \cdot {\vec {v}})=\nabla \cdot (\nabla \otimes {\vec {v}})}
"""

"""Verify Simplification doesn't treat del as a vector, but rather as a differential vector operator
Most of the above vector properties (except for those that rely explicitly on del's differential properties—for example, the product rule) rely only on symbol rearrangement, and must necessarily hold if the del symbol is replaced by any other vector. This is part of the value to be gained in notationally representing this operator as a vector.

Though one can often replace del with a vector and obtain a vector identity, making those identities mnemonic, the reverse is not necessarily reliable, because del does not commute in general.

A counterexample that relies on del's failure to commute:

{\displaystyle {\begin{aligned}({\vec {u}}\cdot {\vec {v}})f&\equiv ({\vec {v}}\cdot {\vec {u}})f\\(\nabla \cdot {\vec {v}})f&=\left({\frac {\partial v_{x}}{\partial x}}+{\frac {\partial v_{y}}{\partial y}}+{\frac {\partial v_{z}}{\partial z}}\right)f={\frac {\partial v_{x}}{\partial x}}f+{\frac {\partial v_{y}}{\partial y}}f+{\frac {\partial v_{z}}{\partial z}}f\\({\vec {v}}\cdot \nabla )f&=\left(v_{x}{\frac {\partial }{\partial x}}+v_{y}{\frac {\partial }{\partial y}}+v_{z}{\frac {\partial }{\partial z}}\right)f=v_{x}{\frac {\partial f}{\partial x}}+v_{y}{\frac {\partial f}{\partial y}}+v_{z}{\frac {\partial f}{\partial z}}\\\Rightarrow (\nabla \cdot {\vec {v}})f&\neq ({\vec {v}}\cdot \nabla )f\\\end{aligned}}}{\displaystyle {\begin{aligned}({\vec {u}}\cdot {\vec {v}})f&\equiv ({\vec {v}}\cdot {\vec {u}})f\\(\nabla \cdot {\vec {v}})f&=\left({\frac {\partial v_{x}}{\partial x}}+{\frac {\partial v_{y}}{\partial y}}+{\frac {\partial v_{z}}{\partial z}}\right)f={\frac {\partial v_{x}}{\partial x}}f+{\frac {\partial v_{y}}{\partial y}}f+{\frac {\partial v_{z}}{\partial z}}f\\({\vec {v}}\cdot \nabla )f&=\left(v_{x}{\frac {\partial }{\partial x}}+v_{y}{\frac {\partial }{\partial y}}+v_{z}{\frac {\partial }{\partial z}}\right)f=v_{x}{\frac {\partial f}{\partial x}}+v_{y}{\frac {\partial f}{\partial y}}+v_{z}{\frac {\partial f}{\partial z}}\\\Rightarrow (\nabla \cdot {\vec {v}})f&\neq ({\vec {v}}\cdot \nabla )f\\\end{aligned}}}
A counterexample that relies on del's differential properties:

{\displaystyle {\begin{aligned}(\nabla x)\times (\nabla y)&=\left({\vec {e}}_{x}{\frac {\partial x}{\partial x}}+{\vec {e}}_{y}{\frac {\partial x}{\partial y}}+{\vec {e}}_{z}{\frac {\partial x}{\partial z}}\right)\times \left({\vec {e}}_{x}{\frac {\partial y}{\partial x}}+{\vec {e}}_{y}{\frac {\partial y}{\partial y}}+{\vec {e}}_{z}{\frac {\partial y}{\partial z}}\right)\\&=({\vec {e}}_{x}\cdot 1+{\vec {e}}_{y}\cdot 0+{\vec {e}}_{z}\cdot 0)\times ({\vec {e}}_{x}\cdot 0+{\vec {e}}_{y}\cdot 1+{\vec {e}}_{z}\cdot 0)\\&={\vec {e}}_{x}\times {\vec {e}}_{y}\\&={\vec {e}}_{z}\\({\vec {u}}x)\times ({\vec {u}}y)&=xy({\vec {u}}\times {\vec {u}})\\&=xy{\vec {0}}\\&={\vec {0}}\end{aligned}}}{\displaystyle {\begin{aligned}(\nabla x)\times (\nabla y)&=\left({\vec {e}}_{x}{\frac {\partial x}{\partial x}}+{\vec {e}}_{y}{\frac {\partial x}{\partial y}}+{\vec {e}}_{z}{\frac {\partial x}{\partial z}}\right)\times \left({\vec {e}}_{x}{\frac {\partial y}{\partial x}}+{\vec {e}}_{y}{\frac {\partial y}{\partial y}}+{\vec {e}}_{z}{\frac {\partial y}{\partial z}}\right)\\&=({\vec {e}}_{x}\cdot 1+{\vec {e}}_{y}\cdot 0+{\vec {e}}_{z}\cdot 0)\times ({\vec {e}}_{x}\cdot 0+{\vec {e}}_{y}\cdot 1+{\vec {e}}_{z}\cdot 0)\\&={\vec {e}}_{x}\times {\vec {e}}_{y}\\&={\vec {e}}_{z}\\({\vec {u}}x)\times ({\vec {u}}y)&=xy({\vec {u}}\times {\vec {u}})\\&=xy{\vec {0}}\\&={\vec {0}}\end{aligned}}}
Central to these distinctions is the fact that del is not simply a vector; it is a vector operator. Whereas a vector is an object with both a magnitude and direction, del has neither a magnitude nor a direction until it operates on a function.

For that reason, identities involving del must be derived with care, using both vector identities and differentiation identities such as the product rule.
"""

