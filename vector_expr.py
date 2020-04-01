import sympy as sp
from sympy.core.compatibility import string_types, default_sort_key, with_metaclass
from sympy.core.decorators import call_highest_priority
from sympy.core.singleton import Singleton
from sympy.vector import (
    Vector, 
    divergence, curl
)

class VectorExpr(sp.Expr):
    is_Vector = False
    is_VectorExpr = True
    is_Cross = False
    is_Dot = False
    is_Grad = False
    is_Div = False
    is_Curl = False
    is_Normalized = False
    is_ZeroVector = False
    
    is_number = False
    is_symbol = False
    is_scalar = False
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        args = map(sp.sympify, args)
        return sp.Basic.__new__(cls, *args, **kwargs)
    
    def equals(self, other):
        # TODO: develop a better way
        s1 = sp.srepr(self)
        s2 = sp.srepr(other)
        return s1 == s2
    
    def __hash__(self):
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h
    
    def _hashable_content(self):
        """Return a tuple of information about self that can be used to
        compute the hash. If a class defines additional attributes,
        like ``name`` in Symbol, then this method should be updated
        accordingly to return such relevant attributes.
        Defining more than _hashable_content is necessary if __eq__ has
        been defined by a class. See note about this in Basic.__eq__."""
        return self._args
    
    def __eq__(self, other):
        if not isinstance(other, VectorExpr):
            return False
        return self.equals(other)
    
    def __add__(self, other):
        return VecAdd(self, other, check=True)
    
    def __radd__(self, other):
        return VecAdd(other, self, check=True)
    
    def __div__(self, other):
        return self * other**sp.S.NegativeOne

    def __rdiv__(self, other):
        raise NotImplementedError()
    
    def __neg__(self):
        return VecMul(sp.S.NegativeOne, self, check=True)
    
    def __sub__(self, other):
        return VecAdd(self, -other, check=True)

    def __rsub__(self, other):
        return VecAdd(other, -self, check=True)
    
    def __and__(self, other):
        return VecDot(self, other)
    
    def __mul__(self, other):
        return VecMul(self, other, check=True)
    
    def __rmul__(self, other):
        return VecMul(self, other, check=True)

    def __pow__(self, other):
        if other == sp.S.One:
            return self
        return VecPow(self, other).doit(deep=False)

    def __rpow__(self, other):
        raise NotImplementedError("Vector Power not defined")
        
    def __xor__(self, other):
        return VecCross(self, other)
    
    @property
    def free_symbols(self):
        return set((self,))
    
    def dot(self, other):
        return self & other
    
    def cross(self, other):
        return self ^ other

    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        return self.func(*args)
    
    def normalize(self, **kwargs):
        evaluate = kwargs.get('evaluate', False)
        if isinstance(self, VectorZero):
            return self
        if isinstance(self, Normalize):
            return self
        if not self.is_Vector:
            return sp.Abs(self)
        n = Normalize(self)
        if evaluate:
            return n.doit()
        return n
    
    @property
    def norm(self):
        return self.normalize()

    def magnitude(self):
        if self.is_Vector:
            return Magnitude(self)
        return sp.Abs(self)
    
    @property
    def mag(self):
        return self.magnitude()
    
    # TODO: 
    # the gradient of a vector is a matrix!!!
    # the gradient of a scalar is a vector!!!
    @property
    def gradient(self):
        raise NotImplementedError("Gradient not implemented")
    
    grad = gradient
    
    @property
    def divergence(self):
        return Nabla() & self
    
    div = divergence
    
    def curl(self):
        return Nabla() ^ self
    
    
class VectorSymbol(VectorExpr):
    is_symbol = True
    is_Vector = True
    
    def __new__(cls, name):
        if isinstance(name, string_types):
            name = sp.Symbol(name)
        obj = sp.Basic.__new__(cls, name)
        return obj

    @property
    def name(self):
        return self.args[0].name
    
    def __call__(self, *args):
        raise TypeError("%s object is not callable" % self.__class__)
    
    def magnitude(self):
        return Magnitude(self)
    

class VectorZero(with_metaclass(Singleton, VectorSymbol)):
    is_ZeroVector = True
    
    def __new__(cls):
        name = sp.Symbol("0")
        obj = sp.Basic.__new__(cls, name)
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        return self

class VectorOne(with_metaclass(Singleton, VectorSymbol)):
    def __new__(cls):
        name = sp.Symbol("1")
        obj = sp.Basic.__new__(cls, name)
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        return self

VectorSymbol.zero = VectorZero()
VectorSymbol.one = VectorOne()


class Nabla(with_metaclass(Singleton, VectorSymbol)):
    def __new__(cls):
        name = sp.Symbol(r'\nabla')
        obj = sp.Basic.__new__(cls, name)
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        return self

class Normalize(VectorExpr):
    is_Normalized = True
    is_Vector = True
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        evaluate = kwargs.get('evaluate', True)
        if isinstance(self.args[0], Vector):
            if deep:
                return self.args[0].normalize()
            return VecMul(self.args[0], sp.Pow(self.args[0].magnitude(), -1))
        
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]

        if evaluate:
            return VecMul(self.args[0], VecPow(Magnitude(self.args[0]), -1))
        return self.func(*args)
    
class Magnitude(VectorExpr):
    is_Vector = False
    is_Norm = True
    is_scalar = True
    is_positive = True
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
    
        if isinstance(args[0], Vector):
            return args[0].magnitude()
        return self.func(*args)
    
class VecDot(VectorExpr):
    is_Dot = True
    is_scalar = True
    
    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)

        # TODO: better to look if expr has the attribute is_Vector and check its value
        if not (isinstance(expr1, (VectorExpr, Vector)) and \
            isinstance(expr2, (VectorExpr, Vector)) and \
            expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the dot-operator must be vectors")
        if expr1 == VectorZero() or expr2 == VectorZero():
            return 0
        
    
        obj = sp.Expr.__new__(cls, expr1, expr2)
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        # TODO: this is wrong! Better to look if expr has the attribute is_Vector and check its value
        if isinstance(self.args[0], (Vector, VectorMul, VectorAdd)) and \
            isinstance(self.args[1], (Vector, VectorMul, VectorAdd)):
            return self.args[0].dot(self.args[1])
        if isinstance(self.args[0], (Vector, VectorMul, VectorAdd)) and \
            isinstance(self.args[1], Nabla):
            return divergence(self.args[0])
        if isinstance(self.args[1], (Vector, VectorMul, VectorAdd)) and \
            isinstance(self.args[0], Nabla):
            return divergence(self.args[1])
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        return self.func(*args)

class VecCross(VectorExpr):
    is_Cross = True
    is_Vector = True
    is_commutative = False
    
    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)

        # TODO: better to look if expr has the attribute is_Vector and check its value
        if not (isinstance(expr1, (VectorExpr, Vector)) and \
            isinstance(expr2, (VectorExpr, Vector)) and \
            expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the cross-operator must be vectors")
        if expr1 == VectorZero() or expr2 == VectorZero():
            return VectorZero()
        if expr1 == expr2:
            return VectorZero()

        obj = sp.Expr.__new__(cls, expr1, expr2)
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Vector):
            return self.args[0].cross(self.args[1])
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Nabla):
            return curl(self.args[0])
        if isinstance(self.args[1], Vector) and \
            isinstance(self.args[0], Nabla):
            return -curl(self.args[1])
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        return self.func(*args)
    
    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        t1 = VecCross(sp.Derivative(self.args[0], *symbols, **assumptions), self.args[1])
        t2 = VecCross(self.args[0], sp.Derivative(self.args[1], *symbols, **assumptions))
        return t1 + t2
    
    
class VecAdd(VectorExpr, sp.Add):
    is_VecAdd = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        
        if not args:
            return VectorSymbol.zero

        args = list(map(sp.sympify, args))
            
        # need to perform the following steps because with __new__ we are
        # overriding the constructor of the base class AssocOp.
        
        # remove instances of sp.S.Zero and VectorZero
        args = [a for a in args if not (isinstance(a, VectorZero) or (a == sp.S.Zero))]
        
        # flatten the arguments
        # example: if we were to write 3 * v1 * 4 it would be interpreted as
        # VecMul(3, VecMul(v1, 4)). We want it like: VecMul(12, v1)
        c_part, nc_part, order_symbols = cls.flatten(args)
        args = c_part + nc_part
        is_commutative = not nc_part
        # have to do this step!!!
        sanitize_args(args)
        obj = cls._from_args(args, is_commutative)
        
        # if there is only one argument and it was 1, then obj is the number one,
        # whose property is_Vector is hidden, therefore return 1 immediatly
        if isinstance(obj, sp.Number):
            return VectorZero()
        
        # are there any scalars or vectors?
        are_vectors = any([a.is_Vector for a in args])
        are_scalars = any([a.is_scalar for a in args])
        
        # addition of mixed scalars and vectors is not supported.
        # If there are scalars in the addition, either all arguments
        # are scalars (hence not a vector expression) or there are
        # mixed arguments, hence throw an error
        obj.is_Vector = not are_scalars
        
        if check:
            if all(not isinstance(i, VectorExpr) for i in args):
                return sp.Add.fromiter(args)
            if (are_vectors and are_scalars):
                raise TypeError("Mix of Vector and Scalar symbols")

        return obj


class VecMul(VectorExpr, sp.Mul):
    is_VecMul = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        
        if not args:
            return VectorSymbol.one
        
        args = list(map(sp.sympify, args))
        
        # need to perform the following steps because with __new__ we are
        # overriding the constructor of the base class AssocOp.
        
        # remove instances of sp.S.One
        args = [a for a in args if a is not cls.identity]
        
        # flatten the arguments
        # example: if we were to write 3 * v1 * 4 it would be interpreted as
        # VecMul(3, VecMul(v1, 4)). We want it like: VecMul(12, v1)
        c_part, nc_part, order_symbols = cls.flatten(args)
        args = c_part + nc_part
        # have to do this step!!!
        sanitize_args(args)
        is_commutative = not nc_part
        obj = cls._from_args(args, is_commutative)
        
        # if one of the arguments was 0, then obj is the number zero,
        # whose property is_Vector is hidden, therefore return 0 immediatly
        if isinstance(obj, sp.Number):
            return obj
        
        vectors = [a.is_Vector for a in args]
        cvectors = vectors.count(True)
        scalars = [a.is_scalar for a in args]
        cscalars = scalars.count(True)
        # multiplication of vectors is not supported. If there are
        # multiple vectors, an error will be thrown.
        # If all arguments are scalars, the resulting expression
        # wil not be a vector
        obj.is_Vector = cscalars != len(args)
        
        # Deal with zeros: at this point there should only be ZeroVector,
        # because we dealt with scalar zero earlier with `if obj == sp.S.Zero`.
        # Also, we want to be able to substitute Sympy Vector instances into
        # the VectorExpr, which don't have the attribute is_ZeroVector.
        if any([not isinstance(a, Vector) and (a.is_zero or (a.is_Vector and a.is_ZeroVector))
                       for a in args]):
            if obj.is_Vector:
                return VectorZero()
            return sp.S.Zero
        
        if check:
            if cvectors > 1:
                raise TypeError("Multiplication of instances of VectorSymbol is not supported")
            if all(not isinstance(a, VectorExpr) for a in args):
                return sp.Mul.fromiter(args)
        return obj
    
class VecPow(VectorExpr):
    def __new__(cls, base, exp):
        base = sp.S(base)
        if base.is_Vector:
            raise TypeError("Vector power not available")
        exp =  sp.S(exp)
        return super(VecPow, cls).__new__(cls, base, exp)

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

def sanitize_args(args):
    # flatten may remove instances of VecMul. For example, 
    # v1 - v2 = VecAdd(v1, VecMul(-1, v2)) -> (after flatten) ->
    # = VecAdd(v1, sp.Mul(-1, v2))
    # This is obviously wrong because even if Mul has the property is_Vector,
    # it is not updated with the value of VecMul.
    # Solution: instead of rewriting flatten(), post process the result
    for i, a in enumerate(args):
        if isinstance(a, sp.Add):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(sp.Add, VecAdd)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol) for arg in a.args])
        if isinstance(a, sp.Mul):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(sp.Mul, VecMul)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol) for arg in a.args])
    return args