# import sympy as sp
from sympy import (
    sympify, S, Basic, Expr, Derivative, Abs, Mul, Add, Pow, 
    Symbol, Number, log
)
from sympy.core.compatibility import (
    string_types, default_sort_key, with_metaclass, reduce
)
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.evalf import EvalfMixin
from sympy.core.operations import AssocOp
from sympy.core.singleton import Singleton
from sympy.strategies import flatten

from sympy.vector import (
    Vector, 
    divergence, curl
)

class VectorExpr(Expr):
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

    is_Vector_Scalar = False
    
    def __new__(cls, *args, **kwargs):
        args = map(sympify, args)
        return Basic.__new__(cls, *args, **kwargs)
    
    def equals(self, other):
        return self == other
    
    def __hash__(self):
        # hash cannot be cached using cache_it because infinite recurrence
        # occurs as hash is needed for setting cache dictionary keys
        h = self._mhash
        if h is None:
            h = hash((type(self).__name__,) + self._hashable_content())
            self._mhash = h
        return h
    
    def __eq__(self, other):
        if not isinstance(other, VectorExpr):
            return False
        if type(self) != type(other):
            return False
        a, b = self._hashable_content(), other._hashable_content()
        if a != b:
            return False
        return True
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return VecAdd(self, other, check=True)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return VecAdd(other, self, check=True)
    
    def __div__(self, other):
        return self * other**S.NegativeOne

    def __rdiv__(self, other):
        raise NotImplementedError()

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    
    def __neg__(self):
        return VecMul(S.NegativeOne, self, check=True)
    
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
        if other == S.One:
            return self
        return VecPow(self, other).doit(deep=False)

    def __rpow__(self, other):
        raise NotImplementedError("Vector Power not defined")
        
    def __xor__(self, other):
        return VecCross(self, other)

    # need the following twos for auto-diff evaluation
    def _eval_derivative_n_times(self, s, n):
        return Basic._eval_derivative_n_times(self, s, n)
    
    def _accept_eval_derivative(self, s):
        return s._visit_eval_derivative_scalar(self)
    
    @property
    def free_symbols(self):
        return set((self,))
    
    def dot(self, other):
        return VecDot(self, other)
    
    def cross(self, other):
        return VecCross(self, other)

    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", True)
        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d = Derivative(self, *symbols)
        if not isinstance(d, Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        d2 = Derivative(self, *symbols, **assumptions)
        return D(d2)

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
            return Abs(self)
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
        return Abs(self)
    
    @property
    def mag(self):
        return self.magnitude()
    
    # TODO: 
    # the gradient of a vector is a matrix!!!
    # the gradient of a scalar is a vector!!!
    def gradient(self):
        raise NotImplementedError("Gradient not implemented")
    
    @property
    def grad(self):
        return self.gradient()
    
    def divergence(self):
        return VecDot(Nabla(), self)
    
    @property
    def div(self):
        return self.divergence()
    
    def curl(self):
        return VecCross(Nabla(), self)
   
    @property
    def cu(self):
        return self.curl()







def get_postprocessor(cls):
    def _postprocessor(expr):
        vec_class = {Mul: VecMul, Add: VecAdd}[cls]
        nonvectors = []
        vectors = []
        for term in expr.args:
            if isinstance(term, VectorExpr):
                vectors.append(term)
            else:
                nonvectors.append(term)
        
        if not vectors:
            return cls._from_args(nonvectors)
        
        if nonvectors:
            if cls == Mul:
                for i in range(len(vectors)):
                    if not vectors[i].is_VectorExpr:
                        # If one of the matrices explicit, absorb the scalar into it
                        # (doit will combine all explicit matrices into one, so it
                        # doesn't matter which)
                        vectors[i] = vectors[i].__mul__(cls._from_args(nonvectors))
                        nonvectors = []
                        break

            else:
                # Maintain the ability to create Add(scalar, matrix) without
                # raising an exception. That way different algorithms can
                # replace matrix expressions with non-commutative symbols to
                # manipulate them like non-commutative scalars.
                return cls._from_args(nonvectors + [vec_class(*vectors).doit(deep=False)])
        
        
        if vec_class == VecAdd:
            asd = vec_class(*vectors).doit(deep=False)
            return asd
        asd = vec_class(cls._from_args(nonvectors), *vectors).doit(deep=False)
        return asd
    return _postprocessor


Basic._constructor_postprocessor_mapping[VectorExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
}








class VectorSymbol(VectorExpr):
    is_symbol = True
    is_Vector = True

    def __new__(cls, name, **kwargs):
        if not isinstance(name, (string_types, Symbol, VectorSymbol)):
            raise TypeError("'name' must be a string or a Symbol or a VectorSymbol")
        if isinstance(name, string_types):
            name = Symbol(name)
        obj = Basic.__new__(cls, name)

        # Attributes for the latex printer.
        # if _vec_symbol="" or _unit_vec_symbol="", the printer will use its
        # default symbols to print vectors and unit vectors
        _vec_symbol = kwargs.get('_vec_symbol', "")
        _unit_vec_symbol = kwargs.get('_unit_vec_symbol', "")
        _bold = kwargs.get('_bold', False)
        _italic = kwargs.get('_italic', False)
        
        obj._vec_symbol = _vec_symbol
        obj._unit_vec_symbol = _unit_vec_symbol
        obj._bold = _bold
        obj._italic = _italic

        return obj

    def doit(self, **kwargs):
        return self

    @property
    def name(self):
        return self.args[0].name
    
# class VectorZero(with_metaclass(Singleton, VectorSymbol)):
class VectorZero(VectorSymbol):
    is_ZeroVector = True
    
    def __new__(cls, **kwargs):
        return super().__new__(cls, "0", **kwargs)
    
    def magnitude(self):
        return S.Zero
    
    def normalize(self):
        return self

    def _eval_derivative(self, s):
        return self

class VectorOne(VectorSymbol):
    def __new__(cls, **kwargs):
        return super().__new__(cls, "1", **kwargs)
    
    def _eval_derivative(self, s):
        return VectorZero()

VectorSymbol.zero = VectorZero()
VectorSymbol.one = VectorOne()

class Nabla(VectorSymbol):
    def __new__(cls, **kwargs):
        return super().__new__(cls, r"\nabla", **kwargs)

    def magnitude(self):
        raise TypeError("nabla operator doesn't have magnitude.")

    def normalize(self):
        raise TypeError("nabla operator cannot be normalized.")

    def _eval_derivative(self, s):
        raise NotImplementedError("Differentiation of nabla operator not implemented.")

class Normalize(VectorExpr):
    is_Normalized = True
    is_Vector = True

    def __new__(cls, v):
        v = sympify(v)
        if isinstance(v, Normalize):
            return v
        if not isinstance(v, (VectorExpr, Vector)):
            raise TypeError("Can only normalize instances of VectorExpr or Vector.")
        if not v.is_Vector:
            raise TypeError("VectorExpr must be a vector, not a scalar.")
        if isinstance(v, (Nabla, VectorZero)):
            return v.norm
        return Basic.__new__(cls, v)
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]

        if issubclass(args[0].func, Vector):
            # here args[0] could be an instance of VectorAdd or VecMul or Vector
            return args[0].normalize()

        return VecMul(args[0], VecPow(Magnitude(args[0]), -1))
    
class Magnitude(VectorExpr):
    is_Vector = False
    is_Norm = True
    is_positive = True

    is_Vector_Scalar = True

    def __new__(cls, v):
        v = sympify(v)
        if isinstance(v, Magnitude):
            return v
        if not isinstance(v, (VectorExpr, Vector)):
            # for example, v is a number, or symbol
            return Abs(v)
        if not v.is_Vector:
            # for example, dot-product of two VectorSymbol
            return Abs(v)
        if isinstance(v, Nabla):
            return v.mag
        return Basic.__new__(cls, v)
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]

        if isinstance(args[0], Vector):
            return args[0].magnitude()
        return self.func(*args)

class D(VectorExpr):
    """ Represent an unevaluated derivative.

    This class is necessary because even if it's possible to set the attribute
    is_Vector=True to an unevaluated Derivative (as of Sympy version 1.5.1, 
    which may be a bug), it is not possible to set this attribute to objects of 
    type  Add (read-only property). If v1 and v2 are two vector expression 
    with is_Vector=True:
        d = Derivative(v1 + v2, x).doit()
    will return Add(Derivative(v1, x), Derivative(v2, x)), with 
    is_Vector=False, which is wrong, preventing from further vector expression 
    operations. For example, VecCross(v1, d) will fail because 
    d.is_Vector=False.

    Solution! Wrap unevaluated derivatives with this class.
    """
    is_Vector = False

    def __new__(cls, d):
        d = sympify(d)
        if not isinstance(d, Derivative):
            return d
        
        obj = Basic.__new__(cls, d)
        obj.is_Vector = d.expr.is_Vector
        obj.is_Vector_Scalar = not d.expr.is_Vector
        return obj
    
    def _eval_derivative(self, s):
        d = self.args[0]
        variable_count = list(d.variable_count) + [(s, 1)]
        return D(Derivative(d.expr, *variable_count))

    def _eval_derivative_n_times(self, s, n):
        d = self.args[0]
        variable_count = list(d.variable_count) + [(s, n)]
        d = Derivative(d.expr, *variable_count)
        if len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        return D(d)

class VecDot(VectorExpr):
    is_Dot = True
    is_Vector_Scalar = True
    
    def __new__(cls, expr1, expr2, **kwargs):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)

        check = lambda x: isinstance(x, (VectorExpr, Vector))
        if not (check(expr1) and check(expr2) and \
            expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the dot-operator must be vectors")
        if expr1 == VectorZero() or expr2 == VectorZero():
            return S.Zero
    
        obj = Expr.__new__(cls, expr1, expr2)
        return obj
    
    @property
    def reverse(self):
        # take into account the fact that arg[0] and arg[1] could be mixed 
        # instances of Vector and VectorExpr
        return self.func(self.args[1], self.args[0])

    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Vector):
            return self.args[0].dot(self.args[1])
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Nabla):
            return divergence(self.args[0])
        if isinstance(self.args[1], Vector) and \
            isinstance(self.args[0], Nabla):
            return divergence(self.args[1])
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        if args[0] == args[1]:
            return VecPow(args[0].mag, 2)
        return self.func(*args)
    
    def _eval_derivative(self, s):
        return _diff_cross_dot(self, s)
    
    def _eval_derivative_n_times(self, s, n):
        result = _diff_cross_dot(self, s)
        while n > 1:
            result = result.diff(s)
            n -= 1
        return result


"""
TODO:
In a perfect world I would be happy to override _eval_derivative_n_times
and _eval_derivative. However, by doing that I end up with:
"RecursionError: maximum recursion depth exceeded while calling a Python object"
Instead, I have to override the method `diff` on the classes VecCross, VecDot, 
VecAdd, VecMul, VecPow, D... This however introduces repetition in the code.
"""

class VecCross(VectorExpr):
    is_Cross = True
    is_Vector = True
    is_commutative = False
    
    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)

        check = lambda x: isinstance(x, (VectorExpr, Vector))
        if not (check(expr1) and check(expr2) and \
                expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the cross-operator must be vectors")
        if expr1 == VectorZero() or expr2 == VectorZero():
            return VectorZero()
        if (isinstance(expr1, Vector) and isinstance(expr2, Vector) and
            expr1 == expr2):
            # TODO: At this point I'm dealing with unevaluated cross product.
            # is it better to return VectorZero()?
            return expr1.zero
        if expr1 == expr2:
            return VectorZero()

        obj = Expr.__new__(cls, expr1, expr2)
        return obj
    
    @property
    def reverse(self):
        # take into account the fact that arg[0] and arg[1] could be mixed 
        # instances of Vector and VectorExpr
        return -self.func(self.args[1], self.args[0])
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Vector):
            return self.args[0].cross(self.args[1])
        if isinstance(self.args[0], Vector) and \
            isinstance(self.args[1], Nabla):
            return -curl(self.args[0])
        if isinstance(self.args[1], Vector) and \
            isinstance(self.args[0], Nabla):
            return curl(self.args[1])
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        return self.func(*args)
    
    def _eval_derivative(self, s):
        return _diff_cross_dot(self, s)
    
    def _eval_derivative_n_times(self, s, n):
        result = _diff_cross_dot(self, s)
        while n > 1:
            result = result.diff(s)
            n -= 1
        return result

"""
TODO: probably best to create a base class for VecDot, VecCross and put this 
there, together with _eval_derivative(), _eval_derivative_n_times() and the 
property reverse, once the problem with is_Vector has been fixed (see 
Magnitude.diff comment to understand).
"""
def _diff_cross_dot(expr, s):
    if not isinstance(expr, (VecCross, VecDot)):
        raise TypeError("_diff_cross_dot method only works with instances of VecCross and VecDot.")

    expr0 = expr.args[0].diff(s)
    expr1 = expr.args[1].diff(s)
    t1 = expr.func(expr0, expr.args[1])
    t2 = expr.func(expr.args[0], expr1)
    return t1 + t2

# used for debugging the expressions trees
_spaces = []
_token = "    "
def debug(*args):
    print("".join(_spaces), *args)


def _check_if_scalar(expr):
    """
    Ordinary symbols, numbers, ... are scalars. On the other hand, VectorExpr
    can contains both vectors and scalars. Because instances of the class Vector
    are also scalars, VectorExpr exposes the property is_Vector_Scalar to
    differentiate between a vector quantity and a scalar quantity (still related
    to the vector, for example the magnitude)
    """
    if hasattr(expr, "is_Vector_Scalar"):
        return expr.is_Vector_Scalar
    return expr.is_scalar


# class VecAdd(VectorExpr):
class VecAdd(VectorExpr, Add):
    is_VecAdd = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        evaluate = kwargs.get('evaluate', True)
        
        if not args:
            return VectorZero()

        args = list(map(sympify, args))
        if len(args) == 1:
            # doesn't make any sense to have 1 argument in VecAdd if 
            # evaluate=True
            return args[0]

        if evaluate:
            # remove instances of S.Zero and VectorZero
            args = [a for a in args 
                if not (isinstance(a, VectorZero) or (a == S.Zero))]
            
            if len(args) == 0:
                return VectorZero()
            elif len(args) == 1:
                # doesn't make any sense to have 1 argument in VecAdd if 
                # evaluate=True
                return args[0]

            # Create a new object of type VecAdd. By calling  Basic.__new__ we  
            # avoid infinite loop.
            obj = Basic.__new__(cls, *args)            
            # flatten the expression. For example: 
            # VecAdd(v1, VecAdd(v2, v1))
            # becomes: VecAdd(v1, v2, v1)
            obj = flatten(obj)
            args = _sanitize_args(obj.args)
            c_part, nc_part, order_symbols = Add.flatten(args)
            args = c_part + nc_part
            args = _sanitize_args(args)
            is_commutative = not nc_part

            if len(args) == 1:
                return args[0]
        else:
            # TODO: addition is not commutative if an argument is not commutative!!!
            is_commutative = True

        # # TODO: addition is not commutative if an argument is not commutative!!!
        # obj.is_commutative = True
        obj = cls._from_args(args, is_commutative)

        # TODO: do I need this???
        # if there is only one argument and it was 1, then obj is the number
        # one, whose property is_Vector is hidden, therefore return 1 immediatly
        if isinstance(obj, Number):
            return VectorZero()
        
        # are there any scalars or vectors?
        are_vectors = any([a.is_Vector for a in args])
        # as of Sympy version 1.5.1, an instance of Vector has is_scalar=True
        are_scalars = any([_check_if_scalar(a) for a in args 
            if not isinstance(a, Vector)])
        
        # addition of mixed scalars and vectors is not supported.
        # If there are scalars in the addition, either all arguments
        # are scalars (hence not a vector expression) or there are
        # mixed arguments, hence throw an error
        obj.is_Vector = not are_scalars
        obj.is_Vector_Scalar = not obj.is_Vector

        if check:
            if all(not isinstance(i, (VectorExpr, Vector)) for i in args):
                return Add.fromiter(args)
            if (are_vectors and are_scalars):
                raise TypeError("Mix of Vector and Scalar symbols")
        return obj
    
    def _eval_derivative(self, s):
        return self.func(*[a.diff(s) for a in self.args])

    def doit(self, **kwargs):
        # Need to override this method in order to apply the rules defined
        # below. Adapted from MatAdd.
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        return canonicalize(VecAdd(*args))


from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
    exhaust, do_one, glom)
from sympy.utilities import default_sort_key, sift
from operator import add

def merge_explicit(vecadd):
    """ Merge Vector arguments
    Example
    ========
    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> v1 = VectorSymbol("v1")
    >>> v2 = VectorSymbol("v2")
    >>> C = CoordSys3D("C")
    >>> vn1 = 2 * C.i + 3 * C.j + 4 * C.k
    >>> vn2 = x * C.i + y * C.j + x * y * C.k
    >>> expr = v1 + v2 + vn1 + vn2
    >>> pprint(expr)
    (2*C.i + 3*C.j + 4*C.k) + (x*C.i + y*C.j + x*y*C.k) + v1 + v2
    >>> pprint(merge_explicit(expr))
    ((2 + x)*C.i + (3 + y)*C.j + (4 + x*y)*C.k) + v1 + v2
    """
    # Adapted from sympy.matrices.expressions.matadd.py
    groups = sift(vecadd.args, lambda arg: isinstance(arg, (Vector)))
    if len(groups[True]) > 1:
        return VecAdd(*(groups[False] + [reduce(add, groups[True])]))
    else:
        return vecadd

rules = (
    rm_id(lambda x: x == 0 or isinstance(x, VectorZero)),
    unpack,
    flatten,
    merge_explicit,
    sort(default_sort_key)
)

canonicalize = exhaust(condition(lambda x: isinstance(x, VecAdd),
                                 do_one(*rules)))


# class VecMul(VectorExpr):
class VecMul(VectorExpr, Mul):
    is_VecMul = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        evaluate = kwargs.get('evaluate', True)

        if not args:
            return VectorOne()
        
        args = list(map(sympify, args))
        if len(args) == 1:
            # doesn't make any sense to have 1 argument in VecAdd if 
            # evaluate=True
            return args[0]
        
        vectors = [a.is_Vector for a in args]
        if vectors.count(True) > 1:
            raise TypeError("Multiplication of vector quantities not supported")

        if evaluate:
            # remove instances of S.One
            args = [a for a in args if a is not cls.identity]
            if len(args) == 0:
                return S.One
            if len(args) == 1:
                return args[0]

            # check if any vector is present
            args_is_vector = [a.is_Vector for a in args]
            if any([a == S.Zero for a in args]):
                if any(args_is_vector):
                    return VectorZero()
                return S.Zero
            if any([isinstance(a, VectorZero) for a in args]):
                return VectorZero()

            # Create a new object of type VecAdd. By calling  Basic.__new__ we  
            # avoid infinite loop.
            obj = Basic.__new__(cls, *args)

            # flatten the expression tree
            obj = flatten(obj)
            args = _sanitize_args(obj.args)
            # flatten the arguments
            c_part, nc_part, order_symbols = Mul.flatten(args)
            args = c_part + nc_part
            args = _sanitize_args(args)

            # Mul.flatten may group arguments together. Think for example to:
            # VecMul(v1.mag, v1.mag) -> VecPow(v1.mag, 2)
            # Need to perform this check again.
            if len(args) == 1:
                # the object coming out from flatten, even if it is VecAdd,
                # it always have the property is_Vector=False... Need to enforce
                # the proper value by recreating the object.
                return args[0].func(*args[0].args)
                # return args[0]

        is_commutative = not any([not a.is_commutative for a in args])
        obj = cls._from_args(args, is_commutative)

        vectors = [a.is_Vector for a in args]
        cvectors = vectors.count(True)
        scalars = [_check_if_scalar(a) for a in args]
        cscalars = scalars.count(True)

        # multiplication of vectors is not supported. If there are multiple
        # vectors, an error will be thrown.
        # If all arguments are scalars, the resulting expression wil not
        # be a vector
        obj.is_Vector = cscalars != len(args)
        obj.is_Vector_Scalar = not obj.is_Vector
        
        if check:
            if cvectors > 1:
                raise TypeError("Multiplication of vector quantities not supported")
            if all(not isinstance(a, VectorExpr) for a in args):
                return Mul.fromiter(args)
        return obj

    def _eval_derivative(self, s):
        # adapted from Mul._eval_derivative
        args = list(self.args)
        terms = []
        for i in range(len(args)):
            d = args[i].diff(s)
            if d:
                # Note: reduce is used in step of Mul as Mul is unable to
                # handle subtypes and operation priority:
                terms.append(reduce(lambda x, y: x*y, 
                    (args[:i] + [d] + args[i + 1:]), S.One))
        return VecAdd.fromiter(terms)
    
    def _eval_derivative_n_times(self, s, n):
        # adapted from Mul._eval_derivative_n_times
        from sympy import Integer
        from sympy.ntheory.multinomial import multinomial_coefficients_iterator

        args = self.args
        m = len(args)
        if isinstance(n, (int, Integer)):
            # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
            terms = []
            for kvals, c in multinomial_coefficients_iterator(m, n):
                p = VecMul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            return VecAdd(*terms)
        raise TypeError("Derivative order must be integer")

class VecPow(VectorExpr):
    def __new__(cls, base, exp):
        base = S(base)
        exp =  S(exp)
        if base.is_Vector:
            raise TypeError("Vector power not available")
        if not isinstance(base, VectorExpr):
            if isinstance(exp, VectorExpr) and exp.is_Vector:
                raise TypeError("Vector exponent not available")
            return Pow(base, exp)
        
        if exp == S.One:
            return base

        obj = Basic.__new__(cls, base, exp)
        # at this point I know the base is an instance of VectorExpr with
        # is_Vector=False, hence is_scalar = True
        obj.is_Vector = False
        obj.is_Vector_Scalar = True
        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]
    
    def _eval_derivative(self, s):
        # adapted from Pow._eval_derivative
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        return self * (dexp * log(self.base) + dbase * self.exp / self.base)

def _sanitize_args(args):
    """ If an instance of Add, Mul, Pow contains VectorExpr, substitute
    them with VecAdd, VecMul, VecPow respectively.

    flatten() may remove instances of VecMul. For example:
    v1 - v2 = VecAdd(v1, VecMul(-1, v2)) -> (after flatten) ->
    = VecAdd(v1, Mul(-1, v2))
    This is obviously wrong because even if Mul has the property is_Vector,
    it is not updated with the value of VecMul.
    Solution: instead of rewriting flatten(), post process the result
    """
    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, Add):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(Add, VecAdd)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
        if isinstance(a, Mul):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(Mul, VecMul)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
        if isinstance(a, Pow):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(Pow, VecPow)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
    return args
    


# TODO:
# *. find out why class VecAdd(Add) returns Add objects, not VecAdd
# *. printing of (v1 + (v1 ^ one) * (v1 & v2))