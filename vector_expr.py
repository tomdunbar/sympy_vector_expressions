import sympy as sp
from sympy.core import Add, Basic, sympify
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
        return self == other
    
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
        return self * other**sp.S.NegativeOne

    def __rdiv__(self, other):
        raise NotImplementedError()

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    
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

    # need the following twos for auto-diff evaluation
    def _eval_derivative_n_times(self, s, n):
        return sp.Basic._eval_derivative_n_times(self, s, n)
    
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
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        d2 = sp.Derivative(self, *symbols, **assumptions)
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
    
class VectorSymbol(VectorExpr):
    is_symbol = True
    is_Vector = True
    
    def __new__(cls, name, **kwargs):
        if not isinstance(name, (string_types, sp.Symbol, VectorSymbol)):
            raise TypeError("'name' must be a string or a Symbol or a VectorSymbol")
        if isinstance(name, string_types):
            name = sp.Symbol(name)
        obj = sp.Basic.__new__(cls, name)

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
        return sp.S.Zero
    
    def normalize(self):
        return self

    def _eval_derivative(self, s):
        return self

# class VectorOne(with_metaclass(Singleton, VectorSymbol)):
class VectorOne(VectorSymbol):
    def __new__(cls, **kwargs):
        return super().__new__(cls, "1", **kwargs)
    
    def _eval_derivative(self, s):
        return VectorZero()

VectorSymbol.zero = VectorZero()
VectorSymbol.one = VectorOne()


# class Nabla(with_metaclass(Singleton, VectorSymbol)):
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
        v = sp.sympify(v)
        if isinstance(v, Normalize):
            return v
        if not isinstance(v, (VectorExpr, Vector)):
            raise TypeError("Can only normalize instances of VectorExpr or Vector.")
        if not v.is_Vector:
            raise TypeError("VectorExpr must be a vector, not a scalar.")
        if isinstance(v, (Nabla, VectorZero)):
            return v.norm
        return sp.Basic.__new__(cls, v)
    
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
    is_scalar = True
    is_positive = True

    def __new__(cls, v):
        v = sp.sympify(v)
        if isinstance(v, Magnitude):
            return v
        if not isinstance(v, (VectorExpr, Vector)):
            # for example, v is a number, or symbol
            return sp.Abs(v)
        if not v.is_Vector:
            # for example, dot-product of two VectorSymbol
            return sp.Abs(v)
        if isinstance(v, Nabla):
            return v.mag
        return sp.Basic.__new__(cls, v)
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]

        if isinstance(args[0], Vector):
            return args[0].magnitude()
        return self.func(*args)
    
    def diff(self, *symbols, **assumptions):
        """
        TODO:

        Here comes the problem: I'm using is_scalar and is_Vector to distinguish
        between scalar and vector quantities (scalar related to vectors, ie 
        dot-product, magnitude, ...).
        However, is_scalar is also used by sp.Derivative: if the expression is 
        scalar and the differentiation symbol is different from the expression, 
        it will return 0, which is wrong in this case: for what we know, 
        magnitude is an expression containing also the differentiation symbol.
        Hence, need to override the diff method.
        
        See if it's possible to create a different property instead of using 
        is_scalar, for example is_Vector_scalar.
        Note: in VecAdd, VecMul, VecPow we will need to consider both the new 
        property as well as  is_scalar (used by standard symbolic expressions) 
        to decide if the resulting expression is a vector or a scalar.
        """
        evaluate = assumptions.get('evaluate', True)
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        if evaluate:
            if self.args[0] == VectorOne():
                return VectorZero()
            return D(d)
        return D(d)

class D(VectorExpr):
    """ Represent an unevaluated derivative.

    This class is necessary because even if it's possible to set the attribute
    is_Vector=True to an unevaluated Derivative (as of Sympy version 1.5.1, 
    which may be a bug), it is not possible to set this attribute to objects of 
    type  Add (read-only property). If v1 and v2 are two vector expression 
    with is_Vector=True:
        d = sp.Derivative(v1 + v2, x).doit()
    will return Add(sp.Derivative(v1, x), sp.Derivative(v2, x)), with 
    is_Vector=False, which is wrong, preventing from further vector expression 
    operations. For example, VecCross(v1, d) will fail because 
    d.is_Vector=False.

    Solution! Wrap unevaluated derivatives with this class.
    """
    is_Vector = False

    def __new__(cls, d):
        d = sp.sympify(d)
        if not isinstance(d, sp.Derivative):
            return d
        
        obj = sp.Basic.__new__(cls, d)
        obj.is_Vector = d.expr.is_Vector
        obj.is_scalar = d.expr.is_scalar
        return obj
    
    def diff(self, *symbols, **assumptions):
        # need to override this method because basic auto eval failed with:
        # RecursionError: maximum recursion depth exceeded while calling a Python object
        # in this example:
        # D((v1 + v2).diff(x, evaluate=False)).diff(x)

        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d0 = self.args[0]
        d1 = sp.Derivative(self.args[0].expr, *symbols)
        if not isinstance(d1, sp.Derivative):
            # there are times where this method is called with variable_count
            # set to (x, 0). For example, (2 * v1.diff(x)).diff(x)
            # This 'if' appears to solve the problem....
            return self

        variable_count = list(d0.variable_count) + list(d1.variable_count)
        d = sp.Derivative(self.args[0].expr, *variable_count)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        return D(d)
    
    # def _eval_derivative(self, s):
    #     d = self.args[0]
    #     variable_count = list(d.variable_count) + [(s, 1)]
    #     return D(sp.Derivative(d.expr, *variable_count))

    # def _eval_derivative_n_times(self, s, n):
    #     d = self.args[0]
    #     variable_count = list(d.variable_count) + [(s, n)]
    #     d = sp.Derivative(d.expr, *variable_count)
    #     if len(d.variable_count) > 1:
    #         raise NotImplementedError("Vector differentiation with multiple variables not implemented")
    #     return D(d)

class VecDot(VectorExpr):
    is_Dot = True
    is_scalar = True
    is_Mul = False
    
    def __new__(cls, expr1, expr2, **kwargs):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)

        check = lambda x: isinstance(x, (VectorExpr, Vector))
        if not (check(expr1) and check(expr2) and \
            expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the dot-operator must be vectors")
        if expr1 == VectorZero() or expr2 == VectorZero():
            return sp.S.Zero
    
        obj = sp.Expr.__new__(cls, expr1, expr2)
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
    
    def diff(self, *symbols, **assumptions):
        """
        NOTE: see Magnitude.diff comment to understand why I need to overwrite
        this method.
        """
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        
        evaluate = assumptions.get('evaluate', True)
        if evaluate:
            s, n = d.variable_count[0]
            result = _diff_cross_dot(self, s)
            while n > 1:
                result = result.diff(s)
                n -= 1
            return result
        return D(d)

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
    is_Mul = False
    
    def __new__(cls, expr1, expr2):
        expr1 = sp.sympify(expr1)
        expr2 = sp.sympify(expr2)

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

        obj = sp.Expr.__new__(cls, expr1, expr2)
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
    
    def diff(self, *symbols, **assumptions):
        # assumptions.setdefault("evaluate", True)
        evaluate = assumptions.get('evaluate', True)
        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        if evaluate:
            s, n = d.variable_count[0]
            result = _diff_cross_dot(self, s)
            while n > 1:
                result = result.diff(s)
                n -= 1
            return result
        return D(sp.Derivative(self, *symbols, **assumptions))

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

"""
In a perfect world I would be happy to derive VecAdd from sp.Add,
after all, it's an addition!
If I do that, I would have problems with n-th derivatives.
Consider the following code:

def _eval_derivative_n_times(self, s, n):
    result = self
    for i in range(n):
        result = result._eval_derivative(s)
    print(type(result))     # VecAdd
    return result

The print statement would clearly show that result is of type VecAdd, hovewer
the returned type would be sp.Add. Why? Seems like the result is intercepted by
some other function and post-processed. I don't which function....

Similarly, (v1 ^ v2).diff(x, 2) would return an instance of sp.Add, not VecAdd.

Temporary solution: discard the hierarchy from sp.Add, derives directly from
AssocOp. Now n-th derivatives return VecAdd, but no simplification is done over
the arguments!
"""

class VecAdd(VectorExpr):
# class VecAdd(VectorExpr, sp.Add):
    is_VecAdd = True
    is_commutative = True
    is_Add = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        evaluate = kwargs.get('evaluate', True)
        
        if not args:
            return VectorZero()

        args = list(map(sp.sympify, args))
        if len(args) == 1:
            # doesn't make any sense to have 1 argument in VecAdd if 
            # evaluate=True
            return args[0]

        if evaluate:
            # remove instances of sp.S.Zero and VectorZero
            args = [a for a in args 
                if not (isinstance(a, VectorZero) or (a == sp.S.Zero))]
            
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
            args = _custom_flatten(cls, args)

        # create the final object representing this VecAdd
        obj = Basic.__new__(cls, *args)

        # TODO: addition is not commutative if an argument is not commutative!!!
        obj.is_commutative = True

        # TODO: do I need this???
        # if there is only one argument and it was 1, then obj is the number
        # one, whose property is_Vector is hidden, therefore return 1 immediatly
        if isinstance(obj, sp.Number):
            return VectorZero()
        
        # are there any scalars or vectors?
        are_vectors = any([a.is_Vector for a in args])
        # as of Sympy version 1.5.1, an instance of Vector has is_scalar=True
        are_scalars = any([a.is_scalar for a in args 
            if not isinstance(a, Vector)])
        
        # addition of mixed scalars and vectors is not supported.
        # If there are scalars in the addition, either all arguments
        # are scalars (hence not a vector expression) or there are
        # mixed arguments, hence throw an error
        obj.is_Vector = not are_scalars
        obj.is_scalar = not obj.is_Vector

        if check:
            if all(not isinstance(i, (VectorExpr, Vector)) for i in args):
                return sp.Add.fromiter(args)
            if (are_vectors and are_scalars):
                raise TypeError("Mix of Vector and Scalar symbols")
        return obj
    
    def _eval_derivative(self, s):
        return self.func(*[a.diff(s) for a in self.args])
    
    def diff(self, *symbols, **assumptions):
        evaluate = assumptions.get('evaluate', True)
        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        if evaluate:
            s, n = d.variable_count[0]
            res = self._eval_derivative(s)

            for i in range(n - 1):
                res = res.diff(s)
            return res
        return D(sp.Derivative(self, *symbols, **assumptions))

    def doit(self, **kwargs):
        # Need to override this method in order to apply the rules defined
        # below. Adapted from MatAdd.
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        return canonicalize(VecAdd(*args))

# "necessary" since it doesn't derive from Add.
VecAdd.identity = sp.S.Zero

def _custom_flatten(cls, args):
    """
    Apply Add.flatten() or Mul.flatten() accordingly to the `cls`
    parameter.
    """
    # Because of the way Add.flatten() or Mul.flatten() are coded, they tend 
    # to not work very well with custom types. We have to work around the 
    # problems: `flatten` methods knows how to deal with ordinary symbols.

    # TODO: by excluding numbers and symbols from the following substitution,
    # we might spare ourselves the last step of this method!!!

    # Associate each unique argument to a different dummy symbol.
    # subs_dict_fw is used to go from custom types to dummy symbols.
    # subs_dict_bw is used to go from dummy symbols back to custom types.
    subs_dict_fw = {}
    subs_dict_bw = {}
    set_args = list(set(args))
    for a in set_args:
        d = sp.Dummy()
        subs_dict_fw[a] = d
        subs_dict_bw[d] = a
    # dummy args
    d_args = [subs_dict_fw[a] for a in args]

    cls_map = {
        VecAdd: sp.Add,
        VecMul: sp.Mul
    }
    # apply the correct flatten method
    c_part, nc_part, order_symbols = cls_map[cls].flatten(d_args)
    d_args = c_part + nc_part

    # substitute back the custom types
    args = []
    for a in d_args:
        # `flatten` may group arguments togheter. For example, if we are dealing
        # with VecAdd and the dummy arguments are [dummy_1, dummy_2, dummy_1]
        # after the `flatten` operation we end up with [2 * dummy_1, dummy_2]
        # Note, (2 * dummy_1) is of type Mul. If we were to use the method 
        # `subs`, it would fail due to "RecursionError: maximum recursion depth 
        # exceeded while calling a Python object", because our custom
        # type is not compatible with Mul. Hence, need to reconstruct the 
        # expression with VecMul.
        if isinstance(a, sp.Dummy):
            args.append(subs_dict_bw[a])
        elif isinstance(a, sp.Mul):
            coeff, mul = a.as_coeff_mul()
            args.append(VecMul(coeff, subs_dict_bw[mul[0]]))
        else:
            # deal with Pow objects
            args.append(a.subs(subs_dict_bw))
            # raise NotImplementedError("_custom_flatten not implemented for arguments of type ", type(a))
    
    # because every original argument was substituted with a dummy symbol,
    # even numbers and symbols, there might still be some simplification
    # to be done.
    not_ve_args = [a for a in args if not isinstance(a, VectorExpr)]    
    ve_args = [a for a in args if isinstance(a, VectorExpr)]

    if len(not_ve_args) > 0:
        not_ve_args = cls_map[cls](*not_ve_args)
        if isinstance(not_ve_args, cls_map[cls]):
            not_ve_args = list(not_ve_args.args)
        else:
            not_ve_args = [not_ve_args]
    args = not_ve_args + ve_args

    return args


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


class VecMul(VectorExpr):
# class VecMul(VectorExpr, sp.Mul):
    is_VecMul = True
    is_commutative = True
    # is_Mul = True
    
    def __new__(cls, *args, **kwargs):
        check = kwargs.get('check', True)
        evaluate = kwargs.get('evaluate', True)

        if not args:
            return VectorOne()
        
        args = list(map(sp.sympify, args))
        if len(args) == 1:
            # doesn't make any sense to have 1 argument in VecAdd if 
            # evaluate=True
            return args[0]
        
        vectors = [a.is_Vector for a in args]
        if vectors.count(True) > 1:
            raise TypeError("Multiplication of vector quantities not supported")

        if evaluate:
            # remove instances of sp.S.One
            args = [a for a in args if a is not cls.identity]
            if len(args) == 0:
                return sp.S.One
            if len(args) == 1:
                return args[0]

            # check if any vector is present
            args_is_vector = [a.is_Vector for a in args]
            if any([a == sp.S.Zero for a in args]):
                if any(args_is_vector):
                    return VectorZero()
                return sp.S.Zero
            if any([isinstance(a, VectorZero) for a in args]):
                return VectorZero()

            # Create a new object of type VecAdd. By calling  Basic.__new__ we  
            # avoid infinite loop.
            obj = Basic.__new__(cls, *args)
            # flatten the expression. For example: 
            # VecMul(v1.mag, VecMul(v2.mag, one))
            # becomes: VecMul(v1.mag, v2.mag, one)
            obj = flatten(obj)
            args = _sanitize_args(obj.args)
            args = _custom_flatten(cls, args)

            # in the case of multiplication, _custom_flatten may group arguments
            # together. Think for example to:
            # VecMul(v1.mag, v1.mag) -> VecPow(v1.mag, 2)
            # Need to perform this check again.
            if len(args) == 1:
                return args[0]

        # create the final object representing this VecAdd
        obj = Basic.__new__(cls, *args)
        # addition is commutative
        obj.is_commutative = not any([not a.is_commutative for a in args])
        
        vectors = [a.is_Vector for a in args]
        cvectors = vectors.count(True)
        scalars = [a.is_scalar for a in args]
        cscalars = scalars.count(True)

        # multiplication of vectors is not supported. If there are multiple
        # vectors, an error will be thrown.
        # If all arguments are scalars, the resulting expression wil not
        # be a vector
        obj.is_Vector = cscalars != len(args)
        obj.is_scalar = not obj.is_Vector
        
        if check:
            if cvectors > 1:
                raise TypeError("Multiplication of vector quantities not supported")
            if all(not isinstance(a, VectorExpr) for a in args):
                return sp.Mul.fromiter(args)
        return obj

    def diff(self, *symbols, **assumptions):
        evaluate = assumptions.get('evaluate', True)
        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        if evaluate:
            s, n = d.variable_count[0]

            # adapted from Mul._eval_derivative_n_times
            from sympy import Integer
            from sympy.ntheory.multinomial import multinomial_coefficients_iterator

            args = self.args
            m = len(args)   
            # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
            terms = []
            for kvals, c in multinomial_coefficients_iterator(m, n):
                p = VecMul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
                terms.append(c * p)
            return VecAdd(*terms)

            for i in range(n - 1):
                res = res.diff(s)
            return res
        return D(sp.Derivative(self, *symbols, **assumptions))

    # def _eval_derivative(self, s):
    #     # adapted from Mul._eval_derivative
    #     args = list(self.args)
    #     terms = []
    #     for i in range(len(args)):
    #         d = args[i].diff(s)
    #         if d:
    #             # Note: reduce is used in step of Mul as Mul is unable to
    #             # handle subtypes and operation priority:
    #             terms.append(reduce(lambda x, y: x*y, 
    #                 (args[:i] + [d] + args[i + 1:]), sp.S.One))
    #     return VecAdd.fromiter(terms)
    
    # def _eval_derivative_n_times(self, s, n):
    #     # adapted from Mul._eval_derivative_n_times
    #     from sympy import Integer
    #     from sympy.ntheory.multinomial import multinomial_coefficients_iterator

    #     args = self.args
    #     m = len(args)
    #     if isinstance(n, (int, Integer)):
    #         # https://en.wikipedia.org/wiki/General_Leibniz_rule#More_than_two_factors
    #         terms = []
    #         for kvals, c in multinomial_coefficients_iterator(m, n):
    #             p = VecMul(*[arg.diff((s, k)) for k, arg in zip(kvals, args)])
    #             terms.append(c * p)
    #         return VecAdd(*terms)
    #     raise TypeError("Derivative order must be integer")

VecMul.identity = sp.S.One

class VecPow(VectorExpr):
    def __new__(cls, base, exp):
        base = sp.S(base)
        exp =  sp.S(exp)
        if base.is_Vector:
            raise TypeError("Vector power not available")
        if not isinstance(base, VectorExpr):
            if isinstance(exp, VectorExpr) and exp.is_Vector:
                raise TypeError("Vector exponent not available")
            return sp.Pow(base, exp)
        
        if exp == sp.S.One:
            return base

        obj = Basic.__new__(cls, base, exp)
        # at this point I know the base is an instance of VectorExpr with
        # is_Vector=False, hence is_scalar = True
        obj.is_Vector = False
        obj.is_scalar = True
        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]
    
    def diff(self, *symbols, **assumptions):
        # assumptions.setdefault("evaluate", True)
        evaluate = assumptions.get('evaluate', True)
        # need an unevaluated derivative to check the number of 
        # differentiation symbols
        d = sp.Derivative(self, *symbols)
        if not isinstance(d, sp.Derivative):
            return self
        elif len(d.variable_count) > 1:
            raise NotImplementedError("Vector differentiation with multiple variables not implemented")
        if evaluate:
            return self._eval_derivative(d.variable_count[0][0])
        return D(sp.Derivative(self, *symbols, **assumptions))
    
    def _eval_derivative(self, s):
        # adapted from Pow._eval_derivative
        dbase = self.base.diff(s)
        dexp = self.exp.diff(s)
        return self * (dexp * sp.log(self.base) + dbase * self.exp / self.base)

def _sanitize_args(args):
    """ If an instance of Add, Mul, Pow contains VectorExpr, substitute
    them with VecAdd, VecMul, VecPow respectively.

    flatten() may remove instances of VecMul. For example:
    v1 - v2 = VecAdd(v1, VecMul(-1, v2)) -> (after flatten) ->
    = VecAdd(v1, sp.Mul(-1, v2))
    This is obviously wrong because even if Mul has the property is_Vector,
    it is not updated with the value of VecMul.
    Solution: instead of rewriting flatten(), post process the result
    """
    args = list(args)
    for i, a in enumerate(args):
        if isinstance(a, sp.Add):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(sp.Add, VecAdd)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
        if isinstance(a, sp.Mul):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(sp.Mul, VecMul)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
        if isinstance(a, sp.Pow):
            if any([isinstance(arg, VectorExpr) for arg in a.args]):
                args[i] = a.replace(sp.Pow, VecPow)
                args[i].is_Vector = any([isinstance(arg, VectorSymbol)
                    for arg in a.args])
    return args


# TODO:
# *. find out why class VecAdd(Add) returns Add objects, not VecAdd
# *. printing of (v1 + (v1 ^ one) * (v1 & v2))