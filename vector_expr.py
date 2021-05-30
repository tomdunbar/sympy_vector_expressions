# import sympy as sp
from sympy import (
    sympify, S, Basic, Expr, Derivative, Abs, Mul, Add, Pow, 
    Symbol, Number, log, Wild, fraction, postorder_traversal
)
from sympy.core.compatibility import (
    default_sort_key, with_metaclass, reduce
)
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.evalf import EvalfMixin
from sympy.core.function import UndefinedFunction
from sympy.core.operations import AssocOp
from sympy.core.singleton import Singleton
from sympy.strategies import flatten, typed
from sympy.utilities.iterables import ordered

from sympy.vector import (
    Vector, 
    divergence, curl, gradient
)

# TODO:

class VectorExpr(Expr):
    """ Superclass for vector expressions.

    Example:
    a = VectorSymbol("a")
    b = VectorSymbol("a")
    c = VectorSymbol("a")
    expr = (a ^ (b ^ c)) + (a & b) * c + a.mag * b
    """

    is_Vector = False
    is_VectorExpr = True
    
    is_number = False
    is_symbol = False
    is_scalar = False
    is_commutative = True

    _op_priority = 11
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__radd__')
    def __add__(self, other):
        return VecAdd(self, other)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__add__')
    def __radd__(self, other):
        return VecAdd(other, self)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rdiv__')
    def __div__(self, other):
        return self * other**S.NegativeOne

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__div__')
    def __rdiv__(self, other):
        raise NotImplementedError()

    __truediv__ = __div__
    __rtruediv__ = __rdiv__
    
    def __neg__(self):
        return VecMul(S.NegativeOne, self)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return VecAdd(self, -other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return VecAdd(other, -self)
    
    @_sympifyit('other', NotImplemented)
    def __and__(self, other):
        return VecDot(self, other)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return VecMul(self, other)
    
    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return VecMul(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rpow__')
    def __pow__(self, other):
        return VecPow(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__pow__')
    def __rpow__(self, other):
        return VecPow(other, self)
    
    @_sympifyit('other', NotImplemented)
    def __xor__(self, other):
        return VecCross(self, other)
    
    def dot(self, other):
        return VecDot(self, other)
    
    def cross(self, other):
        return VecCross(self, other)

    def diff(self, *symbols, **assumptions):
        assumptions.setdefault("evaluate", False)            #TD 5/30/21  Changed to False
        return D(Derivative(self, *symbols, **assumptions))

    def normalize(self, **kwargs):
        # TODO: what if I do a check for evaluate=True in Normalize
        # constructor?
        evaluate = kwargs.get('evaluate', False)
        n = Normalize(self)
        if evaluate:
            n.doit(**kwargs)
        return n
    
    @property
    def norm(self):
        return self.normalize()

    def magnitude(self):
        return Magnitude(self)
    
    @property
    def mag(self):
        return self.magnitude()
    
    def gradient(self):
        return Grad(self)
    
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

    def laplace(self):
        return Laplace(self)
    
    @property
    def lap(self):
        return self.laplace()
    
    def expand(self, deep=True, modulus=None, power_base=True, power_exp=True,
            mul=True, log=True, multinomial=True, basic=True, **hints):
        if isinstance(self, VectorSymbol):
            return self

        _spaces.append(_token)
        debug("VectorExpr expand", self)
        # first expand dot and cross products
        A, B = [WVS(t) for t in ["A", "B"]]
        def func(expr, pattern, prev=None):
            # debug("\n", hints)
            old = expr
            found = list(ordered(list(expr.find(pattern))))
            # print("func", expr)
            # debug("found", found)
            # print("found", found)
            for f in found:
                fexp = f.expand(**hints)
                # debug("\t fexp", fexp)
                if f != fexp:
                    expr = expr.xreplace({f: fexp})
            # debug("expr", expr)
            if old != expr:
                expr = func(expr, pattern)
            return expr

        expr = func(self, A ^ B)
        # print("expr", expr)
        expr = func(expr, A & B)
        # print("expr", expr)
        expr = func(expr, Grad)
        expr = func(expr, Laplace)
        # print("expr", expr)

        del _spaces[-1]
        return Expr.expand(expr, deep=deep, modulus=modulus, power_base=power_base,
            power_exp=power_exp, mul=mul, log=log, multinomial=multinomial, 
            basic=basic, **hints)

def get_postprocessor(cls):
    def _postprocessor(expr):
        vec_class = {Mul: VecMul, Add: VecAdd, Pow: VecPow}[cls]
        nonvectors = []
        vectors = []

        print("POST-PROCESSOR", cls)

        for term in expr.args:
            print("\t", term.func, term)
            # if isinstance(term, (VectorExpr, Vector)):
            if term.has(VectorExpr) or term.has(Vector):
                vectors.append(term)
            else:
                nonvectors.append(term)
        
        if not vectors:
            print("YEAH")
            return cls._from_args(nonvectors)
        
        
        print("\tvectors", vectors)
        print("\tnonvectors", nonvectors)

        if vec_class == VecAdd or vec_class == VecMul:
            return vec_class(*nonvectors, *vectors, evaluate=True)
        else:
            return vec_class(*expr.args)
    return _postprocessor

Basic._constructor_postprocessor_mapping[VectorExpr] = {
    "Mul": [get_postprocessor(Mul)],
    "Add": [get_postprocessor(Add)],
    "Pow": [get_postprocessor(Pow)],
}

def wtf():
    from sympy import collect_const
    a = VectorSymbol("a")
    b = VectorSymbol("b")
    expr = 3 * a + 3 * b
    print(collect_const(expr))


class VectorSymbol(VectorExpr):
    """ Symbolic representation of a Vector object.
    """
    is_symbol = True
    is_Vector = True
    
    _vec_symbol = ""
    _unit_vec_symbol = ""
    _bold = False
    _italic = False
    
    def __new__(cls, name, **kwargs):
        if not isinstance(name, (str, Symbol, VectorSymbol)):
            raise TypeError("'name' must be a string or a Symbol or a VectorSymbol")
        if isinstance(name, VectorSymbol):
            return name
        if isinstance(name, str):
            name = Symbol(name)
        obj = Basic.__new__(cls, name)

        # TODO:
        # Say we created the following two symbols:
        # v1a = VectorSymbol("v1", _vec_symbol="\hat{%s}")
        # v1b = VectorSymbol("v1")
        #       v1a == v1b  -> True
        # Do I want to include the attributes in the hash to make them
        # different?
        
        obj._vec_symbol = kwargs.get('_vec_symbol', "")
        obj._unit_vec_symbol = kwargs.get('_unit_vec_symbol', "")
        obj._bold = kwargs.get('_bold', False)
        obj._italic = kwargs.get('_italic', False)

        return obj

    def doit(self, **kwargs):
        return self
    
    @property
    def free_symbols(self):
        return set((self,))

    @property
    def name(self):
        return self.args[0].name

class VectorZero(VectorSymbol):
    is_ZeroVector = True
    
    def __new__(cls, name="0", **kwargs):
        return super().__new__(cls, name, **kwargs)
    
    def magnitude(self):
        return S.Zero
    
    def normalize(self):
        return self

    def _eval_derivative(self, s):
        return self

class VectorOne(VectorSymbol):
    def __new__(cls, name="1", **kwargs):
        return super().__new__(cls, name, **kwargs)
    
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
    
    @_sympifyit('other', NotImplemented)
    def gradient(self, other):
        return Grad(self, other)
    
    @_sympifyit('other', NotImplemented)
    def laplace(self, other):
        return Laplace(self, other)
    
    grad = gradient
    lap = laplace

    def _eval_derivative(self, s):
        raise NotImplementedError("Differentiation of nabla operator not implemented.")

class WildVectorSymbol(Wild, VectorSymbol):
    def __new__(cls, name, exclude=(), properties=(), **assumptions):
        obj = Wild.__new__(cls, name, exclude=(), properties=(), **assumptions)
        return obj

WVS = WildVectorSymbol

class Normalize(VectorExpr):
    """ Symbolic representation of a normalized symbolic vector. Given a vector
    v, the normalized form is: v / v.magnitude
    """
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
    """ Symbolic representation of the magnitude of a symbolic vector.
    """
    is_Vector = False
    is_positive = True

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
        obj.is_scalar = d.expr.is_scalar  #TD 5/30/21
        return obj
    
    def _eval_derivative(self, s):
        d = self.args[0]
        variable_count = list(d.variable_count) + [(s, 1)]
        return D(Derivative(d.expr, *variable_count))

class DotCross(VectorExpr):
    """ Abstract base class for VecCross and VecDot.
    Not to be instantiated directly.
    """
    @property
    def reverse(self):
        raise NotImplementedError

    def diff(self, *symbols, **assumptions):
        if isinstance(self.args[0], Nabla):
            return D(Derivative(self, *symbols))
        return super().diff(*symbols, **assumptions)
    
    def _eval_derivative(self, s):
        expr0 = self.args[0].diff(s)
        expr1 = self.args[1].diff(s)
        t1 = self.func(expr0, self.args[1])
        t2 = self.func(self.args[0], expr1)
        return t1 + t2
    
    def expand(self, **hints):
        # dot = hints.get('dot', True)
        # cross = hints.get('cross', True)
        prod = hints.get('prod', False)
        quotient = hints.get('quotient', False)

        _spaces.append(_token)
        debug("DotCross expand", self)
        # left = self.args[0]
        # right = self.args[1]
        left = self.args[0].expand(**hints)
        right = self.args[1].expand(**hints)
            
        get_args = lambda expr: expr.args if isinstance(expr, VecAdd) else [expr]

        # deal with divergence/curl
        if isinstance(left, Nabla):
            num, den = fraction(right)
            if den != S.One and quotient:
                # quotient rule
                return (den * self.func(left, num) - self.func(Grad(den), num)) / den**2
            if prod and den == 1 and isinstance(right, VecMul):
                # product rule
                if len(right.args) > 1:
                    vector = None
                    scalars = []
                    for a in right.args:
                        if a.is_Vector:
                            vector = a
                        else:
                            scalars.append(a)
                    new_args = [VecMul(*scalars, self.func(left, vector))]
                    for i, s in enumerate(scalars):
                        new_args.append(
                            VecMul(*scalars[:i], *scalars[i+1:], self.func(Grad(s), vector))
                        )
                    return VecAdd(*new_args)
                return self

            right = get_args(right)
            return VecAdd(*[self.func(left, r) for r in right])

        debug("\t", left, right)
        if not (isinstance(left, VecAdd) or isinstance(right, VecAdd)):
            debug("\t ASD0 not VecAdd", self.func(left, right))
            del _spaces[-1]
            return self.func(left, right)
        
        left = get_args(left)
        right = get_args(right)
        debug("\t ASD1 left", left)
        debug("\t ASD2 right", right)
        def _get_vector(expr):
            if isinstance(expr, (VectorSymbol, VecCross, VecDot, Grad, Laplace, Advection)):
                return expr
            return [t for t in expr.args if t.is_Vector][0]
        
        def _get_coeff(expr):
            if isinstance(expr, (VectorSymbol, Grad, Advection, Laplace)):
                return 1
            return VecMul(*[t for t in expr.args if not t.is_Vector])
             
        # get_vector = lambda expr: [t for t in expr.args if t.is_Vector][0]
        # get_coeff = lambda expr: VecMul(*[t for t in expr.args if not t.is_Vector])
        terms = []
        for l in left:
            cl = _get_coeff(l)
            vl = _get_vector(l)
            for r in right:
                cr = _get_coeff(r)
                vr = _get_vector(r)
                # print("vl", vl)
                # print("vr", vr)
                c = cl * cr
                terms.append(c * self.func(vl, vr))
        debug("\t ASD3 terms", terms)
        del _spaces[-1]
        return VecAdd(*terms)

class VecDot(DotCross):
    """ Symbolic representation of the dot product between two symbolic vectors.
    """
    is_Vector = False    #TD 5/30/21  Added these two lines to pass test_d tests
    is_scalar = True     #TD 5/30/21
    is_Dot = True
    
    def __new__(cls, expr1, expr2, **kwargs):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)

        check = lambda x: isinstance(x, (VectorExpr, Vector))
        if not (check(expr1) and check(expr2) and \
            expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the dot-operator must be vectors:\n" +
                "\t Left: " + str(expr1.func) + ", " + str(expr1.is_Vector) + ", {}\n".format(expr1) +
                "\t Right: " + str(expr2.func) + ", " + str(expr2.is_Vector) + ", {}\n".format(expr2)
        )
        if expr1 == VectorZero() or expr2 == VectorZero():
            return S.Zero
        if expr1 == expr2 and isinstance(expr1, Nabla):
            raise TypeError("Dot product of two nabla operators not supported.\n" +
                "To create the Laplacian operator, use the class Laplace.")
        if isinstance(expr2, Nabla):
            raise TypeError("To compute the divergence, nabla operator must be the first argument.\n" + 
                "To write the advection operator, use the class Advection."
        )
        obj = Expr.__new__(cls, expr1, expr2)
        return obj
    
    @property
    def reverse(self):
        # take into account the fact that arg[0] and arg[1] could be mixed 
        # instances of Vector and VectorExpr
        return self.func(self.args[1], self.args[0])

    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)

        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]

        if isinstance(args[0], Vector) and \
            isinstance(args[1], Vector):
            return args[0].dot(args[1])
        if isinstance(args[0], Vector) and \
            isinstance(args[1], Nabla):
            return divergence(args[0])
        if isinstance(args[1], Vector) and \
            isinstance(args[0], Nabla):
            return divergence(args[1])
        
        if args[0] == args[1]:
            return VecPow(args[0].mag, 2)
        return self.func(*args)
    
    def expand(self, **hints):
        dot = hints.get('dot', True)
        if dot:
            return super().expand(**hints)
        left = self.args[0].expand(**hints)
        right = self.args[1].expand(**hints)
        return self.func(left, right)

class VecCross(DotCross):
    """ Symbolic representation of the cross product between two symbolic 
    vectors.
    """
    is_Cross = True
    is_Vector = True
    is_commutative = False

    def __new__(cls, expr1, expr2):
        expr1 = sympify(expr1)
        expr2 = sympify(expr2)

        check = lambda x: isinstance(x, (VectorExpr, Vector))
        if not (check(expr1) and check(expr2) and \
                expr1.is_Vector and expr2.is_Vector):
            raise TypeError("Both side of the cross-operator must be vectors\n" +
                "\t Left: " + str(expr1.func) + ", " + str(expr1.is_Vector) + ", {}\n".format(expr1) +
                "\t Right: " + str(expr2.func) + ", " + str(expr2.is_Vector) + ", {}\n".format(expr2)
            )
        if expr1 == VectorZero() or expr2 == VectorZero():
            return VectorZero()
        if (isinstance(expr1, Vector) and isinstance(expr2, Vector) and
            expr1 == expr2):
            # TODO: At this point I'm dealing with unevaluated cross product.
            # is it better to return VectorZero()?
            return expr1.zero
        if expr1 == expr2 and isinstance(expr1, Nabla):
            raise TypeError("Cross product of two nabla operators not supported.")
        if isinstance(expr2, Nabla):
            raise TypeError("To compute the curl, nabla operator must be the first argument.\n" + 
                "To write the advection operator, use the class Advection.\n" + 
                "\t Left: " + str(expr1.func) + "{}\n".format(expr1) +
                "\t Right: " + str(expr2.func) + "{}\n".format(expr2)
        )
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

        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]

        if isinstance(args[0], Vector) and \
            isinstance(args[1], Vector):
            return args[0].cross(args[1])
        if isinstance(args[0], Vector) and \
            isinstance(args[1], Nabla):
            return -curl(args[0])
        if isinstance(args[1], Vector) and \
            isinstance(args[0], Nabla):
            return curl(args[1])

        return self.func(*args)
    
    def expand(self, **hints):
        cross = hints.get('cross', True)
        if cross:
            return super().expand(**hints)
        left = self.args[0].expand(**hints)
        right = self.args[1].expand(**hints)
        return self.func(left, right)


class Advection(VectorExpr):
    """ Symbolic representation of the following operator/expression:
        (v & nabla) * f
    where:
        v : vector
        f : vector or scalar field
    Note that this is different than (nabla & v) * f, because (nabla & v) is the
    divergence of v.
    """
    is_Vector = True
    is_commutative = False

    def __new__(cls, v, f, n = Nabla()):
        v = sympify(v)
        f = sympify(f)
        if not isinstance(n, Nabla):
            raise TypeError("n must be an instance of the class Nabla.")
        if (not v.is_Vector) or isinstance(v, Nabla):
            raise TypeError("v must be a vector or an expression with is_Vector=True. It must not be nabla.")
        if isinstance(f, Nabla):
            raise TypeError("f (the field) cannot be nabla.")

        obj = Expr.__new__(cls, v, f, n)
        obj.is_Vector = v.is_Vector
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)

        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]
        
        v, f, n = args
        if not f.is_Vector and isinstance(v, Vector):
            return v & Grad(f).doit(**kwargs)
        if isinstance(f, Vector) and isinstance(v, Vector):
            s = Vector.zero
            for e, comp in f.components.items():
                s += e * Advection(v, comp).doit(**kwargs)
            return s

        return self.func(*args)


# used for debugging the expressions trees: append a _token to _spaces each time
# an object is created. Remember to remove the token once the object has been
# created
_spaces = []
_token = "    "
_DEBUG = False
def debug(*args):
    if _DEBUG:
        print("".join(_spaces), *args)

# _print_Gradient is already used in pretty_print, hence the name Gradient
class Grad(VectorExpr):
    """ Symbolic representation of the gradient of a scalar field.
    """
    is_Vector = True
    
    def __new__(cls, arg1, arg2=None, **kwargs):
        # Ideally, only the scalar field is required. However, it can
        # accept also the nabla operator (in case of some rendering
        # customization)
        if not arg2:
            n = Nabla()
            f = sympify(arg1)
        else:
            n = sympify(arg1)
            f = sympify(arg2)
        
        if not isinstance(n, Nabla):
            raise TypeError("The first argument of the gradient operator must be Nabla.")
        if isinstance(f, Nabla):
            raise NotImplementedError("Differentiation of nabla operator not implemented.")
        if f.is_Vector:
            raise TypeError("Gradient of vector fields not implemented.")
        return Basic.__new__(cls, n, f)
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]

        # TODO: is the following condition ok?
        if not isinstance(args[1], VectorExpr):
            return gradient(args[1])
        return self.func(*args)
    
    def expand(self, **hints):
        gradient = hints.get('gradient', False)
        prod = hints.get('prod', False)
        quotient = hints.get('quotient', False)

        if gradient:
            if isinstance(self.args[1], Add):
                return VecAdd(*[self.func(a).expand(**hints) for a in self.args[1].args])
            
            # if isinstance(self.args[1], Mul):
            num, den = fraction(self.args[1])
            if den != S.One and quotient:
                # quotient rule
                return (den * Grad(num) - num * Grad(den)) / den**2
            if prod and den == 1:
                # product rule
                args = self.args[1].args
                if len(args) > 1:
                    new_args = []
                    for i, a in enumerate(args):
                        new_args.append(
                            VecMul(*args[:i], *args[i+1:], self.func(a))
                        )
                    return VecAdd(*new_args)
        return self

class Laplace(VectorExpr):
    """ Symbolic representation of the Laplacian operator.
    """
    is_Vector = False
    
    def __new__(cls, arg1, arg2=None, **kwargs):
        # Ideally, only the scalar/vector field is required. However, it can
        # accept also the nabla operator (in case of some rendering
        # customization)
        if not arg2:
            n = Nabla()
            f = sympify(arg1)
        else:
            n = sympify(arg1)
            f = sympify(arg2)
        
        if not isinstance(n, Nabla):
            raise TypeError("The first argument of the laplace operator must be Nabla.")
        if isinstance(f, Nabla):
            raise NotImplementedError("Differentiation of nabla operator not implemented.")
        # if f.is_Vector:
        #     raise TypeError("Gradient of vector fields not implemented.")
        obj = Basic.__new__(cls, n, f)
        if f.is_Vector:
            obj.is_Vector = True
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        args = self.args
        if deep:
            args = [arg.doit(**kwargs) for arg in args]

        if not isinstance(args[1], (Vector, VectorExpr)):
            # scalar field: we use the identity nabla^2 = nabla.div(f.grad())
            return VecDot(args[0], Grad(args[0], args[1])).doit(deep=deep)
        elif isinstance(args[1], Vector):
            components = args[1].components
            laplace = lambda c: VecDot(args[0], Grad(args[0], c)).doit()
            v = Vector.zero
            for base, comp in components.items():
                v += base * laplace(comp)
            return v
        return self.func(*args)
    
    def expand(self, **hints):
        laplacian = hints.get('laplacian', False)
        prod = hints.get('prod', False)
        if laplacian:
            if isinstance(self.args[1], Add):
                return VecAdd(*[self.func(a).expand(**hints) for a in self.args[1].args])
            if prod:
                # product rule:
                n, f = self.args
                args = f.args
                if (len(args) == 2) and all([not a.is_Vector for a in args]):
                    return args[0] * Laplace(n, args[1]) + args[1] * Laplace(n, args[0]) + 2 * (Grad(n, args[0]) & Grad(n, args[1]))
        return self

class VecAdd(VectorExpr, Add):
    """ A sum of Vector expressions.

    VecAdd inherits from and operates like SymPy Add.
    """
    is_VecAdd = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', True)

        if not args:
            return VectorZero()

        args = list(map(sympify, args))
        # If for any reasons we create VecAdd(0), I want 0 to be returned.
        # Skipping this check, VectorZero() will be returned instead
        if len(args) == 1 and args[0] == S.Zero:
            return S.Zero
        
        if evaluate:
            # remove instances of S.Zero and VectorZero
            args = [a for a in args 
                if not (isinstance(a, VectorZero) or (a == S.Zero) or (a == Vector.zero))]
            if len(args) == 0:
                return VectorZero()
            elif len(args) == 1:
                # doesn't make any sense to have 1 argument in VecAdd if 
                # evaluate=True
                return args[0]

        obj = AssocOp.__new__(cls, *args, evaluate=evaluate)
        obj = _sanitize_args(obj)
        
        # print("VecAdd OBJ", obj.func, obj)
        if not isinstance(obj, cls):
            return obj

        # are there any scalars or vectors?
        any_vectors = any([a.is_Vector for a in obj.args])
        all_vectors = all([a.is_Vector for a in obj.args])

        # addition of mixed scalars and vectors is not supported.
        # If there are scalars in the addition, either all arguments
        # are scalars (hence not a vector expression) or there are
        # mixed arguments, hence throw an error
        obj.is_Vector = all_vectors

        if (any_vectors and (not all_vectors)):
            asd = obj.args[1]
            print("####", asd.is_Vector, asd.args)
            print("####", [a.is_Vector for a in asd.args])
            for a in postorder_traversal(asd):
                print(a.func, a.is_Vector, a)
            raise TypeError("VecAdd: Mix of Vector and Scalar symbols:\n\t" + 
                "\n\t".join(str(a.func) + ", " + str(a.is_Vector) + ", " + str(a) for a in obj.args)
            )
        return obj

    def doit(self, **kwargs):
        # Need to override this method in order to apply the rules defined
        # below. Adapted from MatAdd.
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        return canonicalize(VecAdd(*args))


from sympy.strategies import (sort, condition, exhaust, do_one)
from sympy.utilities.iterables import sift
from operator import add, mul

def merge_explicit(vecadd):
    """ Merge Vector arguments
    Example
    ========
    >>> from sympy.vector import CoordSys3D
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
    # there has to be some bug deep inside the core, I absolutely need this 
    # function to correctly perform addition
    def recreate_args(args):
        return [a.func(*a.args) for a in args]
    # Adapted from sympy.matrices.expressions.matadd.py
    groups = sift(vecadd.args, lambda arg: isinstance(arg, (Vector)))
    if len(groups[True]) > 1:
        return VecAdd(*(recreate_args(groups[False]) + [reduce(add, recreate_args(groups[True]))]))
        # return VecAdd(*(groups[False] + [reduce(add, groups[True])]))
    else:
        return vecadd

rules = (
    merge_explicit,
    sort(default_sort_key)
)

canonicalize = exhaust(condition(lambda x: isinstance(x, VecAdd),
                                 do_one(*rules)))

class VecMul(VectorExpr, Mul):
    """ A product of Vector expressions.

    VecMul inherits from and operates like SymPy Mul.
    """
    is_VecMul = True
    is_commutative = True
    
    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', True)

        print("VecMul __new__", args)
        if not args:
            # it makes sense to return VectorOne, after all we are talking
            # about VecMul. However, this would play against us when using
            # expand(), because we would have multiplications of multiple 
            # vectors (one of which, VectorOne). 
            # Hence, better to return S.One.
            return S.One

        args = list(map(sympify, args))
        if len(args) == 1:
            # doesn't make any sense to have 1 argument in VecAdd if 
            # evaluate=True
            return args[0]
        
        # # TODO: look through args (which are unprocessed as of now) for
        # # (a & nabla) * x (where x can be a scalar field or a vector field)
        # dot, field = None, None
        # skip_idx = []
        # dot_field = []
        # for i in range(len(args) - 1):
        #     if isinstance(args[i], VecDot) and isinstance(args[i].args[1], Nabla):
        #         dot = args[i]
        #         field = args[i + 1]
        #         skip_idx.extend([i, i + 1])

        vectors = [a.is_Vector for a in args]
        if vectors.count(True) > 1:
            raise TypeError("Multiplication of vector quantities not supported\n\t" + 
                "\n\t".join(str(a.func) + ", " + str(a) for a in args)
            )

        # remove instances of S.One
        args = [a for a in args if a is not cls.identity]
        if len(args) == 0:
            return S.One
        if len(args) == 1:
            return args[0]
            
        if evaluate:
            # check if any vector is present
            args_is_vector = [a.is_Vector for a in args]
            if any([a == S.Zero for a in args]):
                if any(args_is_vector):
                    return VectorZero()
                return S.Zero
            if any([(isinstance(a, VectorZero) or (a == Vector.zero)) for a in args]):
                return VectorZero()
            if (all([not isinstance(a, VectorExpr) for a in args]) and 
                any([not isinstance(a, Vector) for a in args]) and
                len(args) > 1):
                return reduce(mul, args)
                # return _prod(args)
        
        print("\t pre AssocOp.__new__", args)
        obj = AssocOp.__new__(cls, *args, evaluate=evaluate)
        obj = _sanitize_args(obj)
        print("\t post AssocOp.__new__", obj.func)
        for a in obj.args:
            print("\t\t", a.func, a.is_Vector, a)
        # obj = obj.func._exec_constructor_postprocessors(obj)

        # At this point, obj might not be of type VecMul. We must return
        # otherwise the next checks are going to be applied.
        # Example #1: VecMul(3, a + b) -> VecAdd(3 * a, 3 * b)
        # Example #2:
        # C = CoordSys3D("C")
        # vn1 = C.i + 2 * C.j + 3 * C.k
        # vn2 = x * C.i + y * C.j + z * C.k
        # VecMul(3, VecDot(vn1, vn2)).doit() -> Add(x, 2 * y, 3 * z)
        if not isinstance(obj, VecMul):
            return obj
        
        vectors = [a.is_Vector for a in obj.args]
        cvectors = vectors.count(True)
        obj.is_Vector = cvectors > 0

        if cvectors > 1:
            raise TypeError("VecMul: Multiplication of vector quantities not supported\n\t" + 
                "\n\t".join(str(a.func) + ", " + str(a.is_Vector) + ", " + str(a) for a in obj.args)
            )
        return obj
    
    def doit(self, **kwargs):
        deep = kwargs.get('deep', True)
        if deep:
            args = [arg.doit(**kwargs) for arg in self.args]
        else:
            args = self.args
        
        are_vectors = any([isinstance(a, Vector) for a in args])
        expr = canonicalize_vecmul(VecMul(*args))
        # if there were instances of Vector, and the evaluated expression is
        # VectorZero, we should return Vector.zero (from sympy.vector module)
        if isinstance(expr, VectorZero) and are_vectors:
            return Vector.zero
        return expr

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

from sympy import symbols
from sympy.vector import CoordSys3D
def asd():
    x, y, z = symbols("x:z")
    a, b, c = [VectorSymbol(t) for t in ["a", "b", "c"]]
    nabla = Nabla()
    R = CoordSys3D("R")
    xx = R.x * R.y
    yy = R.x
    v = R.x * R.i + R.y * R.j + R.k * R.z
    o = R.i + R.j + R.k
    expr = nabla ^ (x * y * a)
    # expr = expr.expand(prod=True).subs({x: xx, y: yy, a: v})
    # expr = VecMul(2, v, a.mag, x, evaluate=False)
    # expr = VecMul(VectorZero(), 2, evaluate=False)
    expr = VecMul(Vector.zero, 2, evaluate=False)
    # expr = VecAdd(o, v, Vector.zero)
    print(expr)
    print(expr.doit())

# import operator
# from functools import reduce
# def _prod(factors):
#     return reduce(operator.mul, factors, 1)

def merge_explicit_mul(vecmul):
    """ Merge explicit Vector and Expr (Numbers, Symbols, ...) arguments
    Example
    ========
    >>> from sympy.vector import CoordSys3D
    >>> v1 = VectorSymbol("v1")
    >>> v2 = VectorSymbol("v2")
    >>> C = CoordSys3D("C")
    >>> vn1 = R.x * R.i + R.y * R.j + R.k * R.z
    >>> expr = VecMul(2, v1.mag, vn1, x, evaluate=False)
    >>> pprint(expr)
    2*x*(R.x*R.i + R.y*R.j + R.z*R.k)*Magnitude(VectorSymbol(v1))
    >>> pprint(merge_explicit(expr))
    (2*R.x*x*R.i + 2*R.y*x*R.j + 2*R.z*x*R.k)*Magnitude(VectorSymbol(v1))
    """

    groups = sift(vecmul.args, lambda arg: not isinstance(arg, VectorExpr))
    print("\t", groups)
    if len(groups[True]) > 1:
        return VecMul(*(groups[False] + [reduce(mul, groups[True])]))
    else:
        return vecmul

rules_vecmul = (merge_explicit_mul, )
canonicalize_vecmul = exhaust(typed({VecMul: do_one(*rules_vecmul)}))




class VecPow(VectorExpr, Pow):
    """ A power of Vector expressions.

    VecPow inherits from and operates like SymPy Pow.
    """
    def __new__(cls, base, exp):
        base = S(base)
        exp =  S(exp)
        if base.is_Vector:
            raise TypeError("Vector power not available")
        if exp.is_Vector:
            raise TypeError("Vector power not available")

        if ((not isinstance(base, VectorExpr)) and 
            (not isinstance(exp, VectorExpr))):
            return Pow(base, exp)
        
        if base == S.One:
            return base
        if exp == S.One:
            return base

        obj = Basic.__new__(cls, base, exp)
        # at this point I know the base is an instance of VectorExpr with
        # is_Vector=False, hence is_scalar = True
        obj.is_Vector = False
        return obj

    @property
    def base(self):
        return self.args[0]

    @property
    def exp(self):
        return self.args[1]

def _compute_is_vector(expr):
    """ Compute the value for the attribute is_Vector based on the
    arguments of expr.
    """
    if isinstance(expr, VecAdd):
        return all([a.is_Vector for a in expr.args])
    if isinstance(expr, VecMul):
        vectors = [a.is_Vector for a in expr.args]
        cvectors = vectors.count(True)
        return cvectors > 0
    if isinstance(expr, D):
        return expr.args[0].expr.is_Vector
    if isinstance(expr, (VecPow, VecDot, Magnitude)):
        return False
    if isinstance(expr, (VectorSymbol, VecCross, Normalize, Grad)):
        return True
    if isinstance(expr, Laplace):
        return expr.args[1].is_Vector
    if isinstance(expr, Advection):
        return expr.args[1].is_Vector
    return expr.is_Vector

# def _sanitize_args(expr):
#     """ Add.flatten and Mul.flatten could change the value of the 
#     attribute is_Vector of any VectorExpr. Need to recompute the correct
#     value.
#     """

#     for a in postorder_traversal(expr):
#         if isinstance(a, VectorExpr):
#             a.is_Vector = _compute_is_vector(a)
#     return expr

def _sanitize_args(obj):
    """ The methods Add.flatten and Mul.flatten can create arguments of obj that
    are not post-processed.
    _sanitize_args looks for arguments of type Add, Mul, Pow and call the
    post-processor. If the original argument contained a vector expression, 
    after the post-processor the new argument will be of type VecAdd, VecMul or
    VecPow.
    Also, because Add.flatten and Mul.flatten could change the value of the 
    attribute is_Vector of any VectorExpr, we are going to recompute the correct
    value.
    """
    args = list(obj.args)
    print("SANITIZE")
    for i, a in enumerate(args):
        if isinstance(a, (Add, Mul, Pow)) and (not isinstance(a, VectorExpr)):
            print("\tasd", a.func, a)
            args[i] = args[i].func._exec_constructor_postprocessors(args[i]).doit()

        if isinstance(a, VectorExpr):
            args[i].is_Vector = _compute_is_vector(args[i])
    
    # the following is used to avoid infinite recursion
    if not isinstance(obj, (Add, Mul)):
        return obj.func(*args)
    obj = obj.func._from_args(args)
    if isinstance(obj, VectorExpr):
        obj.is_Vector = _compute_is_vector(obj)
    return obj

if __name__ == "__main__":
    # a = VectorSymbol("a")
    # b = VectorSymbol("b")
    # x = Symbol("x")
    # # # b = VectorSymbol("b")
    # # # r = VecMul(a + b, 3)
    # # # print(r.func, r.is_Vector, r)
    # # # a.norm.doit().diff(x)
    # nabla = Nabla()
    # expr = VecMul(4, (x * a.div + 1))
    # print(expr.is_Vector)
    # # expr = VecMul(3, a + b)
    # # print(expr.is_Vector)
    # wtf()
    a = VectorSymbol("a")
    x = Symbol("x")
    Add(x, x)

    