from itertools import combinations
from collections import OrderedDict
from sympy import (
    S, Mul, Add, Abs, preorder_traversal, collect_const, Wild
)
from sympy.utilities.iterables import ordered
from vector_expr import (
    VectorExpr, VecAdd, VecMul, VecPow, VecDot, VecCross, 
    Magnitude, Normalize, WVS, VectorSymbol, VectorOne, DotCross,
    Grad, Laplace, VectorZero, Nabla, DotNablaOp
)

# TODO: 
# 1. Create test for the method with and without parameter 'match'
# 2. expr = a.mag * (a ^ b) + a.mag * (a ^ c)
#    simplify(expr)
#    a.mag * (a ^ (b + c))
# 3. Fix Del Operator:
#       https://en.wikipedia.org/wiki/Del
# 4. Vector Identities
#       https://en.wikipedia.org/wiki/Vector_calculus_identities

def _find_and_replace(expr, pattern, rep, matches=None):
    """ Given an expression, a search `pattern` and a replacement 
    pattern `rep`, search for `pattern` in the expression and substitute 
    it with `rep`. If matches is given, skip the pattern search and 
    substitute each match with the corresponding `rep` pattern.
    """
    if not matches:
        # list of matches ordered accordingly to the length of the match.
        # The shorter the match expression, the lower it should be in the
        # expression tree. Substitute from the bottom up!
        found = list(ordered(list(expr.find(pattern))))
        if len(found) > 0:
            f = found[0]
            expr = expr.xreplace({f: rep.xreplace(f.match(pattern))})
            # repeat the procedure with the updated expression
            expr = _find_and_replace(expr, pattern, rep)   
        return expr
    
    if not isinstance(matches, (list, tuple)):
        matches = [matches]
    for match in matches:
        expr = expr.xreplace({match: rep.xreplace(match.match(pattern))})
    return expr

def bac_cab(expr, forward=True, matches=None):
    """ Implement the replacement rule:
    A x (B x C) = B (A . C) - C (A . B)
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    
    If forward=True, search for the pattern A ^ (B ^ C), otherwise
    search for the pattern B (A . C) - C (A . B).

    If `matches` is given, replace only the matching subexpressions.

    This is a commodity function to use `bac_cab_forward` and 
    `bac_cab_backward` with a single command.
    """
    if forward:
        return bac_cab_forward(expr, matches)
    else:
        return bac_cab_backward(expr, matches)


def bac_cab_forward(expr, matches=None):
    """ Implement the replacement rule:
    A x (B x C) = B (A . C) - C (A . B)
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    
    If `matches` is given, replace only the matching subexpressions.
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    pattern = A ^ (B ^ C)
    rep = (B * (A & C)) - (C * (A & B))
    return _find_and_replace(expr, pattern, rep, matches) 


def dot_cross(expr, matches=None):
    """ Implement the replacement rule:
    A . (B x C) = (A x B) . C
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    
    If `matches` is given, replace only the matching subexpressions.
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    pattern = A & (B ^ C)
    rep = (A ^ B) & C
    return _find_and_replace(expr, pattern, rep, matches) 

def find_double_cross(expr):
    """ Given a vector expression, return a list of elements satisfying
    the pattern A x (B x C), where A, B, C are vectors and `x` is the
    cross product. The list is ordered in such a way that nested matches
    comes first (similarly to scanning the expression tree from bottom 
    to top).
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    pattern = A ^ (B ^ C)
    return list(ordered(list(expr.find(pattern))))

def find_bac_cab(expr):
    """ Given a vector expression, find the terms satisfying the pattern:
                B * (A & C) - C * (A & B)
    where & is the dot product. The list is ordered in such a way that 
    nested matches comes first (similarly to scanning the expression tree from 
    bottom to top).
    """
    def _check_pairs(terms):
        c1 = Mul(*[a for a in terms[0].args if not hasattr(a, "is_Vector_Scalar")])
        c2 = Mul(*[a for a in terms[1].args if not hasattr(a, "is_Vector_Scalar")])
        n1, _ = c1.as_coeff_mul()
        n2, _ = c2.as_coeff_mul()
        if n1 * n2 > 0:
            # opposite sign
            return False
        if Abs(c1) != Abs(c2):
            return False
        v1 = [a for a in terms[0].args if (hasattr(a, "is_Vector_Scalar") and a.is_Vector)][0]
        v2 = [a for a in terms[1].args if (hasattr(a, "is_Vector_Scalar") and a.is_Vector)][0]
        if v1 == v2:
            return False
        dot1 = [a for a in terms[0].args if isinstance(a, VecDot)][0]
        dot2 = [a for a in terms[1].args if isinstance(a, VecDot)][0]
        if not ((v1 in dot2.args) and (v2 in dot1.args)):
            return False
        v = list(set(dot1.args).intersection(set(dot2.args)))[0]
        if v == v1 or v == v2:
            return False
        return True
    
    def _check(arg):
        if (isinstance(arg, VecMul) and
           any([a.is_Vector for a in arg.args]) and
           any([isinstance(a, VecDot) for a in arg.args])):
            return True
        return False

    found = set()
    for arg in preorder_traversal(expr):
        possible_args = list(filter(_check, arg.args))
        # possible_args = list(arg.find(A * (B & C)))
        # possible_args = list(filter(lambda x: isinstance(x, VecMul), possible_args))
        if len(possible_args) > 1:
            combs = list(combinations(possible_args, 2))
            for c in combs:
                if _check_pairs(c):
                    found.add(VecAdd(*c))
    found = list(ordered(list(found)))
    return found

def bac_cab_backward(expr, matches=None):
    """ Implement the replacement rule:
        B (A . C) - C (A . B) = A x (B x C)
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    
    If `matches` is given, replace only the matching subexpressions.
    """
    def _get_abc(match):
        terms = match.args
        # at this point, c1 and c2 should be equal in magnitude
        c1 = Mul(*[a for a in terms[0].args if not hasattr(a, "is_Vector_Scalar")])
        c2 = Mul(*[a for a in terms[1].args if not hasattr(a, "is_Vector_Scalar")])
        n1, _ = c1.as_coeff_mul()
        n2, _ = c2.as_coeff_mul()
        dot1 = [a for a in terms[0].args if isinstance(a, VecDot)][0]
        dot2 = [a for a in terms[1].args if isinstance(a, VecDot)][0]
        A = list(set(dot1.args).intersection(set(dot2.args)))[0]
        get_vector = lambda expr: [t for t in expr.args if t.is_Vector][0]
        if n1 > 0:
            n = c1
            B = get_vector(terms[0])
            C = get_vector(terms[1])
        else:
            n = c2
            B = get_vector(terms[1])
            C = get_vector(terms[0])
        return n, A, B, C
    
    if matches:
        if not isinstance(matches, (list, tuple)):
            matches = [matches]

        for m in matches:
            n, A, B, C = _get_abc(m)
            expr = expr.subs({m: VecMul(n, (A ^ (B ^ C)))})
        return expr
    else:
        matches = find_bac_cab(expr)
        if len(matches) > 0:
            n, A, B, C = _get_abc(matches[0])
            expr = expr.subs({matches[0]: VecMul(n, (A ^ (B ^ C)))})
            expr = bac_cab_backward(expr)
        return expr

def collect(expr, match):
    """ Implement a custom collecting algorithm to collect dot and cross 
    products. There is a noticeable difference with expr.collect: given the
    following expression:
                expr = a * t + b * t + c * t**2 + d * t**3 + e * t**3
                expr.collect(t)
                    t**3 * (d + e) + t**2 * c + t * (a + b)
    Whereas:
                collect(expr, t)
                    t * (a + b + t * c + t**2 * d + t**2 * e)
                collect(expr, t**2)
                    a * t + b * t + t**2 * (c + t * d + t * e)
    
    Parameters
    ----------
        expr : the expression to process
        match : the pattern to collect. If match is not an instance of VecDot,
                VecCross or VecPow, standard expr.collect(match) will be used.
    """
    # the method expr.collect doesn't know how to treat cross and dot products.
    # Since dot-product is a scalar, it can also be raised to a power. We need
    # to select which algorithm to run.
    if not isinstance(match, (VecDot, VecCross, VecPow)):
        return expr.collect(match)
    
    # extract the pattern and exponent from the power term
    pattern = match
    n = 1
    if isinstance(match, VecPow):
        pattern = match.base
        n = match.exp
    
    if not expr.has(pattern):
        return expr
    
    if not isinstance(expr, VecAdd):
        return expr
    
    collect = []
    not_collect = []
    for arg in expr.args:
        if not arg.has(pattern):
            not_collect.append(arg)
        else:
            p = [a for a in arg.args if isinstance(a, VecPow) and isinstance(a.base, pattern.func)]
            if p:
                # vecpow(pattern, exp)
                if p[0].exp >= n:
                    collect.append(arg / match)
                else:
                    not_collect.append(arg)
            else:
                if n == 1:
                    if arg != match:
                        term = arg.func(*[a for a in arg.args if a != match])
                        collect.append(term)
                    elif match.is_Vector:
                        collect.append(1)
                    else:
                        # TODO. need to test this!!!
                        collect.append(VectorOne())

                else:
                    not_collect.append(arg)

    return VecAdd(VecMul(match, VecAdd(*collect)), *not_collect)

def _as_coeff_vector(v):
    """ Similarly to as_coeff_mul(), it extract the multiplier coefficients from
    and the vector part from the VectorExpr.
    """
    if isinstance(v, (VectorSymbol, Magnitude, Normalize)):
        return 1, v
    if isinstance(v, VecAdd):
        # TODO: implement more sofisticated technique to collect common
        # symbols from vector quantities
        v = collect_const(v)
        k, mul = v.as_coeff_mul()
        v = mul[0]
        return k, v
    if isinstance(v, VecMul):
        k = [a for a in v.args if not a.is_Vector]
        v = [a for a in v.args if a.is_Vector]
        # TODO: as of now, hope for the best, that there is only one
        # argument with is_Vector = True
        return VecMul(*k), v[0]
    return 1, v

def _as_coeff_product(expr):
    """ Similarly to as_coeff_mul(), it extract the multiplier coefficients from
    the arguments of the (dot/cross) product. It returns a VecMul where its
    arguments are:
        * the coefficients multiplied togheter
        * the vectors of the (dot/cross) product
    """
    if not isinstance(expr, (VecCross, VecDot)):
        return expr
    k1, a = _as_coeff_vector(expr.args[0])
    k2, b = _as_coeff_vector(expr.args[1])
    return (k1 * k2) * expr.func(a, b)

def _collect_coeff_from_product(expr, pattern):
    """ Collect the coefficients of nested dot (or cross) products. For example:
        (x * A) ^ ((y * B) ^ (z * C)) = (x * y * z) * A ^ (B ^ C)
    """
    found = list(ordered(list(expr.find(pattern))))
    for i, f in enumerate(found):
        newf = _as_coeff_product(f)
        for j in range(i + 1, len(found)):
            found[j] = found[j].xreplace({f: newf})
        expr = expr.xreplace({f: newf})
    return expr

def _terms_with_commong_args(args_list, first_arg=True):
    """ Given a list of arguments, return a list of lists, where each list 
    contains terms with a common argument.
    For example:
    expr = (a ^ b) + (a ^ c) + (b ^ c)
    _terms_with_commong_args(expr.args, True)
        [[a ^ b, a ^ c]]
    _terms_with_commong_args(expr.args, False)
        [[a ^ c, b ^ c]]
    
    Parameters
    ----------
        args_list : a list containing arguments of an expression
        first_arg : Boolean, default to True. If True, look for common first 
            arguments in the list of args. If False, look for common second
            arguments.
    """
    args_list = list(args_list)
    if not args_list:
        return None
    
    idx = 0
    if not first_arg:
        idx = 1

    # select the first argument
    f = args_list[0]
    possible_terms = [f]
    # look in the remaining of the list for other arguments sharing
    # the same first (or second) argument
    for i in range(1, len(args_list)):
        t = args_list[i]
        if f.args[idx] == t.args[idx]:
            possible_terms.append(t)
    
    # remove from the list of arguments all the possible terms
    args_set = set(args_list).difference(set(possible_terms))
    result = []
    if len(possible_terms) > 1:
        # if there are two or more terms sharing the first (or second)
        # arguments, then append them to the result
        result = [possible_terms]
    
    # repeat the procedure in the remaining arguments
    other_results = _terms_with_commong_args(args_set, first_arg)
    if other_results:
        for r in other_results:
            result.append(r)
    return result

def collect_cross_dot(expr, pattern=VecCross, first_arg=True):
    """ Collect additive dot/cross products with common arguments.

    Example: expr = (a ^ b) + (a ^ c) + (b ^ c)
    collect_cross_dot(expr)
        (a ^ (b + c)) + (b ^ c)
    collect_cross_dot(expr, VecCross, False)
        (a ^ b) + ((a + b) ^ c)

    Parameters
    ----------
        expr : the expression to process
        pattern : the class to look for. Can be VecCross or VecDot.
        first_arg : Boolean, default to True. If True, look for common first 
            arguments in the instances of the class `pattern`. Otherwise, look 
            for common second arguments.
    """
    if not issubclass(pattern, DotCross):
        return expr

    copy = expr
    subs_list = []
    for arg in preorder_traversal(expr):
        # we are interested to the args of instance pattern.func in the
        # current tree level
        found = [a for a in arg.args if isinstance(a, pattern)]
        terms = _terms_with_commong_args(found, first_arg)

        if first_arg:
            get_v = lambda t: pattern(t[0].args[0], VecAdd(*[a.args[1] for a in t]))
        else:
            get_v = lambda t: pattern(VecAdd(*[a.args[0] for a in t]), t[0].args[1])
        
        if terms:
            for t in terms:
                subs_list.append({VecAdd(*t) : get_v(t)})
    for s in subs_list:
        expr = expr.subs(s)
    
    if copy == expr:
        return expr
    return collect_cross_dot(expr, pattern, first_arg)

def _dot_to_mag(expr):
    """ Look for dot products of the form (A & A) = A.mag**2 and perform this 
    simplification.
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    found = expr.find(A & A)
    for f in found:
        expr = expr.subs(f, f.args[0].mag**2)
    return expr

def _curl_of_grad(expr):
    w = Wild("w")
    found = expr.find(Nabla() ^ Grad(w))
    for f in found:
        expr = expr.subs(f, VectorZero())
    return expr

def _div_of_curl(expr):
    w = WVS("w")
    found = expr.find(Nabla() & (Nabla() ^ w))
    for f in found:
        expr = expr.subs(f, S.Zero)
    return expr

def simplify(expr, **kwargs):
    """ Apply a few simplification rules to expr.

    **kwargs:
        dot_to_mag : Default to True.
            a & a = a.mag**2
        bac_cab : Default to True.
            b * (a & c) - c * (a & b) = a ^ (b ^ c)
        coeff: Default to True
            (k1 * a) & (((k2 * b) & (k3 * c)) * d) = (k1 * k2 * k3) (a & ((b & c) * d))
            (k1 * a) ^ ((k2 * b) ^ (k3 * c)) = (k1 * k2 * k3) (a ^ (b ^ c))
        dot : Default to True
            (a & b) + (a & c) = a & (b + c)
            (a & b) + (c & b) = (a + c) & b
        cross : Default to True
            (a ^ b) + (a ^ c) = a ^ (b + c)
            (a ^ b) + (c ^ b) = (a + c) ^ b
    """
    if not isinstance(expr, VectorExpr):
        return expr.simplify()
    
    dot_to_mag = kwargs.get('dot_to_mag', True)
    bac_cab = kwargs.get('bac_cab', True)
    coeff = kwargs.get('coeff', True)
    dot = kwargs.get('dot', True)
    cross = kwargs.get('cross', True)
    
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]   

    # Identity H: nabla ^ (Grad(x)) = 0
    expr = _curl_of_grad(expr)
    # Identity I: nabla & (nabla ^ a) = 0
    expr = _div_of_curl(expr)

    ############################################################################
    ###################### RULE 0: Identities H and I ##########################
    ############################################################################
    expr = identities(expr, curl_of_grad=True, div_of_curl=True)

    ############################################################################
    ###################### RULE 1: (v & v) = v.mag**2 ##########################
    ############################################################################
    if dot_to_mag:
        expr = _dot_to_mag(expr)
    
    ############################################################################
    ################ RULE 2: b (a & c) - c (a & b) = a ^ (b ^ c) ###############
    ############################################################################
    if bac_cab:
        matches = find_bac_cab(expr)
        expr = bac_cab_backward(expr, matches)
    
    ############################################################################
    ####################### RULE 3: nested dot products ########################
    # (k1 * a) & (((k2 * b) & (k3 * c)) * d) = (k1 * k2 * k3) (a & ((b & c) * d)) 
    ############################################################################
    if coeff:
        expr = _collect_coeff_from_product(expr, A & B)

    ############################################################################
    ####################### RULE 4: nested cross products ######################
    ###### (k1 * a) ^ ((k2 * b) ^ (k3 * c)) = (k1 * k2 * k3) (a ^ (b ^ c)) #####
    ############################################################################
    if coeff:
        expr = _collect_coeff_from_product(expr, A ^ B)

    ############################################################################
    ####################### RULE 5: collect dot products #######################
    ############################################################################
    if dot:
        expr = collect_cross_dot(expr, VecDot)
        expr = collect_cross_dot(expr, VecDot, False)

    ############################################################################
    ####################### RULE 6: collect cross products #####################
    ############################################################################
    if cross:
        expr = collect_cross_dot(expr)
        expr = collect_cross_dot(expr, VecCross, False)

    # Repeat Rule #1 just in case other simplifications created new occurences
    if dot_to_mag:
        expr = _dot_to_mag(expr)

    return expr

w = Wild("w")
wa = WVS("w_a")
wb = WVS("w_b")
nabla = Nabla()

_id = {
    # Identity C
    "prod_div": [
        nabla & (w * wa),
        (Grad(w) & wa) + (w * (nabla & wa))
    ],
    # Identity D
    "prod_curl": [
        nabla ^ (w * wa),
        (Grad(w) ^ wa) + (w * (nabla ^ wa))
    ],
    # Identity E
    "div_of_cross": [
        nabla & (wa ^ wb),
        ((nabla ^ wa) & wb) - ((nabla ^ wb) & wa)
    ],
    # Identity F
    "curl_of_cross": [
        nabla ^ (wa ^ wb),
        ((nabla & wb) * wa) + DotNablaOp(wb, wa) - ((nabla & wa) * wb) - DotNablaOp(wa, wb)
    ],
    # Identity G
    "grad_of_dot": [
        Grad(wa & wb),
        DotNablaOp(wa, wb) + DotNablaOp(wb, wa) + (wa ^ (nabla ^ wb)) + (wb ^ (nabla ^ wa))
    ],
    # Identity H
    "curl_of_grad": [
        nabla ^ Grad(w),
        VectorZero()
    ],
    # Identity I
    "div_of_curl": [
        nabla & (nabla ^ wa),
        S.Zero
    ],
    # Identity J
    "curl_of_curl": [
        nabla ^ (nabla ^ wa),
        Grad(nabla & wa) - Laplace(wa)
    ],
}

def identities(expr, **hints):
    """ Apply to expr the identities specified in **hints.

    In the following, w is scalar, wa/wb are vectors.

    Hints: set a flag to True to apply the identity. By default all
    flags are set to False.

        prod_div = True
            nabla & (w * wa) = (Grad(w) & wa) + (w * (nabla & wa))
        
        prod_curl = True
            nabla ^ (w * wa) = (Grad(w) ^ wa) + (w * (nabla ^ wa))
        
        div_of_cross = True
            nabla & (wa ^ wb) = ((nabla ^ wa) & wb) - ((nabla ^ wb) & wa)
        
        curl_of_cross = True
            nabla ^ (wa ^ wb) = ((nabla & wb) * wa) + DotNablaOp(wb, wa) - ((nabla & wa) * wb) - DotNablaOp(wa, wb)
        
        grad_of_dot = True
            Grad(wa & wb) = DotNablaOp(wa, wb) + DotNablaOp(wb, wa) + (wa ^ (nabla ^ wb)) + (wb ^ (nabla ^ wa))
        
        curl_of_grad = True
            nabla ^ Grad(w) = VectorZero()

        div_of_curl = True
            nabla & (nabla ^ wa) = S.Zero

        curl_of_curl = True
            nabla ^ (nabla ^ wa) = Grad(nabla & wa) - Laplace(wa)
    """

    for hint, val in hints.items():
        if hint in _id.keys() and (val == True):
            pattern, subs = _id[hint]
            found = list(ordered(list(expr.find(pattern))))
            for i, f in enumerate(found):
                fsubs = subs.xreplace(f.match(pattern))
                # update found list
                for j in range(i + 1, len(found)):
                    found[j] = found[j].subs(f, fsubs)
                expr = expr.subs(f, fsubs)
    return expr

# TODO:
# 1. Nested identities "prod_div" and "prod_curl" don't work.
# 