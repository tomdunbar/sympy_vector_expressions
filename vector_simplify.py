from itertools import combinations
from sympy import (
    Mul, Add, Abs, preorder_traversal, collect_const
)
from sympy.utilities.iterables import ordered
from vector_expr import (
    VectorExpr, VecAdd, VecMul, VecPow, VecDot, VecCross, 
    Magnitude, Normalize, WVS, VectorSymbol, VectorOne, DotCross
)

# TODO: 
# 1. bac_cab_forward, dot_cross are identical in structure. Bring them 
#    together
# 2. Create test for the method with and without parameter 'match'
# 3. Test dot-cross products expansions:
#    (a ^ (b ^ (c + d))) = a ^ (b ^ c) + a ^ (b ^ d)
# 4. expr = a.mag * (a ^ b) + a.mag * (a ^ c)
#    simplify(expr)
#    a.mag * (a ^ (b + c))
# 5. dot_cross backward/forward
# 6. bac_cab_forward, dot_cross: use 'matches' instead of 'match', thus a 
#    for loop

def bac_cab_forward(expr, match=None):
    """ Implement the replacement rule:
    A x (B x C) = B (A . C) - C (A . B)
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    pattern = A ^ (B ^ C)
    rep = (B * (A & C)) - (C * (A & B))
    if not match:
        # list of matches ordered accordingly to the length of the match.
        # The shorter the match expression, the lower it should be in the
        # expression tree. Substitute from the bottom up!
        found = list(ordered(list(expr.find(pattern))))
        if len(found) > 0:
            f = found[0]
            expr = expr.xreplace({f: rep.xreplace(f.match(pattern))})
            # repeat the procedure with the updated expression
            expr = bac_cab_forward(expr)   
        return expr
    return expr.xreplace({match: rep.xreplace(match.match(pattern))})

def dot_cross(expr, match=None):
    """ Implement the replacement rule:
    A . (B x C) = (A x B) . C
    where:
        A, B, C are vectors;
        x represents the cross product
        . represents the dot product
    """
    A, B, C = [WVS(t) for t in ["A", "B", "C"]]
    pattern = A & (B ^ C)
    rep = (A ^ B) & C
    if not match:
        found = list(ordered(list(expr.find(pattern))))
        if len(found) > 0:
            f = found[0]
            expr = expr.xreplace({f: rep.xreplace(f.match(pattern))})
            expr = dot_cross(expr)   
        return expr
    return expr.xreplace({match: rep.xreplace(match.match(pattern))})

def find_triple_cross_product(expr):
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