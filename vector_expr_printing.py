import sympy as sp
from sympy.core.function import _coeff_isneg
from sympy.printing.latex import LatexPrinter, translate, _between_two_numbers_p
from sympy.printing.precedence import precedence_traditional, PRECEDENCE
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.core.compatibility import Iterable
from vector_expr import (
    VecAdd, VecCross, VecDot, VecMul, VecPow, VectorExpr,
    Magnitude, Normalize, VectorSymbol, VectorOne, VectorZero, D
)

# TODO:
# 1. _print_Derivative prints partial symbol, for example:
#       (a ^ b).diff(x, evaluate=False)
#

# adapted from sympy.printing.latex
def vector_latex(expr, fold_frac_powers=False, fold_func_brackets=False,
          fold_short_frac=None, inv_trig_style="abbreviated",
          itex=False, ln_notation=False, long_frac_ratio=None,
          mat_delim="[", mat_str=None, mode="plain", mul_symbol=None,
          order=None, symbol_names=None, root_notation=True,
          mat_symbol_style="plain", imaginary_unit="i", gothic_re_im=False,
          decimal_separator="period", vec_symbol=r"\vec{%s}", unit_vec_symbol=r"\hat{%s}",
            normalize_style="frac"):
    if symbol_names is None:
        symbol_names = {}

    settings = {
        'fold_frac_powers': fold_frac_powers,
        'fold_func_brackets': fold_func_brackets,
        'fold_short_frac': fold_short_frac,
        'inv_trig_style': inv_trig_style,
        'itex': itex,
        'ln_notation': ln_notation,
        'long_frac_ratio': long_frac_ratio,
        'mat_delim': mat_delim,
        'mat_str': mat_str,
        'mode': mode,
        'mul_symbol': mul_symbol,
        'order': order,
        'symbol_names': symbol_names,
        'root_notation': root_notation,
        'mat_symbol_style': mat_symbol_style,
        'imaginary_unit': imaginary_unit,
        'gothic_re_im': gothic_re_im,
        'decimal_separator': decimal_separator,
        # added the following settings
        'vec_symbol': vec_symbol,
        'unit_vec_symbol': unit_vec_symbol,
        'normalize_style': normalize_style,
    }

    return MyLatexPrinter(settings).doprint(expr)

class MyLatexPrinter(LatexPrinter):
    _default_settings = {
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "order": None,
        "symbol_names": {},
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "gothic_re_im": False,
        "decimal_separator": "period",
        'vec_symbol': r"\vec{%s}",
        'unit_vec_symbol': r"\hat{%s}",
        'normalize_style': "frac",
    }
    
    def __init__(self, settings=None):
        # I need to call super because I added more settings to the class,
        # so that it will update _setting with all the entries of 
        # this class' _default_settings.
        super().__init__(settings)
    
    def _print_VectorSymbol(self, expr, vec_symbol=None):
        # print("_print_VecSymbol", type(expr), expr)

        # this method is used to print:
        # 1. VectorSymbol instances, that uses the vec_symbol provided in the settings.
        # 2. The unit vector symbol (normalized vector), that uses uni_vec_symbol. In this
        #    case, the expression in a string representing the expression given into 
        #    _print_Normalize()
        if expr in self._settings['symbol_names']:
            return self._settings['symbol_names'][expr]
        if hasattr(expr, "name"):
            # this has been adapted from _deal_with_super_sub
            string = expr.name
            result = expr.name
            if not '{' in string and string != r"\nabla":
                name, supers, subs = split_super_sub(string)

                name = translate(name)
                supers = [translate(sup) for sup in supers]
                subs = [translate(sub) for sub in subs]
                if vec_symbol is None:
                    if expr._vec_symbol == "":
                        vec_symbol = self._settings["vec_symbol"]
                    else:
                        vec_symbol = expr._vec_symbol

                result = vec_symbol % name
                # glue all items together:
                if supers:
                    result += "^{%s}" % " ".join(supers)
                if subs:
                    result += "_{%s}" % " ".join(subs)

            if expr._bold:
                result = r"\mathbf{{{}}}".format(result)
            if expr._italic:
                result = r"\mathit{%s}" % result

            return result
        return vec_symbol % expr

    def _print_VecDot(self, expr):
        expr1, expr2 = expr.args
        s1 = _wrap_cross_dot_arg(self, expr1)
        s2 = _wrap_cross_dot_arg(self, expr2)
        return r"%s \cdot %s" % (s1, s2)

    def _print_VecCross(self, expr):
        expr1, expr2 = expr.args
        s1 = _wrap_cross_dot_arg(self, expr1)
        s2 = _wrap_cross_dot_arg(self, expr2)
        return r"%s \times %s" % (s1, s2)

    def _print_Normalize(self, expr):
        v = expr.args[0]
        style = self._settings["normalize_style"]
        if style == "frac":
            return r"\frac{%s}{\|%s\|}" % (self.parenthesize(v, precedence_traditional(expr), True),
                               self.parenthesize(v, precedence_traditional(expr), True))
        unit_vec_symbol = v._unit_vec_symbol
        if unit_vec_symbol == "":
            unit_vec_symbol = self._settings["unit_vec_symbol"]
        # print the symbol as a unit vector
        return self._print_VectorSymbol(self.parenthesize(v, precedence_traditional(expr), True), vec_symbol=unit_vec_symbol)
        
    def _print_Magnitude(self, expr):
        v = expr.args[0]
        return r"\|%s\|" % self.parenthesize(v, precedence_traditional(expr), True)
        
    def _print_VecPow(self, expr):
        base, exp = expr.base, expr.exp
        if exp == sp.S.NegativeOne:
            return r"\frac{1}{%s}" % self._print(base)
        else:
            if not isinstance(base, VectorExpr):
                return self._helper_print_standard_power(expr, "%s^{%s}")

            base_str = r"\left(%s\right)^{%s}"
            if isinstance(base, Magnitude):
                base_str = "%s^{%s}"
            return base_str % (
                self._print(base),
                self.parenthesize(exp, precedence_traditional(expr), True),
            )
    
    def _print_D(self, expr):
        # D is just a wrapper class, doesn't need any rendering
        return self._print_Derivative(expr.args[0])
    
    def _print_Derivative(self, expr):
        if upgraded_requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))
        
        put_into_num = False
        if (isinstance(expr.expr, VectorSymbol) or 
            (isinstance(expr.expr, Magnitude) and 
            isinstance(expr.expr.args[0], VectorSymbol))):
            put_into_num = True
            latex_expr = self._print(expr.expr)
        elif isinstance(expr.expr, (VecDot, VecCross, VecMul)):
            latex_expr = r"\left(%s\right)" % self._print(expr.expr)
        else:
            latex_expr = self.parenthesize(
            expr.expr,
            PRECEDENCE["Mul"],
            strict=True
        )

        if dim == 1:
            if put_into_num:
                return r"\frac{%s %s}{%s}" % (diff_symbol, latex_expr, tex)
            return r"\frac{%s}{%s} %s" % (diff_symbol, tex, latex_expr)
        else:
            if put_into_num:
                return r"\frac{%s^{%s} %s}{%s}" % (diff_symbol, self._print(dim), latex_expr, tex)
            return r"\frac{%s^{%s}}{%s} %s" % (diff_symbol, self._print(dim), tex, latex_expr)

    def _print_VecMul(self, expr):
        # Almost identical to _print_Mul
        from sympy.core.power import Pow
        from sympy.physics.units import Quantity
        include_parens = False
        if _coeff_isneg(expr):
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        from sympy.simplify import fraction
        numer, denom = fraction(expr, exact=True)
        separator = self._settings['mul_symbol_latex']
        numbersep = self._settings['mul_symbol_latex_numbers']

        def convert(expr):
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                _tex = last_term_tex = ""

                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If quantities are present append them at the back
                args = sorted(args, key=lambda x: isinstance(x, Quantity) or
                              (isinstance(x, Pow) and
                               isinstance(x.base, Quantity)))

                for i, term in enumerate(args):
                    term_tex = self._print(term)

                    # NOTE: only difference wrt _print_Mul
                    if (self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1))
                        or isinstance(term, (VecCross, VecDot))):
                        term_tex = r"\left(%s\right)" % term_tex

                    if _between_two_numbers_p[0].search(last_term_tex) and \
                            _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator

                    _tex += term_tex
                    last_term_tex = term_tex
                return _tex

        if denom is sp.S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] and ldenom <= 2 and \
                    "^" not in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = sp.S.One
                    b = sp.S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex

    def _print_WildVectorSymbol(self, expr):
        return self._print_VectorSymbol(expr)

def _wrap_cross_dot_arg(printer, expr):
    s = printer._print(expr)
    wrap = False
    if isinstance(expr, D) and isinstance(expr.args[0].expr, VectorSymbol):
        wrap = False
    elif not isinstance(expr, VectorSymbol):
        wrap = True
    
    if wrap:
        s = r"\left(%s\right)" % s
    return s

def upgraded_requires_partial(expr):
    """Return whether a partial derivative symbol is required for printing
    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions don't have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    """
    
    # TODO: because at the moment I only implemented univariate derivative,
    # I can return False. Once partial derivatives are implemented, need to
    # figure out a way to return the correct value.
    if isinstance(expr, VectorExpr):
        return False

    if isinstance(expr, sp.Derivative):
        return upgraded_requires_partial(expr.expr)

    if not isinstance(expr.free_symbols, Iterable):
        return len(set(expr.variables)) > 1

    return sum(not s.is_integer for s in expr.free_symbols) > 1