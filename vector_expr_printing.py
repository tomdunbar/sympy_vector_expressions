import sympy as sp
from sympy.core.function import _coeff_isneg
from sympy.printing.latex import LatexPrinter, translate
from sympy.printing.precedence import precedence_traditional, PRECEDENCE
from sympy.printing.conventions import split_super_sub, requires_partial
from vector_expr import (
    VecAdd, VecCross, VecDot, VecMul, VecPow, VectorExpr,
    Magnitude, Normalize, VectorSymbol, VectorOne, VectorZero, D
)
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
    
    def _print_VecAdd(self, expr):
        # print("_print_VecAdd", expr)

        # NOTE: only needed if VecAdd doesn't derive from sp.Add
        # return self._print_Add(sp.Add(*expr.args))
        terms = list(expr.args)
        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif _coeff_isneg(term):
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex
        # print("VecAdd output", tex)
        return tex

    # def _print_VecAdd(self, expr):
    #     print("_print_VecAdd", type(expr), expr)
    #     # NOTE: only needed if VecAdd doesn't derive from sp.Add
    #     return self._print_Add(sp.Add(*expr.args))

    def _print_VecDot(self, expr):
        # print("_print_VecDot", type(expr), expr)
        expr1, expr2 = expr.args
        s1 = _wrap(self, expr1, expr)
        s2 = _wrap(self, expr2, expr)
        return r"%s \cdot %s" % (s1, s2)
    
    def _print_VecCross(self, expr):
        # print("_print_VecCross", type(expr), expr)
        expr1, expr2 = expr.args
        s1 = _wrap(self, expr1, expr)
        s2 = _wrap(self, expr2, expr)
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

    # def _print_VecMul(self, expr):
    #     print("_print_VecMul", expr)

    #     parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
    #                                          False)
    #     args = expr.args
    #     if isinstance(args[0], sp.Mul):
    #         args = args[0].as_ordered_factors() + list(args[1:])
    #     else:
    #         args = list(args)

    #     if isinstance(expr, VecMul) and _coeff_isneg(expr):
    #         if args[0] == -1:
    #             args = args[1:]
    #         else:
    #             args[0] = -args[0]
    #         return '- ' + ' '.join(map(parens, args))
    #     else:
    #         return ' '.join(map(parens, args))
    
    def _print_VecMul(self, expr):
        # print("_print_VecMul", type(expr), len(expr.args), expr.args)
        # parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
        #                                      False)
        parens = lambda x: self.parenthesize(x, PRECEDENCE["Mul"])
        
        args = list(expr.args)
        if len(args) == 0:
            return ""

        def asd(args):
            is_neg = False
            for i, a in enumerate(args):
                if a.is_number and a.is_negative:
                    args[i] = sp.Abs(a)
                    is_neg = not is_neg
            return args, is_neg

        is_neg = False
        # print("_print_Mul", args)
        # args, is_neg = asd(args)
        # print("_print_Mul", args, is_neg)
        # tex = "- "


        args_latex = []

        for a in args:
            if isinstance(a, VecAdd):
                s = self._print_VecAdd(a)
            elif isinstance(a, VecCross):
                s = self._print_VecCross(a)
            elif isinstance(a, VecDot):
                s = self._print_VecDot(a)
            else:
                # if a == sp.S.One:
                #     s = ""
                # else:
                #     s = parens(a)
                s = parens(a)



            # print("_print_VecMul ARG", type(a), a, s)
            if isinstance(a, (VecDot, VecCross, VecAdd)):
                s = r"\left(%s\right)" % s
            args_latex.append(s)
        
        # print("_print_VecMul out:", ' '.join(args_latex))
        if is_neg:
            return "- " + ' '.join(args_latex)
        return ' '.join(args_latex)
        # if isinstance(expr, VecMul) and _coeff_isneg(expr):
        #     if args[0] == -1:
        #         args = args[1:]
        #     else:
        #         args[0] = -args[0]
        #     return '- ' + ' '.join(map(parens, args))
        # else:
        #     return ' '.join(map(parens, args))
        
    def _print_VecPow(self, expr):
        base, exp = expr.base, expr.exp
        if exp == sp.S.NegativeOne:
            return r"\frac{1}{%s}" % self.parenthesize(base, precedence_traditional(expr), True)
        else:
            return r"\left(%s\right)^{%s}" % (
                self.parenthesize(base, precedence_traditional(expr), True),
                self.parenthesize(exp, precedence_traditional(expr), True),
            )
    
    def _print_D(self, expr):
        # D is just a wrapper class, doesn't need any rendering
        return self._print_Derivative(expr.args[0])
    
    def _print_Derivative(self, expr):
        # kind of an hack to wrap VectorExpression instances in parentheses
        latex = super()._print_Derivative(expr)
        expr_latex = None
        if isinstance(expr.expr, VectorExpr):
            expr_latex = self._print(expr.expr)
        if expr_latex:
            expr_wrapped = r"\left(%s\right)" % expr_latex
            latex = latex.replace(expr_latex, expr_wrapped)
        return latex

def _wrap(printer, e, expr):
    s = printer.parenthesize(e, PRECEDENCE['Mul'])
    # TODO: need to check the type of e to apply the correct need_brackets
    if not isinstance(e, VectorSymbol):
        if (e.is_Add and printer._needs_add_brackets(e) or
            e.is_Mul and printer._needs_mul_brackets(e) ):
            s = r"\left( %s \right)" % s
    return s