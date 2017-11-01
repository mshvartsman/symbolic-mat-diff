from sympy import (Symbol, MatrixSymbol, ZeroMatrix, Add, Mul, MatAdd, MatMul,
                   Determinant, Inverse, Trace, Transpose)
from .symbols import d, Kron, SymmetricMatrixSymbol
from .simplifications import simplify_matdiff

MATRIX_DIFF_RULES = {
    # e =expression, s = a list of symbols respsect to which
    # we want to differentiate
    Symbol: lambda e, s: d(e) if (e in s) else 0,
    MatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    SymmetricMatrixSymbol: lambda e, s: d(e) if (e in s) else ZeroMatrix(*e.shape),
    Add: lambda e, s: Add(*[_matDiff_apply(arg, s) for arg in e.args]),
    Mul: lambda e, s: _matDiff_apply(e.args[0], s) if len(e.args)==1 else Mul(_matDiff_apply(e.args[0],s),Mul(*e.args[1:])) + Mul(e.args[0], _matDiff_apply(Mul(*e.args[1:]),s)),
    MatAdd: lambda e, s: MatAdd(*[_matDiff_apply(arg, s) for arg in e.args]),
    MatMul: lambda e, s: _matDiff_apply(e.args[0], s) if len(e.args)==1 else MatMul(_matDiff_apply(e.args[0],s),MatMul(*e.args[1:])) + MatMul(e.args[0], _matDiff_apply(MatMul(*e.args[1:]),s)),
    Kron: lambda e, s: _matDiff_apply(e.args[0],s) if len(e.args)==1 else Kron(_matDiff_apply(e.args[0],s),Kron(*e.args[1:]))
                  + Kron(e.args[0],_matDiff_apply(Kron(*e.args[1:]),s)),
    Determinant: lambda e, s: MatMul(Determinant(e.args[0]), Trace(e.args[0].I*_matDiff_apply(e.args[0], s))), 
    # inverse always has 1 arg, so we index
    Inverse: lambda e, s: -Inverse(e.args[0]) * _matDiff_apply(e.args[0], s) * Inverse(e.args[0]),
    # trace always has 1 arg
    Trace: lambda e, s: Trace(_matDiff_apply(e.args[0], s)),
    # transpose also always has 1 arg, index
    Transpose: lambda e, s: Transpose(_matDiff_apply(e.args[0], s))
    }


def _matDiff_apply(expr, syms):
    if expr.__class__ in list(MATRIX_DIFF_RULES.keys()):
        return MATRIX_DIFF_RULES[expr.__class__](expr, syms)
    elif expr.is_constant():
        return 0
    else:
        raise TypeError("Don't know how to differentiate class %s", expr.__class__) 


def _diff_to_grad(expr, s):
    # if expr is a trace, sum of traces, or scalar times a trace, we can do it
    # scalar times a trace 
    if expr.is_Mul and expr.args[0].is_constant() and expr.args[1].is_Trace:
        return (expr.args[0] * expr.args[1].arg.xreplace({d(s): 1}))
    else: 
        raise RuntimeError("Don't know how to convert %s to gradient!" % expr)

def matDiff(expr, syms):
    # diff wrt 1 element wrap in list
    try:
        _ = syms.__iter__
    except AttributeError:
        syms = [syms]

    def diff_and_simplify(expr, s):
        expr = _matDiff_apply(expr, [s])
        expr = simplify_matdiff(expr, d(s))
        return expr

    return [diff_and_simplify(expr, s).doit() for s in syms]



def matGrad(expr, syms):
    """Compute matrix gradient by matrix differentiation
    """
    diff = matDiff(expr, syms)
    grad = [_diff_to_grad(e, s) for e, s in zip(diff, syms)]
    return grad
