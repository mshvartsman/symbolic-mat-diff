from sympy import MatrixExpr, Basic, sympify, MatrixSymbol, Inverse, Function


class d(MatrixExpr):
    """Unevaluated matrix differential (e.g. dX, where X is a matrix)

    """

    def __new__(cls, mat):
        mat = sympify(mat)

        if not mat.is_Matrix:
            raise TypeError("input to matrix derivative, %s, is not a matrix" % str(mat))

        return Basic.__new__(cls, mat)

    @property
    def arg(self):
        return self.args[0]

    @property
    def shape(self):
        return (self.arg.rows, self.arg.cols)


class SymmetricMatrixSymbol(MatrixSymbol):
    """Symmetric matrix
    """
    is_Symmetric = True

    def _eval_transpose(self):
        return self

    def _eval_inverse(self):
        inv = Inverse(self)
        inv.is_Symmetric = True
        return inv

Kron = Function("Kron",commutative=False)