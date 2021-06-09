class MatrixKind(Kind):
    """
    Kind for all matrices in SymPy.
    Basic class for this kind is ``MatrixBase`` and ``MatrixExpr``,
    but any expression representing the matrix can have this.
    Parameters
    ==========
    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the matrix contains only numbers.
    Examples
    ========
    Any instance of matrix class has ``MatrixKind``.
    >>> from sympy import MatrixSymbol
    >>> A = MatrixSymbol('A', 2,2)
    >>> A.kind
    MatrixKind(NumberKind)
    Although expression representing a matrix may be not instance of
    matrix class, it will have ``MatrixKind`` as well.
    >>> from sympy import Integral
    >>> from sympy.matrices.expressions import MatrixExpr
    >>> from sympy.abc import x
    >>> intM = Integral(A, x)
    >>> isinstance(intM, MatrixExpr)
    False
    >>> intM.kind
    MatrixKind(NumberKind)
    Use ``isinstance()`` to check for ``MatrixKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.
    >>> from sympy import Matrix
    >>> from sympy.matrices import MatrixKind
    >>> from sympy.core.kind import NumberKind
    >>> M = Matrix([1, 2])
    >>> isinstance(M.kind, MatrixKind)
    True
    >>> M.kind is MatrixKind(NumberKind)
    True
    See Also
    ========
    shape : Function to return the shape of objects with ``MatrixKind``.
    """
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    def __repr__(self):
        return "MatrixKind(%s)" % self.element_kind


def _matrixify(mat):
    """If `mat` is a Matrix or is matrix-like,
    return a Matrix or MatrixWrapper object.  Otherwise
    `mat` is passed through without modification."""

    if getattr(mat, 'is_Matrix', False) or getattr(mat, 'is_MatrixLike', False):
        return mat

    if not(getattr(mat, 'is_Matrix', True) or getattr(mat, 'is_MatrixLike', True)):
        return mat

    shape = None

    if hasattr(mat, 'shape'): # numpy, scipy.sparse
        if len(mat.shape) == 2:
            shape = mat.shape
    elif hasattr(mat, 'rows') and hasattr(mat, 'cols'): # mpmath
        shape = (mat.rows, mat.cols)

    if shape:
        return _MatrixWrapper(mat, shape)

    return mat


def a2idx(j, n=None):
    """Return integer after making positive and validating against n."""
    if type(j) is not int:
        jindex = getattr(j, '__index__', None)
        if jindex is not None:
            j = jindex()
        else:
            raise IndexError("Invalid index a[%r]" % (j,))
    if n is not None:
        if j < 0:
            j += n
        if not (j >= 0 and j < n):
            raise IndexError("Index out of range: a[%s]" % (j,))
    return int(j)


def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.
    Currently the strategy is that immutability is contagious.
    Examples
    ========
    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.common import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__

    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__

    raise TypeError("Incompatible classes %s, %s" % (A.__class__, B.__class__))