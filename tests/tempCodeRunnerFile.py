t_vector_symbols(self):
        assert isinstance(self.v1.diff(x), D)
        assert isinstance(self.v1.diff(x, 2), D)
        assert self.v1.diff(x, 2).args[0].variable_count[0][1] == 2
        assert isinstance(self.one.diff(x), VectorZero)
        assert isinstance(self.zero.diff(x), VectorZero)

        with self.assertRaises(NotImplementedError) as context:
            self.nabla.diff(x)
        
        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            self.v1.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            self.one.diff(x, y)
        with self.assertRaises(NotImplementedError) as context:
            self.zero.diff(x, y)

    def test_magnitude(self):
        expr = self.v1.mag
        assert isinstance(expr.diff(x), D)
        assert isinstance(expr.diff(x, evaluate=False), D)

        # derivatives of magnitude of vector one
        assert isinstance(self.one.diff(x, evaluate=False), D)
        assert self.one.diff(x) == VectorZero()

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            self.v1.mag.diff(x, y)

    def test_vecadd(self):
        expr = self.v1 + self.v2
        dexpr = expr.diff(x)
        assert isinstance(dexpr, VecAdd)
        assert self._check_args(
            dexpr,
            self.v1.diff(x) + self.v2.diff(x)
        )
        assert isinstance(expr.diff(x, evaluate=False), D)
        dexpr = expr.diff(x, 3)
        assert isinstance(dexpr, VecAdd)
        assert dexpr == self.v1.diff(x, 3) + self.v2.diff(x, 3)
        dexpr = expr.diff(x, 3, evaluate=False)
        assert isinstance(dexpr, D)
        assert dexpr.args[0].variable_count[0][1] == 3

        # partial differentiation not implemented
        with self.assertRaises(NotImplementedError) as context:
            expr