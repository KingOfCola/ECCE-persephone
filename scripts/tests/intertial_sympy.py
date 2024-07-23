from sympy import symbols, Function, oo, sqrt, integrate, Min


x, alpha, beta = symbols("x alpha beta")
u1, u2, u3 = symbols("u1 u2 u3")
t = symbols("t")


class upsilon(Function):
    def inverse(self, argindex=3):
        return U(x, self.args[1], self.args[2])

    def _eval_simplify(self, ratio, measure, **kwargs):
        inv = self.inverse()
        arg = self.args[0]
        if (
            isinstance(self.args[0], inv.__class__)
            & (inv.args[1] == arg.args[1])
            & (inv.args[2] == arg.args[2])
        ):
            return self.args[0].args[0]
        return self


class U(Function):
    def inverse(self, argindex=3):
        return upsilon(x, self.args[1], self.args[2])

    def _eval_simplify(self, ratio, measure, **kwargs):
        inv = self.inverse()
        arg = self.args[0]
        if (
            isinstance(self.args[0], inv.__class__)
            & (inv.args[1] == arg.args[1])
            & (inv.args[2] == arg.args[2])
        ):
            return self.args[0].args[0]
        return self


def F1(u1, alpha):
    return u1


def F2(u1, u2, alpha):
    return (
        alpha
        / (1 - alpha)
        * integrate(
            F1(Min(u1, t), alpha),
            (
                t,
                ((U(u2, alpha, 1 - alpha)) - (1 - alpha)) / alpha,
                U(u2, alpha, 1 - alpha) / alpha,
            ),
        )
    )


F3 = (
    alpha
    / (1 - alpha)
    * integrate(
        F1.subs(x, Min(u1, t)),
        (
            t,
            ((U(u1, alpha, 1 - alpha)) - (1 - alpha)) / alpha,
            U(u1, alpha, 1 - alpha) / alpha,
        ),
    )
)
