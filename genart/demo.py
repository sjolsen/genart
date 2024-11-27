from genart import program


class DemoProgram(program.Program):
    """Embedded DSL for hard-coding examples."""

    @classmethod
    def coerce(self, other) -> 'DemoProgram':
        if isinstance(other, DemoProgram):
            return other
        elif isinstance(other, int):
            return DemoProgram(program.Constant(other), ())
        else:
            raise TypeError

    def __neg__(self) -> 'DemoProgram':
        return DemoProgram(program.Syntax.NEG, (self,))

    def __invert__(self) -> 'DemoProgram':
        return DemoProgram(program.Syntax.INV, (self,))

    def __mul__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.MUL, (self, self.coerce(other)))

    def __rmul__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.MUL, (self.coerce(other), self))

    def __floordiv__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.DIV, (self, self.coerce(other)))

    def __rfloordiv__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.DIV, (self.coerce(other), self))

    def __mod__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.MOD, (self, self.coerce(other)))

    def __rmod__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.MOD, (self.coerce(other), self))

    def __add__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.ADD, (self, self.coerce(other)))

    def __radd__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.ADD, (self.coerce(other), self))

    def __sub__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.SUB, (self, self.coerce(other)))

    def __rsub__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.SUB, (self.coerce(other), self))

    def __lshift__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.LSH, (self, self.coerce(other)))

    def __rlshift__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.LSH, (self.coerce(other), self))

    def __rshift__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.RSH, (self, self.coerce(other)))

    def __rrshift__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.RSH, (self.coerce(other), self))

    def __and__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.AND, (self, self.coerce(other)))

    def __rand__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.AND, (self.coerce(other), self))

    def __xor__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.XOR, (self, self.coerce(other)))

    def __rxor__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.XOR, (self.coerce(other), self))

    def __or__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.IOR, (self, self.coerce(other)))

    def __ror__(self, other) -> 'DemoProgram':
        return DemoProgram(program.Syntax.IOR, (self.coerce(other), self))


x = DemoProgram(program.Variable(0), ())
y = DemoProgram(program.Variable(1), ())


DEMO1 = x - (-108 | (y % (x - 8)))
DEMO2 = (16 * x + y // x) * (8 + y) + (x >> x) * (~x - 1)
DEMO3 = y % ((2 - x) ^ -(y | x))
DEMO4 = ((18 | y) << (1 & (y >> x))) ^ ((y & x) << (x >> y)) * (x + y)
DEMO5 = -y - (x >> -161 * (x << y))
DEMO6 = ((48 | y) // x) // -(x & y)
DEMO7 = ((~x << 130 // y) + y % 0) << (y * x & -8) % 36

DEMOS: tuple[DemoProgram, ...] = (
    DEMO1,
    DEMO2,
    DEMO3,
    DEMO4,
    DEMO5,
    DEMO6,
    DEMO7,
)
