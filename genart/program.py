import abc
import contextlib
import dataclasses
import math
import operator
import random
from typing import Callable, Optional

import numpy


class ComputeNode(abc.ABC):
    """Base class for randomly generated compute nodes."""
    precedence: int
    shape: int

    @abc.abstractmethod
    def pretty_print(self, children: list[str]) -> str:
        pass

    @abc.abstractmethod
    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        pass


@dataclasses.dataclass(frozen=True)
class Constant(ComputeNode):
    value: int
    precedence: int = 0
    shape: int = 0

    def pretty_print(self, children: list[str]) -> str:
        return str(self.value)

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        return numpy.full(xy.shape[1:], self.value)


@dataclasses.dataclass(frozen=True)
class Variable(ComputeNode):
    index: int
    precedence: int = 0
    shape: int = 0

    def pretty_print(self, children: list[str]) -> str:
        return 'xy'[self.index]

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        return xy[self.index]


@dataclasses.dataclass(frozen=True)
class UnaryOperator(ComputeNode):
    sigil: str
    function: Callable[[numpy.ndarray], numpy.ndarray]
    precedence: int = 1
    shape: int = 1

    def pretty_print(self, children: list[str]) -> str:
        [a] = children
        return f'{self.sigil}{a}'

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        [a] = inputs
        return self.function(a)


@dataclasses.dataclass(frozen=True)
class BinaryOperator(ComputeNode):
    sigil: str
    function: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
    precedence: int
    shape: int = 2

    def pretty_print(self, children: list[str]) -> str:
        [a, b] = children
        return f'{a} {self.sigil} {b}'

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        [a, b] = inputs
        return self.function(a, b)


class ExecutionError(Exception):
    pass


@contextlib.contextmanager
def execution_guard():
    """Handle numeric errors."""
    with numpy.errstate(all='ignore'):
        try:
            return (yield)
        except (ArithmeticError, ValueError) as e:
            raise ExecutionError from e


@dataclasses.dataclass(frozen=True)
class Program:
    node: ComputeNode
    children: tuple['Program', ...]

    def __str__(self) -> str:
        children: list[str] = []
        for c in self.children:
            if c.node.precedence < self.node.precedence:
                children.append(str(c))
            else:
                children.append(f'({c})')

        return self.node.pretty_print(children)

    def run(self, xy: numpy.ndarray) -> numpy.ndarray:
        state: dict['Program', numpy.ndarray] = {}
        stack = [self]

        while stack:
            top = stack[-1]
            pending = [c for c in top.children if c not in state]
            if pending:
                stack.extend(pending)
            else:
                inputs = [state[c] for c in top.children]
                with execution_guard():
                    state[top] = top.node.compute(xy, inputs)
                stack.pop()

        return state[self]


class Syntax:
    """Namespace to make match work."""
    X = Variable(0)
    Y = Variable(1)

    NEG = UnaryOperator('-', operator.neg)
    INV = UnaryOperator('~', operator.invert)

    MUL = BinaryOperator('*', operator.mul, 2)
    DIV = BinaryOperator('//', operator.floordiv, 2)
    MOD = BinaryOperator('%', operator.mod, 2)

    ADD = BinaryOperator('+', operator.add, 3)
    SUB = BinaryOperator('-', operator.sub, 3)

    # Collapse bitwise operators into the same precedence as +/- for readability
    LSH = BinaryOperator('<<', operator.lshift, 3)
    RSH = BinaryOperator('>>', operator.rshift, 3)
    AND = BinaryOperator('&', operator.and_, 3)
    XOR = BinaryOperator('^', operator.xor, 3)
    IOR = BinaryOperator('|', operator.or_, 3)


def simplify(p: Program) -> Program:
    while (q := simplify_step(p)) != p:
        p = q
    return q


def simplify_step(p: Program) -> Program:
    node = p.node
    children = tuple(simplify(c) for c in p.children)

    # Constant folding
    match node:
        case UnaryOperator(function=op):
            [a] = children
            match a.node:
                case Constant(n):
                    return Program(Constant(op(n)), ())

        case BinaryOperator(function=op):
            [a, b] = children
            match [a.node, b.node]:
                case [Constant(m), Constant(n)]:
                    return Program(Constant(op(m, n)), ())

    # Algebraic identities
    match node:
        case Syntax.NEG:
            [a] = children
            match a.node:
                # -0 = 0
                case Constant(0):
                    return a
                # -(-a) = a
                case Syntax.NEG:
                    [aa] = a.children
                    return aa

        case Syntax.INV:
            [a] = children
            match a.node:
                # ~(~a) = a
                case Syntax.INV:
                    [aa] = a.children
                    return aa

        case Syntax.MUL:
            [a, b] = children
            match a.node:
                # 0 * b = b
                case Constant(0):
                    return a
                # 1 * b = b
                case Constant(1):
                    return b
                # -1 * b = -b
                case Constant(1):
                    return Program(Syntax.NEG, (b,))
            match b.node:
                # a * 0 = 0
                case Constant(0):
                    return b
                # a * 1 = a
                case Constant(1):
                    return a
                # a * -1 = -a
                case Constant(1):
                    return Program(Syntax.NEG, (a,))

        case Syntax.DIV:
            [a, b] = children
            match a.node:
                # 0 // b = 0
                case Constant(0):
                    return a

        case Syntax.MOD:
            [a, b] = children
            match a.node:
                # 0 % b = 0
                case Constant(0):
                    return a

        case Syntax.ADD:
            [a, b] = children
            match a.node:
                # 0 + b = b
                case Constant(0):
                    return b
            match b.node:
                # a + 0 = a
                case Constant(0):
                    return a
                # a + -b = a - b
                case Syntax.NEG:
                    [bb] = b.children
                    return Program(Syntax.SUB, (a, bb))
            # a + a == 2 * a
            if a == b:
                two = Program(Constant(2), ())
                return Program(Syntax.MUL, (two, a))

        case Syntax.SUB:
            [a, b] = children
            match a.node:
                # 0 - b = -b
                case Constant(0):
                    return Program(Syntax.NEG, (b,))
            match b.node:
                # a - 0 = a
                case Constant(0):
                    return a
                # a - -b = a + b
                case Syntax.NEG:
                    [bb] = b.children
                    return Program(Syntax.ADD, (a, bb))
            # a - a == 0
            if a == b:
                return Program(Constant(0), ())

        case Syntax.LSH:
            [a, b] = children
            match a.node:
                # 0 << b = 0
                case Constant(0):
                    return a
            match b.node:
                # a << 0 = a
                case Constant(0):
                    return a

        case Syntax.RSH:
            [a, b] = children
            match a.node:
                # 0 >> b = 0
                case Constant(0):
                    return a
            match b.node:
                # a >> 0 = a
                case Constant(0):
                    return a

        case Syntax.AND:
            [a, b] = children
            match a.node:
                # 0 & b = 0
                case Constant(0):
                    return a
            match b.node:
                # a & 0 = 0
                case Constant(0):
                    return b
            # a & a = a
            if a == b:
                return a

        case Syntax.XOR:
            [a, b] = children
            match a.node:
                # 0 ^ b = b
                case Constant(0):
                    return b
            match b.node:
                # a ^ 0 = a
                case Constant(0):
                    return a
            # a ^ a = 0
            if a == b:
                return Program(Constant(0), ())

        case Syntax.IOR:
            [a, b] = children
            match a.node:
                # 0 | b = b
                case Constant(0):
                    return b
            match b.node:
                # a | 0 = a
                case Constant(0):
                    return a
            # a | a = a
            if a == b:
                return a

    return Program(node, children)


def free_variables(p: Program) -> frozenset[Variable]:
    children = frozenset().union(*(free_variables(c) for c in p.children))
    if isinstance(p.node, Variable):
        return children | frozenset([p.node])
    else:
        return children


RANDOM_LEAVES: tuple[Optional[ComputeNode], ...] = (
    None,  # Constant
    Syntax.X, Syntax.Y,
)

RANDOM_NODES: tuple[Optional[ComputeNode], ...] = RANDOM_LEAVES + (
    # Unary operators
    Syntax.NEG, Syntax.INV,
    # Syntax.Binary operators
    Syntax.MUL, Syntax.DIV, Syntax.MOD,
    Syntax.ADD, Syntax.SUB,
    Syntax.LSH, Syntax.RSH,
    Syntax.AND, Syntax.XOR, Syntax.IOR,
)


def random_node(max_depth: int) -> ComputeNode:
    if max_depth > 0:
        result = random.choice(RANDOM_NODES)
    else:
        result = random.choice(RANDOM_LEAVES)

    if result:
        return result
    else:
        # Choose from [0, 256)
        log = 1.0 - random.random()
        value = math.floor(256 ** log) - 1
        return Constant(value)


def random_program(max_depth: int) -> Program:
    node = random_node(max_depth)
    children = tuple(random_program(max_depth - 1) for _ in range(node.shape))
    return Program(node, children)


def generate_program(max_depth: int = 4) -> Program:
    all_vars = frozenset([Syntax.X, Syntax.Y])
    while True:
        p = random_program(max_depth)
        try:
            with execution_guard():
                p = simplify(p)
        except ExecutionError:
            continue
        if all_vars <= free_variables(p):
            return p
