import abc
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
                state[top] = top.node.compute(xy, inputs)
                stack.pop()

        return state[self]


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


RANDOM_LEAVES: tuple[Optional[ComputeNode], ...] = (
    None, # Constant
    X, Y,
)

RANDOM_NODES: tuple[Optional[ComputeNode], ...] = RANDOM_LEAVES + (
    # Unary operators
    NEG, INV,
    # Binary operators
    MUL, DIV, MOD,
    ADD, SUB,
    LSH, RSH,
    AND, XOR, IOR,
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


def random_program(max_depth: int = 4) -> Program:
    node = random_node(max_depth)
    children = tuple(random_program(max_depth - 1) for _ in range(node.shape))
    return Program(node, children)


def demo() -> Program:
    """Returns the program [x - (-108 | (y % (x - 8)))]."""
    x = Program(Variable(0), ())
    y = Program(Variable(1), ())
    const = lambda c: Program(Constant(c), ())
    binop = lambda op: lambda a, b: Program(op, (a, b))
    sub = binop(SUB)
    or_ = binop(IOR)
    mod = binop(MOD)
    return sub(x, or_(const(-108), mod(y, (sub(x, const(8))))))
