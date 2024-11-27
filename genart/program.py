import abc
import dataclasses
import math
import operator
import random
from typing import Callable, Optional

import numpy


class ComputeNode(abc.ABC):
    """Base class for randomly generated compute nodes."""
    shape: int

    @abc.abstractmethod
    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        pass


@dataclasses.dataclass(frozen=True)
class Constant(ComputeNode):
    value: int
    shape: int = 0

    def __str__(self) -> str:
        return str(self.value)

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        return numpy.full(xy.shape[1:], self.value)


@dataclasses.dataclass(frozen=True)
class Variable(ComputeNode):
    index: int
    shape: int = 0

    def __str__(self) -> str:
        return 'xy'[self.index]

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        return xy[self.index]


@dataclasses.dataclass(frozen=True)
class Operator(ComputeNode):
    function: Callable[[numpy.ndarray, ...], numpy.ndarray]
    shape: int

    def __str__(self) -> str:
        return self.function.__name__

    def compute(self, xy: numpy.ndarray, inputs: list[numpy.ndarray]) -> numpy.ndarray:
        return self.function(*inputs)


@dataclasses.dataclass(frozen=True)
class Program:
    node: ComputeNode
    children: tuple['Program', ...]

    def __str__(self) -> str:
        if self.children:
            children = ', '.join(str(c) for c in self.children)
            return f'{self.node}({children})'
        else:
            return str(self.node)

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


RANDOM_LEAVES: tuple[Optional[ComputeNode], ...] = (
    # Constant
    None,
    # x, y
    Variable(0),
    Variable(1),
)

RANDOM_NODES: tuple[Optional[ComputeNode], ...] = RANDOM_LEAVES + (
    # Unary operators
    Operator(operator.neg, 1),
    Operator(operator.invert, 1),
    # Binary operators
    Operator(operator.add, 2),
    Operator(operator.sub, 2),
    Operator(operator.mul, 2),
    Operator(operator.floordiv, 2),
    Operator(operator.mod, 2),
    Operator(operator.and_, 2),
    Operator(operator.or_, 2),
    Operator(operator.xor, 2),
    Operator(operator.lshift, 2),
    Operator(operator.rshift, 2),
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
    binop = lambda op: lambda a, b: Program(Operator(op, 2), (a, b))
    sub = binop(operator.sub)
    or_ = binop(operator.or_)
    mod = binop(operator.mod)
    return sub(x, or_(const(-108), mod(y, (sub(x, const(8))))))
