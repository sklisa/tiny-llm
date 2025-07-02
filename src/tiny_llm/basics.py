import math

import mlx.core as mx


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    return mx.matmul(x, mx.transpose(w)) + bias


def silu(x: mx.array) -> mx.array:
    pass
