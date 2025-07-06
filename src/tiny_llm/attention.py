import mlx.core as mx

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    '''
    key: 1 x H x L x D
    value: 1 x H x L x D
    query: 1 x H x L x D
    output: 1 x H x L x D
    mask: 1 x H x L x L
    '''
    if scale is None:
        scale = 1.0 / mx.sqrt(key.shape[-1])
    scores = mx.matmul(query, mx.swapaxes(key, -1, -2)) * scale
    if mask is not None:
        scores = scores + mask
    scores = softmax(scores, axis=-1)
    return mx.matmul(scores, value)


class SimpleMultiHeadAttention:
    '''
    E is hidden_size or embed_dim or dims or model_dim => H x D
    H is num_heads
    D is head_dim
    L is seq_len, in PyTorch API it's S (source len)
    N.. is number of batches

    w_q/w_k/w_v: E x (H x D)
    output/input: N x L x E
    w_o: output weigth matrix (H x D) x E
    '''
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        '''
        query/key/value: N x L x E 
        => N.. x L x H x D
        => N.. x H x L x D
        run attention and convert back to the original shape
        '''
        # apply weights through linear layer
        qkv = [linear(x, w) for x, w in zip((query, key, value), (self.wq, self.wk, self.wv))]

        # reshape and transpose for multi-head attention
        def reshape_and_transpose(x: mx.array) -> mx.array:
            # N.. x L x E -> N.. x L x H x D
            new_shape = (*x.shape[:-1], self.num_heads, self.hidden_size // self.num_heads)
            reshaped = x.reshape(new_shape)
            # N.. x L x H x D -> N.. x H x L x D
            return mx.swapaxes(reshaped, -2, -3)

        query_t, key_t, value_t = [reshape_and_transpose(x) for x in qkv]

        new_mask = None
        if mask is not None:
            expanded_mask = mx.expand_dims(mask, axis=0)
            target_shape = (self.num_heads,) + mask.shape
            new_mask = mx.broadcast_to(expanded_mask, target_shape)

        attention_output = scaled_dot_product_attention_simple(query_t, key_t, value_t, None, new_mask)
        
        # map back to N x L x H x D
        attention_output_t = mx.swapaxes(attention_output, -2, -3)

        # reshape back to N x L x E
        new_shape = (*attention_output_t.shape[:-2], self.hidden_size)
        attention_output_reshape = attention_output_t.reshape(*new_shape)

        return linear(attention_output_reshape, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
