"""
architecture.py - Decoder-only Transformer neural core for TLM.

This replaces the old TinyRNN engine with a hand-written, from-scratch
implementation of the architecture in the classic diagram:

    Output Probabilities
            |
         Softmax
            |
          Linear
            |
    +---------------+
    |   Add & Norm   |  <-\
    |  Feed Forward  |     |
    |   Add & Norm   |     | x Nx
    | Masked Multi-  |     |
    | Head Attention |  <-/
    +---------------+
            |
    (+)-- Positional Encoding
     |
    Output Embedding
            |
    Outputs (shifted right)

This is the GPT-style "decoder-only" Transformer: a stack of Nx identical
blocks, each doing causally-masked self-attention followed by a small
feed-forward network, with residual ("Add") connections and LayerNorm
after each sub-layer. There is no encoder and no encoder-decoder
cross-attention here -- only the self-attention path shown in the diagram.

Just like the RNN engine it replaces, NOTHING here comes from a
framework (no PyTorch/TensorFlow/JAX, no autograd). Every forward
computation has a matching hand-derived backward computation written
right below it. NumPy (or CuPy, if selected) is used purely as fast
array math -- matmuls, exp, etc. -- the same way it was used for the RNN.

Training processes ONE sequence (of length T = SEQ_LENGTH tokens) at a
time, exactly like the old TinyRNN's loss_and_grads(inputs, targets, ...),
so this drops in to the same training-loop shape.
"""

import math
import os
import struct

import numpy as np

_HAS_NUMPY = True
MAGIC_V1 = b"TLMD1"  # old checkpoints: header had no max_seq_len / tie flag
MAGIC = b"TLMD2"     # "TLM Decoder v2" binary checkpoint header


# ============================================================================
# Device / backend abstraction (kept for parity with the rest of the project)
# ============================================================================

def is_cupy_available():
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


def is_cugpu_available():
    try:
        import cugpu  # noqa: F401
        return True
    except ImportError:
        return False


def to_host(x):
    """Bring an array back to NumPy on the CPU -- a no-op if it's already
    NumPy. Used for the few things that must run on the host regardless
    of backend: RNG sampling and writing checkpoint bytes to disk."""
    return x.get() if hasattr(x, "get") else x


def scatter_add(xp, dest, indices, values):
    """dest[indices] += values, accumulating duplicate indices (what
    NumPy's np.add.at does for the embedding gradient). CuPy has no
    ufunc.at, so it uses cupyx.scatter_add instead -- same result,
    GPU-compatible."""
    if xp is np:
        np.add.at(dest, indices, values)
    else:
        import cupyx
        cupyx.scatter_add(dest, indices, values)


def get_array_module(device="cpu"):
    """Returns the array backend to compute with. 'cugpu' currently routes
    through NumPy here (cugpu exposes individual numpy-in/numpy-out ops
    rather than a drop-in module) -- wiring specific hot ops (matmul, add)
    through cugpu.* is a follow-up, not required for correctness."""
    if device == "gpu":
        try:
            import cupy as cp
            return cp
        except ImportError:
            print("[!] CuPy not available, falling back to NumPy (CPU).")
            return np
    return np


def prompt_device():
    cugpu_ok = is_cugpu_available()
    cupy_ok = is_cupy_available()
    print("Select Compute Backend:")
    print("  1. CPU (NumPy)         - Universal standard CPU [Default]")
    print(f"  2. GPU (cugpu / ILGPU) - AMD / Intel / NVIDIA "
          f"({'Detected [OK]' if cugpu_ok else 'not detected'})")
    print(f"  3. GPU (CuPy / CUDA)   - NVIDIA CUDA accelerated "
          f"({'Detected [OK]' if cupy_ok else 'CuPy/CUDA not detected'})")
    while True:
        choice = input("> ").strip()
        if choice in ("", "1"):
            return "cpu"
        if choice == "2":
            if cugpu_ok:
                return "cugpu"
            print("[!] Cannot select Option 2: cugpu is not installed/built.")
        elif choice == "3":
            if cupy_ok:
                return "gpu"
            print("[!] Cannot select Option 3: CuPy/CUDA is not detected.")
        else:
            print("Please enter 1, 2, or 3.")


# ============================================================================
# Math building blocks (each with a hand-derived backward pass)
# ============================================================================

def xavier(rng, shape):
    fan_in, fan_out = shape[0], shape[-1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def wgrad(a, b):
    """Weight gradient for y = a @ W: the sum over every batch/time
    position of outer(a, b). Written as ONE big matmul over the flattened
    (batch*time) rows so it runs on BLAS (or cuBLAS) -- the equivalent
    np.einsum is 5-13x slower at these sizes because it doesn't."""
    return a.reshape(-1, a.shape[-1]).T @ b.reshape(-1, b.shape[-1])


_NP_RNG = np.random.default_rng()


def dropout_forward(x, p, xp):
    """Inverted dropout. Returns (output, mask). With p == 0 the mask is
    None and x passes through untouched, so training without dropout is
    bit-for-bit what it was before dropout existed."""
    if p <= 0.0:
        return x, None
    if xp is np:
        r = _NP_RNG.random(x.shape, dtype=np.float32)
    else:
        r = xp.random.random_sample(x.shape, dtype=np.float32)
    mask = (r >= p).astype(x.dtype) / (1.0 - p)
    return x * mask, mask


def dropout_backward(dout, mask):
    return dout if mask is None else dout * mask


def softmax(x, axis=-1, xp=np):
    x = x - xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=axis, keepdims=True)


def gelu(x, xp=np):
    """tanh-approximation GELU, same one GPT-2 uses."""
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + xp.tanh(c * (x + 0.044715 * x ** 3)))


def gelu_backward(x, dout, xp=np):
    c = math.sqrt(2.0 / math.pi)
    x3 = x ** 3
    inner = c * (x + 0.044715 * x3)
    t = xp.tanh(inner)
    sech2 = 1.0 - t * t
    dinner_dx = c * (1.0 + 3.0 * 0.044715 * x ** 2)
    dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner_dx
    return dout * dgelu_dx


def layer_norm_forward(x, gamma, beta, eps=1e-5, xp=np):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std_inv = 1.0 / xp.sqrt(var + eps)
    xhat = (x - mu) * std_inv
    out = gamma * xhat + beta
    cache = (xhat, std_inv, gamma)
    return out, cache


def layer_norm_backward(dout, cache, xp=np):
    xhat, std_inv, gamma = cache
    N = dout.shape[-1]
    dgamma = xp.sum(dout * xhat, axis=tuple(range(dout.ndim - 1)))
    dbeta = xp.sum(dout, axis=tuple(range(dout.ndim - 1)))
    dxhat = dout * gamma
    dx = std_inv / N * (
        N * dxhat
        - xp.sum(dxhat, axis=-1, keepdims=True)
        - xhat * xp.sum(dxhat * xhat, axis=-1, keepdims=True)
    )
    return dx, dgamma, dbeta


def sinusoidal_positional_encoding(max_len, d_model):
    """Fixed (non-trainable) sine/cosine positional encoding -- the
    small wave icon feeding into the (+) in the diagram."""
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    position = np.arange(0, max_len).reshape(-1, 1).astype(np.float32)
    div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) *
                       -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


# ============================================================================
# Masked Multi-Head Self-Attention
# ============================================================================

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rng, xp=np):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.xp = xp

        self.Wq = xavier(rng, (d_model, d_model))
        self.Wk = xavier(rng, (d_model, d_model))
        self.Wv = xavier(rng, (d_model, d_model))
        self.Wo = xavier(rng, (d_model, d_model))
        self.bq = np.zeros(d_model, dtype=np.float32)
        self.bk = np.zeros(d_model, dtype=np.float32)
        self.bv = np.zeros(d_model, dtype=np.float32)
        self.bo = np.zeros(d_model, dtype=np.float32)

    def params(self):
        return {"Wq": self.Wq, "bq": self.bq, "Wk": self.Wk, "bk": self.bk,
                "Wv": self.Wv, "bv": self.bv, "Wo": self.Wo, "bo": self.bo}

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, Dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x, causal_mask):
        """x: (B, T, d_model). causal_mask: (T, T) bool, True = attend allowed."""
        Q = x @ self.Wq + self.bq
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        xp = self.xp
        Qh, Kh, Vh = self._split_heads(Q), self._split_heads(K), self._split_heads(V)

        scores = (Qh @ Kh.transpose(0, 1, 3, 2)) / math.sqrt(self.d_head)
        scores = xp.where(causal_mask[None, None, :, :], scores, xp.float32(-1e9))
        attn = softmax(scores, axis=-1, xp=xp)                # (B, H, T, T)
        context = attn @ Vh                                    # (B, H, T, Dh)
        merged = self._merge_heads(context)                   # (B, T, D)
        out = merged @ self.Wo + self.bo

        cache = dict(x=x, Qh=Qh, Kh=Kh, Vh=Vh, attn=attn, merged=merged)
        return out, cache

    def backward(self, dout, cache):
        x, Qh, Kh, Vh = cache['x'], cache['Qh'], cache['Kh'], cache['Vh']
        attn, merged = cache['attn'], cache['merged']

        xp = self.xp
        # Weight grads reduce over both batch and time (every position in
        # every sequence contributed to shared weights), so a plain .T @
        # (2D-only) becomes an explicit sum over "b,t" via einsum instead.
        dWo = wgrad(merged, dout)
        dbo = dout.sum(axis=(0, 1))
        dmerged = dout @ self.Wo.T
        dcontext = self._split_heads(dmerged)                 # (B, H, T, Dh)

        dattn = dcontext @ Vh.transpose(0, 1, 3, 2)            # (B, H, T, T)
        dVh = attn.transpose(0, 1, 3, 2) @ dcontext            # (B, H, T, Dh)

        # softmax backward (per row, over the last axis)
        dscores = attn * (dattn - xp.sum(dattn * attn, axis=-1, keepdims=True))
        dscores = dscores / math.sqrt(self.d_head)

        dQh = dscores @ Kh
        dKh = dscores.transpose(0, 1, 3, 2) @ Qh

        dQ = self._merge_heads(dQh)
        dK = self._merge_heads(dKh)
        dV = self._merge_heads(dVh)

        dWq = wgrad(x, dQ); dbq = dQ.sum(axis=(0, 1))
        dWk = wgrad(x, dK); dbk = dK.sum(axis=(0, 1))
        dWv = wgrad(x, dV); dbv = dV.sum(axis=(0, 1))

        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        grads = {"Wq": dWq, "bq": dbq, "Wk": dWk, "bk": dbk,
                 "Wv": dWv, "bv": dbv, "Wo": dWo, "bo": dbo}
        return dx, grads


# ============================================================================
# Feed Forward block (Linear -> GELU -> Linear)
# ============================================================================

class FeedForward:
    def __init__(self, d_model, d_ff, rng, xp=np):
        self.xp = xp
        self.W1 = xavier(rng, (d_model, d_ff))
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = xavier(rng, (d_ff, d_model))
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        h_pre = x @ self.W1 + self.b1
        h = gelu(h_pre, xp=self.xp)
        out = h @ self.W2 + self.b2
        cache = dict(x=x, h_pre=h_pre, h=h)
        return out, cache

    def backward(self, dout, cache):
        x, h_pre, h = cache['x'], cache['h_pre'], cache['h']
        xp = self.xp
        dW2 = wgrad(h, dout)
        db2 = dout.sum(axis=(0, 1))
        dh = dout @ self.W2.T
        dh_pre = gelu_backward(h_pre, dh, xp=self.xp)
        dW1 = wgrad(x, dh_pre)
        db1 = dh_pre.sum(axis=(0, 1))
        dx = dh_pre @ self.W1.T
        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return dx, grads


# ============================================================================
# One decoder block: Masked MHA -> Add & Norm -> Feed Forward -> Add & Norm
# ============================================================================

class DecoderBlock:
    def __init__(self, d_model, num_heads, d_ff, rng, xp=np):
        self.xp = xp
        self.attn = MultiHeadAttention(d_model, num_heads, rng, xp=xp)
        self.ffn = FeedForward(d_model, d_ff, rng, xp=xp)
        self.gamma1 = np.ones(d_model, dtype=np.float32)
        self.beta1 = np.zeros(d_model, dtype=np.float32)
        self.gamma2 = np.ones(d_model, dtype=np.float32)
        self.beta2 = np.zeros(d_model, dtype=np.float32)

    def params(self):
        p = {"gamma1": self.gamma1, "beta1": self.beta1,
             "gamma2": self.gamma2, "beta2": self.beta2}
        p.update({f"attn.{k}": v for k, v in self.attn.params().items()})
        p.update({f"ffn.{k}": v for k, v in self.ffn.params().items()})
        return p

    def forward(self, x, causal_mask, dropout_p=0.0):
        attn_out, attn_cache = self.attn.forward(x, causal_mask)
        attn_out, drop1 = dropout_forward(attn_out, dropout_p, self.xp)
        res1 = x + attn_out                                  # Add
        norm1, ln1_cache = layer_norm_forward(res1, self.gamma1, self.beta1, xp=self.xp)  # Norm

        ffn_out, ffn_cache = self.ffn.forward(norm1)
        ffn_out, drop2 = dropout_forward(ffn_out, dropout_p, self.xp)
        res2 = norm1 + ffn_out                                # Add
        norm2, ln2_cache = layer_norm_forward(res2, self.gamma2, self.beta2, xp=self.xp)  # Norm

        cache = dict(attn_cache=attn_cache, ln1_cache=ln1_cache,
                     ffn_cache=ffn_cache, ln2_cache=ln2_cache,
                     drop1=drop1, drop2=drop2)
        return norm2, cache

    def backward(self, dout, cache):
        dres2, dgamma2, dbeta2 = layer_norm_backward(dout, cache['ln2_cache'], xp=self.xp)
        dnorm1_from_res2 = dres2                              # Add: splits equally
        dffn_out = dropout_backward(dres2, cache['drop2'])
        dnorm1_from_ffn, ffn_grads = self.ffn.backward(dffn_out, cache['ffn_cache'])
        dnorm1 = dnorm1_from_res2 + dnorm1_from_ffn

        dres1, dgamma1, dbeta1 = layer_norm_backward(dnorm1, cache['ln1_cache'], xp=self.xp)
        dx_from_res1 = dres1                                  # Add: splits equally
        dattn_out = dropout_backward(dres1, cache['drop1'])
        dx_from_attn, attn_grads = self.attn.backward(dattn_out, cache['attn_cache'])
        dx = dx_from_res1 + dx_from_attn

        grads = {"gamma1": dgamma1, "beta1": dbeta1,
                 "gamma2": dgamma2, "beta2": dbeta2}
        grads.update({f"attn.{k}": v for k, v in attn_grads.items()})
        grads.update({f"ffn.{k}": v for k, v in ffn_grads.items()})
        return dx, grads


# ============================================================================
# Parameter counting (mirrors calculate_model_parameters from the RNN days)
# ============================================================================

def calculate_model_parameters(d_model, vocab_size, num_layers=1, num_heads=4, d_ff=None,
                               tie_weights=False):
    if d_ff is None:
        d_ff = d_model * 4
    embed = vocab_size * d_model
    per_attn = 4 * (d_model * d_model + d_model)          # Wq,Wk,Wv,Wo + biases
    per_ffn = (d_model * d_ff + d_ff) + (d_ff * d_model + d_model)
    per_ln = 2 * d_model * 2                              # 2 LayerNorms x (gamma+beta)
    per_block = per_attn + per_ffn + per_ln
    # Tied: the output projection reuses the embedding matrix, leaving
    # only the bias. Untied: a separate d_model x vocab matrix plus bias.
    output_head = vocab_size if tie_weights else d_model * vocab_size + vocab_size
    return embed + num_layers * per_block + output_head


# ============================================================================
# TinyTransformer: the full decoder-only stack
# ============================================================================

class TinyTransformer:
    """
    Output Embedding -> (+) Positional Encoding
      -> [ Masked Multi-Head Attention -> Add & Norm
           -> Feed Forward -> Add & Norm ] x num_layers
      -> Linear -> Softmax -> Output Probabilities

    One sequence at a time (T = SEQ_LENGTH tokens), trained with
    teacher forcing ("Outputs, shifted right"): targets[t] = inputs[t+1].
    """

    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4,
                 d_ff=None, max_seq_len=256, seed=None, device="cpu",
                 tie_weights=False):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff if d_ff is not None else d_model * 4
        self.max_seq_len = max_seq_len
        self.device = device
        self.xp = get_array_module(device)
        self.tie_weights = bool(tie_weights)
        self.dropout_p = 0.0     # training-time only; set by train.py/finetune.py

        rng = np.random.default_rng(seed)

        # Output Embedding (trainable)
        self.embedding = (rng.standard_normal((vocab_size, d_model)) * 0.02).astype(np.float32)

        # Positional Encoding (fixed, not trainable)
        self.pos_encoding = sinusoidal_positional_encoding(max_seq_len, d_model)

        # Nx decoder blocks
        self.blocks = [DecoderBlock(d_model, num_heads, self.d_ff, rng, xp=self.xp)
                        for _ in range(num_layers)]

        # Final Linear -> Softmax head
        # (With tie_weights, the output projection IS embedding.T -- no
        # separate matrix -- which saves vocab*d_model parameters, usually
        # a big share of a small model, and tends to help quality.)
        self.Wout = None if self.tie_weights else xavier(rng, (d_model, vocab_size))
        self.bout = np.zeros(vocab_size, dtype=np.float32)

        # Adam optimizer state (built lazily to match whatever params() returns)
        self._m = {}
        self._v = {}
        self._t = 0

        self.total_params = calculate_model_parameters(
            d_model, vocab_size, num_layers, num_heads, self.d_ff,
            tie_weights=self.tie_weights)

        # All weights above were built with plain NumPy (deterministic,
        # seedable). Move every one to the selected backend now, once,
        # rather than threading GPU-vs-CPU array creation through every
        # xavier()/zeros() call above.
        if self.xp is not np:
            self.pos_encoding = self.xp.asarray(self.pos_encoding)
            for name, arr in self.params().items():
                self._set_param(name, self.xp.asarray(arr))

    # ------------------------------------------------------------------
    def _causal_mask(self, T):
        return self.xp.tril(self.xp.ones((T, T), dtype=bool))

    def params(self):
        p = {"embedding": self.embedding, "bout": self.bout}
        if not self.tie_weights:
            p["Wout"] = self.Wout
        for i, blk in enumerate(self.blocks):
            p.update({f"blk{i}.{k}": v for k, v in blk.params().items()})
        return p

    def _set_param(self, name, value):
        """Writes an updated array back into the right object in-place."""
        if name in ("embedding", "Wout", "bout"):
            setattr(self, name, value)
            return
        blk_idx, rest = name.split(".", 1)
        blk = self.blocks[int(blk_idx[3:])]
        if "." in rest:
            sub, attr = rest.split(".", 1)
            target = blk.attn if sub == "attn" else blk.ffn
            setattr(target, attr, value)
        else:
            setattr(blk, rest, value)

    # ------------------------------------------------------------------
    def forward(self, input_ids, training=False, last_only=False):
        """input_ids: (T,) for a single sequence, or (B,T) for a batch of B.
        training=True turns dropout on (loss_and_grads does this);
        last_only=True computes logits for just the final position -- all
        generation needs, and with a 16k vocab the output projection over
        every position is by far the most expensive part of a decode step.
        Returns (logits, probs, cache). A 1D input gets 1D-shaped (T,V)
        logits/probs back, exactly as before -- generate() and chat.py's
        own decode loop both pass single sequences and need no changes.
        A 2D input gets (B,T,V) back. Either way, `cache` internally always
        keeps the full (B,T,...) shapes, since loss_and_grads needs them
        for the backward pass regardless of what the caller asked for."""
        xp = self.xp
        input_ids = xp.asarray(input_ids)
        squeeze = input_ids.ndim == 1
        ids2d = input_ids[None, :] if squeeze else input_ids
        B, T = ids2d.shape

        p = self.dropout_p if training else 0.0
        x = self.embedding[ids2d] + self.pos_encoding[:T]     # (B,T,D)
        x, emb_mask = dropout_forward(x, p, xp)
        mask = self._causal_mask(T)

        block_caches = []
        for blk in self.blocks:
            x, cache = blk.forward(x, mask, p)
            block_caches.append(cache)

        x_head = x[:, -1:, :] if last_only else x
        W = self.embedding.T if self.tie_weights else self.Wout
        logits = x_head @ W + self.bout                      # (B,T,V)  Linear
        probs = softmax(logits, axis=-1, xp=self.xp)          # (B,T,V)  Softmax

        cache = dict(input_ids=ids2d, x_final=x, probs=probs,
                     block_caches=block_caches, emb_mask=emb_mask)
        if squeeze:
            return logits[0], probs[0], cache
        return logits, probs, cache

    def loss_and_grads(self, input_ids, target_ids, loss_mask=None):
        """Cross-entropy next-token loss + full backward pass. Mirrors the
        old TinyRNN.loss_and_grads(inputs, targets, hprev) signature/spirit,
        minus the recurrent hidden state (Transformers don't carry one).

        loss_mask: optional array of 0/1 per position (length T). Position t
        contributes to the loss (and gradient) only where loss_mask[t] == 1.
        Used by finetune.py to train only on assistant-reply characters,
        not on the user's question or prompt scaffolding. Defaults to all
        1s (every position counts) -- exactly the old unmasked behavior
        train.py already relies on, so this is fully backward-compatible."""
        xp = self.xp
        # forward()'s return value gets squeezed back to (T,V) for a 1D
        # input, so use cache's internal, always-(B,T,...) versions here
        # instead -- one code path handles a lone sequence (B=1) and a
        # real batch identically.
        _, _, cache = self.forward(input_ids, training=True)
        ids2d = cache['input_ids']            # (B, T)
        probs = cache['probs']                # (B, T, V)
        B, T = ids2d.shape

        target_ids = xp.asarray(target_ids)
        if target_ids.ndim == 1:
            target_ids = target_ids[None, :]

        if loss_mask is None:
            loss_mask = xp.ones((B, T), dtype=np.float32)
        else:
            loss_mask = xp.asarray(loss_mask, dtype=np.float32)
            if loss_mask.ndim == 1:
                loss_mask = loss_mask[None, :]

        b_idx = xp.arange(B)[:, None]
        t_idx = xp.arange(T)[None, :]
        target_probs = probs[b_idx, t_idx, target_ids]        # (B, T)
        per_token_loss = -xp.log(xp.clip(target_probs, 1e-9, 1.0))
        # Mean over valid (non-masked) tokens rather than a raw sum. A sum
        # makes gradient size scale with batch_size * seq_len, so the same
        # learning rate and [-5,5] clip would mean something different for
        # every batch size; a mean keeps them comparable no matter how
        # many sequences or tokens went into a step.
        valid_count = xp.maximum(xp.sum(loss_mask), 1.0)
        loss = xp.sum(per_token_loss * loss_mask) / valid_count

        # dLoss/dLogits for softmax + cross-entropy: probs - one_hot(target),
        # zeroed out at masked-off positions so they contribute no gradient,
        # and divided by valid_count to match the mean above.
        dlogits = probs.copy()
        dlogits[b_idx, t_idx, target_ids] -= 1.0
        dlogits *= loss_mask[:, :, None] / valid_count

        x_final = cache['x_final']                            # (B, T, D)
        dbout = dlogits.sum(axis=(0, 1))
        if self.tie_weights:
            dhead = wgrad(dlogits, x_final)                   # (V, D) -> embedding
            dx = dlogits @ self.embedding                     # (B, T, D)
            grads = {"bout": dbout}
        else:
            dhead = None
            dx = dlogits @ self.Wout.T                        # (B, T, D)
            grads = {"Wout": wgrad(x_final, dlogits), "bout": dbout}
        for i in reversed(range(self.num_layers)):
            dx, blk_grads = self.blocks[i].backward(dx, cache['block_caches'][i])
            grads.update({f"blk{i}.{k}": v for k, v in blk_grads.items()})

        dx = dropout_backward(dx, cache['emb_mask'])
        dembedding = xp.zeros_like(self.embedding)
        scatter_add(xp, dembedding, ids2d, dx)                # (B,T) indices, (B,T,D) values
        if dhead is not None:
            dembedding += dhead                               # tied output head's gradient
        grads["embedding"] = dembedding

        for g in grads.values():
            xp.clip(g, -5, 5, out=g)

        # Always hand back a plain Python float -- train.py/finetune.py
        # average, format, and log this value without needing to know or
        # care which backend produced it.
        return float(to_host(loss)), grads

    def update(self, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
               weight_decay=0.0):
        """Adam optimizer step (Transformers train far more reliably with
        Adam than with the RNN engine's Adagrad, so this engine uses Adam)."""
        self._t += 1
        params = self.params()
        for name, p in params.items():
            g = grads[name]
            if name not in self._m:
                self._m[name] = self.xp.zeros_like(p)
                self._v[name] = self.xp.zeros_like(p)
            self._m[name] = beta1 * self._m[name] + (1 - beta1) * g
            self._v[name] = beta2 * self._v[name] + (1 - beta2) * (g * g)
            m_hat = self._m[name] / (1 - beta1 ** self._t)
            v_hat = self._v[name] / (1 - beta2 ** self._t)
            new_p = p - learning_rate * m_hat / (self.xp.sqrt(v_hat) + eps)
            if weight_decay and p.ndim >= 2:
                # Decoupled (AdamW-style) decay on weight matrices only --
                # not biases or LayerNorm scales -- shrinking weights a
                # little each step to discourage memorizing the data.
                new_p = new_p - learning_rate * weight_decay * p
            self._set_param(name, new_p)

    # ------------------------------------------------------------------
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, rng=None):
        """Autoregressive sampling. Recomputes the forward pass over the
        (sliding-window) context each step -- simple, no KV-cache, matches
        the project's "tiny, transparent" philosophy over raw speed."""
        rng = rng or np.random.default_rng()
        ids = list(input_ids)
        for _ in range(max_new_tokens):
            window = ids[-self.max_seq_len:]
            logits, _, _ = self.forward(np.array(window), last_only=True)
            last_logits = logits[-1] / max(temperature, 1e-6)
            probs = softmax(last_logits, xp=self.xp)
            # Sampling a single vocab-sized vector is cheap, so it always
            # runs on the host -- avoids needing CuPy's own RNG API.
            p = to_host(probs).astype(np.float64)
            p /= p.sum()
            next_id = rng.choice(self.vocab_size, p=p)
            ids.append(int(next_id))
        return ids

    # ------------------------------------------------------------------
    def save(self, filepath):
        """Binary checkpoint: MAGIC header + dims, then every trainable
        array (order given by self.params()) as raw float32 bytes."""
        params = self.params()
        with open(filepath, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<IIIII", self.vocab_size, self.d_model,
                                 self.num_layers, self.num_heads, self.d_ff))
            f.write(struct.pack("<II", self.max_seq_len, 1 if self.tie_weights else 0))
            f.write(struct.pack("<I", len(params)))
            for name, arr in params.items():
                name_b = name.encode("utf-8")
                f.write(struct.pack("<H", len(name_b)))
                f.write(name_b)
                f.write(struct.pack("<I", arr.ndim))
                f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
                f.write(to_host(arr).astype(np.float32).tobytes())

    @classmethod
    def load(cls, filepath, device="cpu"):
        with open(filepath, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic not in (MAGIC, MAGIC_V1):
                raise ValueError(f"Not a TLM checkpoint: {filepath}")
            vocab_size, d_model, num_layers, num_heads, d_ff = struct.unpack("<IIIII", f.read(20))
            if magic == MAGIC:
                max_seq_len, flags = struct.unpack("<II", f.read(8))
                model = cls(vocab_size, d_model, num_layers, num_heads, d_ff,
                            max_seq_len=max_seq_len, device=device,
                            tie_weights=bool(flags & 1))
            else:
                # v1 files never recorded max_seq_len or tying (always untied).
                model = cls(vocab_size, d_model, num_layers, num_heads, d_ff, device=device)
            (n_params,) = struct.unpack("<I", f.read(4))
            for _ in range(n_params):
                (name_len,) = struct.unpack("<H", f.read(2))
                name = f.read(name_len).decode("utf-8")
                (ndim,) = struct.unpack("<I", f.read(4))
                shape = struct.unpack(f"<{ndim}I", f.read(4 * ndim))
                count = 1
                for s in shape:
                    count *= s
                arr = np.frombuffer(f.read(4 * count), dtype=np.float32).reshape(shape).copy()
                model._set_param(name, model.xp.asarray(arr))
        return model
