#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-TTS subtalker 的 CUDA Graph 静态解码引擎。

背景：subtalker（code_predictor，5 层 1024 维）每帧执行 2 token prefill +
14 步解码 + 15 次采样，共 ~300 次 Python 级算子调度。小模型上 CPU 调度
开销远超 GPU 计算（实测该循环占每帧 73ms 中的 ~53ms）。

由于每帧的序列形状完全固定（kv 最大 16 个位置、步数固定 15），
可以把整帧循环一次性捕获成单张 CUDA graph：
  - KV/mask 预分配静态缓冲，mask 数值每帧固定，无需重置；
  - 采样用 Gumbel-max 技巧（与 multinomial 同分布）保证可捕获，
    噪声每帧在图外生成写入静态缓冲；
  - 每帧只需：写 2 个输入缓冲 + 1 次 uniform_ + 1 次 graph.replay()。

数值说明：注意力用 SDPA 替代 flash_attn（形状极小，GPU 时间可忽略），
与官方路径存在 bf16 级微小数值差异，greedy 对比会有极少数近平局翻转，
对随机采样的生产路径分布无影响。
"""

import torch
import torch.nn.functional as F

_NEG_INF = float("-inf")


class SubtalkerGraphEngine:
    """单帧 subtalker 的 CUDA graph：2 token prefill + 14 步解码 + 15 次采样。"""

    def __init__(self, code_predictor, temperature: float, top_k: int):
        # 静态缓冲必须在 inference_mode 之外分配：否则缓冲成为 inference
        # tensor，之后在 no_grad（如官方路径回退调用）下 copy_ 会被禁止。
        # 首次触发构建的调用可能位于 inference_mode 内，这里显式退出。
        with torch.inference_mode(False), torch.no_grad():
            self._init_engine(code_predictor, temperature, top_k)

    def _init_engine(self, code_predictor, temperature: float, top_k: int):
        self.cp = code_predictor
        pmodel = code_predictor.model
        self.layers = list(pmodel.layers)
        self.norm = pmodel.norm
        self.heads = list(code_predictor.lm_head)
        self.embeds = list(pmodel.codec_embedding)
        self.rotary = pmodel.rotary_emb

        cfg = pmodel.config
        self.n_layers = len(self.layers)
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        self.hidden = cfg.hidden_size
        # 注意力是升维投影：heads*head_dim(16*128=2048) != hidden(1024)，
        # o_proj 的输入维度是 heads*head_dim
        self.attn_dim = self.n_heads * self.head_dim
        self.vocab = cfg.vocab_size
        self.max_steps = code_predictor.config.num_code_groups - 1
        self.cache_len = 1 + self.max_steps          # 2 prefill + (max_steps-1) decode
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k else 0
        self.n_groups = self.n_heads // self.n_kv
        self.device = next(pmodel.parameters()).device
        self.dtype = next(pmodel.parameters()).dtype

        self._build_tables()
        self._build_buffers()
        self._capture()

    # ---------- 一次性表/缓冲 ----------

    def _build_tables(self):
        dummy = torch.zeros(1, 1, self.hidden, device=self.device, dtype=self.dtype)
        cos_rows, sin_rows = [], []
        for p in range(self.cache_len):
            cos, sin = self.rotary(dummy, torch.tensor([[p]], device=self.device))
            cos_rows.append(cos[0, 0])
            sin_rows.append(sin[0, 0])
        self.cos_tab = torch.stack(cos_rows)   # (cache_len, head_dim)
        self.sin_tab = torch.stack(sin_rows)

        # prefill mask: q=2 (位置0,1)，只允许 attend 0..1
        m = torch.full((1, 1, 2, self.cache_len), _NEG_INF,
                       device=self.device, dtype=self.dtype)
        m[..., 0:2] = 0.0
        self.mask_prefill = m
        # 每个解码步的 mask：允许 0..pos
        self.mask_steps = []
        for step in range(1, self.max_steps):
            pos = 1 + step
            m = torch.full((1, 1, 1, self.cache_len), _NEG_INF,
                           device=self.device, dtype=self.dtype)
            m[..., : pos + 1] = 0.0
            self.mask_steps.append(m)

    def _build_buffers(self):
        dev, dt = self.device, self.dtype
        self.emb2 = torch.zeros(1, 2, self.hidden, device=dev, dtype=dt)
        self.noise = torch.zeros(self.max_steps, self.vocab, device=dev, dtype=torch.float32)
        self.tokens_out = torch.zeros(1, self.max_steps, device=dev, dtype=torch.long)
        self.k_buf = torch.zeros(self.n_layers, 1, self.n_kv, self.cache_len, self.head_dim,
                                 device=dev, dtype=dt)
        self.v_buf = torch.zeros_like(self.k_buf)

    # ---------- 图内计算 ----------

    def _sdpa(self, q, k, v, mask):
        if self.n_groups == 1:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        try:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        except TypeError:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def _rope(self, q, k, cos, sin):
        # q: (1, H, L, D); cos/sin: (L, D) -> (1, 1, L, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _sample_in_graph(self, logits, noise_row):
        """与 HF 采样语义同分布：temperature -> top_k -> 多项式采样。

        multinomial 无法保证可捕获，改用 Gumbel-max：
        argmax(z + g) 与按 softmax(z) 多项式采样同分布。
        """
        z = logits.float()
        if self.temperature and self.temperature != 1.0:
            z = z / self.temperature
        if self.top_k:
            k = min(self.top_k, z.shape[-1] - 1)
            kth = torch.topk(z, k, dim=-1).values[..., -1:]
            z = z.masked_fill(z < kth, _NEG_INF)
        g = -torch.log(-torch.log(noise_row.clamp_min(1e-20)))
        return (z + g).argmax(dim=-1)

    def _frame(self):
        """单帧完整计算：读 emb2/noise 静态缓冲，写 k_buf/v_buf/tokens_out。"""
        h = self.emb2
        cos2 = self.cos_tab[0:2]
        sin2 = self.sin_tab[0:2]

        # ---- prefill（2 token，位置 0,1） ----
        for li, layer in enumerate(self.layers):
            residual = h
            x = layer.input_layernorm(h)
            a = layer.self_attn
            q = a.q_norm(a.q_proj(x).view(1, 2, self.n_heads, self.head_dim)).transpose(1, 2)
            k = a.k_norm(a.k_proj(x).view(1, 2, self.n_kv, self.head_dim)).transpose(1, 2)
            v = a.v_proj(x).view(1, 2, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = self._rope(q, k, cos2, sin2)
            self.k_buf[li][:, :, 0:2] = k
            self.v_buf[li][:, :, 0:2] = v
            o = self._sdpa(q, self.k_buf[li], self.v_buf[li], self.mask_prefill)
            o = a.o_proj(o.transpose(1, 2).reshape(1, 2, self.attn_dim))
            h = residual + o
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.heads[0](self.norm(h)[:, -1])[0]          # (vocab,)
        t = self._sample_in_graph(logits, self.noise[0])
        self.tokens_out[0, 0:1] = t

        # ---- 14 步解码（位置 2..cache_len-1） ----
        for step in range(1, self.max_steps):
            pos = 1 + step
            e = self.embeds[step - 1](t.view(1, 1))              # (1,1,hidden)
            cos1 = self.cos_tab[pos:pos + 1]
            sin1 = self.sin_tab[pos:pos + 1]
            for li, layer in enumerate(self.layers):
                residual = e
                x = layer.input_layernorm(e)
                a = layer.self_attn
                q = a.q_norm(a.q_proj(x).view(1, 1, self.n_heads, self.head_dim)).transpose(1, 2)
                k = a.k_norm(a.k_proj(x).view(1, 1, self.n_kv, self.head_dim)).transpose(1, 2)
                v = a.v_proj(x).view(1, 1, self.n_kv, self.head_dim).transpose(1, 2)
                q, k = self._rope(q, k, cos1, sin1)
                self.k_buf[li][:, :, pos:pos + 1] = k
                self.v_buf[li][:, :, pos:pos + 1] = v
                o = self._sdpa(q, self.k_buf[li], self.v_buf[li], self.mask_steps[step - 1])
                o = a.o_proj(o.transpose(1, 2).reshape(1, 1, self.attn_dim))
                e = residual + o
                e = e + layer.mlp(layer.post_attention_layernorm(e))
            logits = self.heads[step](self.norm(e)[:, -1])[0]
            t = self._sample_in_graph(logits, self.noise[step])
            self.tokens_out[0, step:step + 1] = t

    # ---------- 捕获与运行 ----------

    def _capture(self):
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.noise.uniform_()
                self._frame()
        torch.cuda.current_stream(self.device).wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._frame()

    def run(self, past_hidden: torch.Tensor, last_id_hidden: torch.Tensor) -> torch.Tensor:
        """输入 (1,1,H) x2，返回 (1, max_steps) 的 codebook token（静态缓冲视图）。"""
        self.emb2[:, 0:1].copy_(past_hidden)
        self.emb2[:, 1:2].copy_(last_id_hidden)
        self.noise.uniform_()
        self.graph.replay()
        return self.tokens_out


_ENGINE = None
_ENGINE_FAILED = False


def get_subtalker_engine(code_predictor, temperature: float = 0.9, top_k: int = 50):
    """进程级惰性单例；构建失败时永久回退 eager 并返回 None。"""
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None:
        return _ENGINE
    if _ENGINE_FAILED:
        return None
    try:
        _ENGINE = SubtalkerGraphEngine(code_predictor, temperature=temperature, top_k=top_k)
        return _ENGINE
    except Exception:
        import traceback
        traceback.print_exc()
        _ENGINE_FAILED = True
        return None
