# pyright: strict, reportUnknownMemberType=false, reportMissingTypeStubs=false
from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaForCausalLM


class OrderTokenizer(nn.Module):
    """Order tokenizer."""

    def __init__(
        self,
        max_order_index: int,
        emb_dim: int,
        num_max_orders: int,
        num_bins_price_level: int,
        num_bins_pred_order_volume: int,
        num_bins_order_interval: int,
    ) -> None:
        super().__init__()
        self.max_order_index = max_order_index
        self.num_max_orders = num_max_orders
        self.emb_dim = emb_dim
        self.dim_order = 15
        self.num_ratio_slots = 10
        self.max_chg_slots = 2000
        self.num_bins_price_level = num_bins_price_level
        self.num_bins_pred_order_volume = num_bins_pred_order_volume
        self.num_bins_order_interval = num_bins_order_interval
        self.emb_order_type = nn.Embedding(3, self.emb_dim)
        self.emb_price_level = nn.Embedding(num_bins_price_level, self.emb_dim)
        self.emb_pred_order_volume = nn.Embedding(num_bins_pred_order_volume, self.emb_dim)
        self.emb_order_interval = nn.Embedding(num_bins_order_interval, self.emb_dim)
        self.emb_chg_to_open = nn.Embedding(self.max_chg_slots * 2 + 1, self.emb_dim)
        self.emb_time_to_open = nn.Embedding(14400 // 5 + 1, self.emb_dim)  # group every 5 seconds..
        self.lob_tokenizer = nn.Sequential(
            nn.Linear(10, self.emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Tokenize inputs."""
        batch_size = features.size(0)
        assert features.size(1) == self.num_max_orders * self.dim_order
        features = features.reshape((batch_size, self.num_max_orders, self.dim_order))
        # normalize chg_to_open and time_to_open with first order in sequence
        features[:, :, 3] = features[:, :, 3] - features[:, 0, 3].unsqueeze(1)
        features[:, :, 4] = features[:, :, 4] - features[:, 0, 4].unsqueeze(1)
        features = features.reshape((batch_size * self.num_max_orders, self.dim_order))
        (order_type, price_level, pred_order_volume, order_interval) = self.split_order_index(
            features[:, 0],
            self.num_bins_price_level,
            self.num_bins_pred_order_volume,
            self.num_bins_order_interval,
        )
        embs = [
            self.emb_order_type(order_type),
            self.emb_price_level(price_level),
            self.emb_pred_order_volume(pred_order_volume),
            self.emb_order_interval(order_interval),
            self.emb_chg_to_open(features[:, 3].clip(min=-self.max_chg_slots, max=self.max_chg_slots) + self.max_chg_slots),
            self.emb_time_to_open(features[:, 4] // 5),
            self.lob_tokenizer(features[:, 5:15].float()),
        ]

        tokens = torch.sum(torch.stack(embs), dim=0)
        tokens = tokens.reshape(batch_size, self.num_max_orders * self.emb_dim)
        return tokens

    @staticmethod
    def split_order_index(
        order_index: Tensor,
        num_bins_price_level: int = 32,
        num_bins_pred_order_volume: int = 32,
        num_bins_order_interval: int = 16,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split order index into order_type, price level, pred_order_volume, and order_interval."""
        order_type = order_index // (num_bins_price_level * num_bins_pred_order_volume * num_bins_order_interval)
        price_level = (order_index % (num_bins_price_level * num_bins_pred_order_volume * num_bins_order_interval)) // (
            num_bins_pred_order_volume * num_bins_order_interval
        )
        pred_order_volume = (order_index % (num_bins_pred_order_volume * num_bins_order_interval)) // num_bins_order_interval
        order_interval = order_index % num_bins_order_interval
        return (order_type, price_level, pred_order_volume, order_interval)


class OrderModel(nn.Module):
    """Multi-granulairty model."""

    def __init__(
        self,
        # model config
        emb_dim: int,
        num_layers: int,
        num_heads: int,
        # data config
        num_bins_price_level: int = 32,
        num_bins_pred_order_volume: int = 32,
        num_bins_order_interval: int = 16,
        num_max_orders: int = 200,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_max_orders = num_max_orders

        # model config
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.output_dim = 3 * num_bins_price_level * num_bins_pred_order_volume * num_bins_order_interval
        self.order_info_tokenizer = OrderTokenizer(
            max_order_index=self.output_dim,
            emb_dim=emb_dim,
            num_max_orders=num_max_orders,
            num_bins_price_level=num_bins_price_level,
            num_bins_pred_order_volume=num_bins_pred_order_volume,
            num_bins_order_interval=num_bins_order_interval,
        )

        llama_config = LlamaConfig(
            hidden_size=emb_dim,
            num_attention_heads=num_heads,
            intermediate_size=4 * emb_dim,
            num_hidden_layers=num_layers,
            attention_dropout=dropout,
            use_cache=False,
            vocab_size=self.output_dim,
        )
        self.decoder = LlamaForCausalLM(llama_config)
        self.linear_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, self.output_dim),
        )

    def tokenize(self, features: Tensor) -> Tensor:
        """Tokenize inputs."""
        order_features = features.reshape(features.size(0), -1)
        assert order_features.size(1) == self.num_max_orders * 15
        inputs = self.order_info_tokenizer(order_features)
        assert inputs.size(0) == features.size(0)
        num_tokens = inputs.size(1) // self.emb_dim
        inputs = inputs.reshape(features.size(0), num_tokens, self.emb_dim)
        return inputs

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass."""
        tokens = self.tokenize(features)
        out = self.decoder(inputs_embeds=tokens, use_cache=False)
        logits = out.logits
        return logits

    def sample(self, features: Tensor, temperature: float = 1.0) -> Tensor:
        """Sample predictions from output probabilities."""
        logits = self(features)
        logits = logits[:, -1, :]  # get last logits
        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=1)
        index = torch.multinomial(probs, 1, replacement=True)
        return index

    def top(self, features: Tensor) -> Tensor:
        """Get top prediction from output probabilities."""
        logits = self(features)
        logits = logits[:, -1, :]  # get last logits
        index = torch.argmax(logits, dim=1)
        return index
