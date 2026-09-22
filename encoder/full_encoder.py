import logging
import os

import timm
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import BertTokenizer


HIDDEN = 768


def masked_mean(sequence_output, attention_mask):
    mask = attention_mask.to(dtype=sequence_output.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (sequence_output * mask).sum(dim=1) / denom


class SpatialSequenceEncoder_Image(nn.Module):
    """Projects ResNet feature maps to spatial BERT-width token sequences."""

    def __init__(self, in_channels=2048, out_channels=HIDDEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        spatial_feat = x.flatten(2).transpose(1, 2)
        global_feat = self.pool(x).flatten(1)
        return spatial_feat, global_feat


class PairedSymmetricCrossAttention(nn.Module):
    """One symmetric interaction for each semantically matched modality pair.

    The main image exchanges evidence with its caption, while OTA-aligned
    object evidence exchanges evidence with the relation sentence. Text-side
    outputs are pooled back into the paired visual outputs, so none of the four
    directional attention results is silently discarded.
    """

    def __init__(
        self, hidden_size=HIDDEN, num_heads=8, dropout=0.2,
        residual_scale_init=0.0,
    ):
        super().__init__()
        self.fine_to_caption = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.caption_to_fine = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.object_to_text = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_object = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.fine_token_norm = nn.LayerNorm(hidden_size)
        self.caption_token_norm = nn.LayerNorm(hidden_size)
        self.object_token_norm = nn.LayerNorm(hidden_size)
        self.text_token_norm = nn.LayerNorm(hidden_size)
        self.fine_pair_norm = nn.LayerNorm(hidden_size)
        self.object_pair_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        # The pretrained multimodal representation is the initial function;
        # attention earns a nonzero contribution through episodic training.
        self.fine_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.object_residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        # Disabled in ordinary training. The visualization utility enables
        # this flag to retain per-head maps without changing model outputs.
        self.capture_diagnostics = False
        self.last_attention = None

    def forward(
        self,
        fine_tokens,
        object_tokens,
        caption_tokens,
        relation_tokens,
        caption_mask,
        relation_mask,
        raw_object_tokens=None,
    ):
        caption_padding = caption_mask == 0
        relation_padding = relation_mask == 0

        need_weights = bool(self.capture_diagnostics)
        attention_kwargs = (
            {"need_weights": True, "average_attn_weights": False}
            if need_weights else {"need_weights": False}
        )
        fine_context, fine_to_caption_weights = self.fine_to_caption(
            fine_tokens, caption_tokens, caption_tokens,
            key_padding_mask=caption_padding, **attention_kwargs,
        )
        caption_context, caption_to_fine_weights = self.caption_to_fine(
            caption_tokens, fine_tokens, fine_tokens, **attention_kwargs)
        object_context, object_to_text_weights = self.object_to_text(
            object_tokens, relation_tokens, relation_tokens,
            key_padding_mask=relation_padding, **attention_kwargs)
        text_context, text_to_object_weights = self.text_to_object(
            relation_tokens, object_tokens, object_tokens, **attention_kwargs)
        if need_weights:
            self.last_attention = {
                "fine_to_caption": fine_to_caption_weights.detach().cpu(),
                "caption_to_fine": caption_to_fine_weights.detach().cpu(),
                "object_to_relation": object_to_text_weights.detach().cpu(),
                "relation_to_object": text_to_object_weights.detach().cpu(),
            }

        fine_tokens = self.fine_token_norm(
            fine_tokens + self.dropout(fine_context)
        )
        caption_tokens = self.caption_token_norm(
            caption_tokens + self.dropout(caption_context)
        ).masked_fill(caption_padding.unsqueeze(-1), 0.0)
        object_tokens = self.object_token_norm(
            object_tokens + self.dropout(object_context)
        )
        relation_tokens = self.text_token_norm(
            relation_tokens + self.dropout(text_context)
        ).masked_fill(relation_padding.unsqueeze(-1), 0.0)

        fine_scale = torch.tanh(self.fine_residual_scale)
        object_scale = torch.tanh(self.object_residual_scale)
        fine_base = fine_tokens.mean(1)
        # OTA produces a relation-aware object sequence, but it should not
        # erase the reliable raw crop evidence. Use the raw object summary as
        # the residual anchor whenever the bridge supplies it.
        object_reference = (
            raw_object_tokens if raw_object_tokens is not None else object_tokens
        )
        object_base = object_reference.mean(1)
        # The pooled summaries must follow the attention topology. The
        # caption-side state has attended to the image stream, so it is the
        # text evidence paired with the fine image. The relation-side state
        # has attended to the OTA-aligned object stream, so it remains paired
        # with object features. Retain the legacy branch for frozen/reproducible
        # pre-V14 configurations only.
        fine_text_summary = masked_mean(relation_tokens, relation_mask)
        fine_pair = self.fine_pair_norm(
            fine_tokens.mean(1)
            + fine_text_summary
        )
        object_pair = self.object_pair_norm(
            object_tokens.mean(1) + masked_mean(relation_tokens, relation_mask)
        )
        fine_feature = fine_base + fine_scale * (fine_pair - fine_base)
        object_feature = object_base + object_scale * (object_pair - object_base)
        return fine_feature, object_feature


class BidirectionalOTABridge(nn.Module):
    """Pure Sinkhorn alignment with no hidden gate or prototype operation."""

    def __init__(self, hidden_size=HIDDEN, regularization=0.1, iterations=20,
                 residual_scale_init=0.0):
        super().__init__()
        self.regularization = regularization
        self.iterations = iterations
        self.main_norm = nn.LayerNorm(hidden_size)
        # Learned coupling space for the transport cost.  Sinkhorn transport
        # itself stays pure; only the features that define the alignment cost
        # are projected so the model can learn what "transportable" means.
        # The projection starts from the identity so the first step is the
        # pure Sinkhorn alignment of the raw features, and gradients refine
        # the coupling space through episodic training.
        self.cost_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.cost_projection.weight)
        # A zero-starting transport residual prevents a randomly initialized
        # bridge from overwriting the loaded image stream before adaptation.
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )
        self.capture_diagnostics = False
        self.last_diagnostics = None

    def forward(
        self, fine_tokens, object_tokens, objects, object_slot_mask=None,
        sample_validity=None,
    ):
        batch, fine_length, _ = fine_tokens.shape
        if object_tokens.size(0) != batch * objects:
            raise ValueError(
                "object token batch must equal batch * objects: "
                f"{object_tokens.size(0)} != {batch} * {objects}"
            )
        tokens_per_object = object_tokens.size(1)
        raw_object_by_slot = object_tokens.reshape(
            batch, objects, tokens_per_object, object_tokens.size(-1)
        )
        token_weights = None
        if object_slot_mask is not None:
            if object_slot_mask.shape != (batch, objects):
                raise ValueError(
                    "object_slot_mask must have shape (batch, objects): "
                    f"got {tuple(object_slot_mask.shape)}, expected {(batch, objects)}"
                )
            slot_weights = object_slot_mask.to(
                device=object_tokens.device, dtype=object_tokens.dtype
            ).clamp_min(0.0)
            # A missing crop is represented by a zero tensor in the loader.
            # Remove its ResNet bias response before computing the OT cost.
            raw_object_by_slot = raw_object_by_slot * slot_weights[
                :, :, None, None
            ]
            token_weights = slot_weights.unsqueeze(-1).expand(
                -1, -1, tokens_per_object
            ).reshape(batch, -1)
            # All current MNRE examples have at least one crop. Retain a
            # finite zero-evidence fallback for malformed future examples.
            empty_rows = token_weights.sum(dim=-1, keepdim=True) <= 1e-8
            fallback = torch.zeros_like(token_weights)
            fallback[:, 0] = 1.0
            token_weights = torch.where(empty_rows, fallback, token_weights)
        raw_object = raw_object_by_slot.reshape(batch, -1, object_tokens.size(-1))
        object_length = raw_object.size(1)

        fine_normalized = F.normalize(
            self.cost_projection(fine_tokens.float()), dim=-1
        )
        object_normalized = F.normalize(
            self.cost_projection(raw_object.float()), dim=-1
        )
        cost = 1.0 - torch.bmm(
            fine_normalized, object_normalized.transpose(1, 2)
        )
        source_marginal = torch.full(
            (batch, fine_length),
            1.0 / fine_length,
            dtype=cost.dtype,
            device=cost.device,
        )
        if token_weights is None:
            target_marginal = torch.full(
                (batch, object_length),
                1.0 / object_length,
                dtype=cost.dtype,
                device=cost.device,
            )
        else:
            target_marginal = token_weights.to(dtype=cost.dtype) / token_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
        kernel = torch.exp(-cost / self.regularization).clamp_min(1e-8)
        source_scale = source_marginal.clone()
        for _ in range(self.iterations):
            target_scale = target_marginal / (
                torch.bmm(
                    kernel.transpose(1, 2), source_scale.unsqueeze(-1)
                ).squeeze(-1)
                + 1e-8
            )
            source_scale = source_marginal / (
                torch.bmm(kernel, target_scale.unsqueeze(-1)).squeeze(-1)
                + 1e-8
            )
        joint_plan = (
            source_scale.unsqueeze(-1)
            * kernel
            * target_scale.unsqueeze(1)
        )
        transport_plan = joint_plan / joint_plan.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        aligned_object = torch.bmm(
            transport_plan.to(raw_object.dtype), raw_object
        )
        scale = torch.tanh(self.residual_scale)
        enhanced_fine = fine_tokens + scale * self.main_norm(aligned_object)
        transport_cost = (joint_plan * cost).sum((1, 2))
        consistency_loss = 1.0 - F.cosine_similarity(
            F.normalize(fine_tokens.mean(1), dim=-1),
            F.normalize(aligned_object.mean(1), dim=-1),
            dim=-1,
        )
        per_example_ot_loss = transport_cost + 0.5 * consistency_loss
        if sample_validity is None:
            ot_loss = per_example_ot_loss.mean()
        else:
            sample_validity = sample_validity.reshape(-1).to(
                device=per_example_ot_loss.device,
                dtype=per_example_ot_loss.dtype,
            ).clamp(0.0, 1.0)
            ot_loss = (
                per_example_ot_loss * sample_validity
            ).sum() / sample_validity.sum().clamp_min(1.0)
        if self.capture_diagnostics:
            self.last_diagnostics = {
                "transport_plan": transport_plan.detach().cpu(),
                "cost": cost.detach().cpu(),
                "fine_tokens": fine_tokens.detach().cpu(),
                "raw_object_tokens": raw_object.detach().cpu(),
                "object_slot_mask": (
                    None if object_slot_mask is None else object_slot_mask.detach().cpu()
                ),
            }
        return enhanced_fine, aligned_object, transport_plan, raw_object, ot_loss


class EntityAnchoredTextEvidenceRouter(nn.Module):
    """Distill relation-bearing evidence from the three textual views.

    The ordered head-tail pair defines a directional relation anchor. For each
    textual view, the anchor identifies supportive tokens and a counterfactual
    background distribution. The resulting evidence vectors are routed using
    both entity-pair compatibility and cross-view agreement. The module adds a
    small residual to the complete multimodal representation; it does not
    replace the image/object path or alter OTA and paired attention.
    """

    SOURCE_NAMES = ("relation", "phrase", "caption")

    def __init__(
        self,
        hidden_size=HIDDEN,
        bottleneck_size=256,
        output_size=HIDDEN * 2,
        dropout=0.1,
        routing_temperature=0.7,
        contrastive_margin=0.1,
        residual_scale_init=0.05,
        phrase_scale_init=0.5,
    ):
        super().__init__()
        self.routing_temperature = routing_temperature
        self.contrastive_margin = contrastive_margin
        self.dropout = nn.Dropout(dropout)
        self.anchor_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, bottleneck_size),
            nn.GELU(),
            nn.LayerNorm(bottleneck_size),
        )
        self.visual_anchor_projection = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, bottleneck_size),
            nn.GELU(),
            nn.LayerNorm(bottleneck_size),
        )
        nn.init.zeros_(self.visual_anchor_projection[1].weight)
        nn.init.zeros_(self.visual_anchor_projection[1].bias)
        self.visual_anchor_scale = nn.Parameter(torch.tensor(0.2))
        self.token_key = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.token_value = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.source_embeddings = nn.Parameter(
            torch.empty(len(self.SOURCE_NAMES), bottleneck_size)
        )
        self.contrast_strength = nn.Parameter(
            torch.zeros(len(self.SOURCE_NAMES))
        )
        self.consensus_strength = nn.Parameter(torch.tensor(0.0))
        self.evidence_norms = nn.ModuleList(
            nn.LayerNorm(bottleneck_size) for _ in self.SOURCE_NAMES
        )
        # Cross-source interaction: the three extracted evidence vectors
        # exchange information before routing, so routing sees source-aware
        # evidence instead of three independent extractions.
        self.cross_source_attention = nn.MultiheadAttention(
            bottleneck_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_source_norm = nn.LayerNorm(bottleneck_size)
        # The grounding-phrase branch lives inside this module: full restores
        # phrase evidence on top of the phrase-free base representation.
        self.phrase_projection = nn.Linear(hidden_size, output_size)
        self.phrase_scale = nn.Parameter(
            torch.full((output_size,), float(phrase_scale_init))
        )
        self.residual_projection = nn.Sequential(
            nn.LayerNorm(bottleneck_size * 4),
            nn.Linear(bottleneck_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        # LayerScale keeps the new route residual at a controlled magnitude at
        # initialization while allowing every output dimension to adapt.
        self.residual_scale = nn.Parameter(
            torch.full((output_size,), float(residual_scale_init))
        )
        nn.init.normal_(self.source_embeddings, std=0.02)

    @staticmethod
    def _masked_softmax(logits, mask):
        mask = mask.to(device=logits.device, dtype=torch.bool)
        if mask.ndim != 2 or mask.shape != logits.shape:
            raise ValueError(
                "text mask must match token logits: "
                f"got {tuple(mask.shape)} and {tuple(logits.shape)}"
            )
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("each text source must contain at least one valid token")
        minimum = torch.finfo(logits.dtype).min
        return torch.softmax(logits.masked_fill(~mask, minimum), dim=-1)

    def _extract_source(self, tokens, mask, anchor, source_index):
        source_type = self.source_embeddings[source_index]
        keys = self.token_key(tokens) + source_type
        values = self.token_value(tokens) + source_type
        normalized_keys = F.normalize(keys, dim=-1)
        normalized_anchor = F.normalize(anchor, dim=-1).unsqueeze(1)
        logits = (normalized_keys * normalized_anchor).sum(-1)

        supportive_weights = self._masked_softmax(logits, mask)
        background_weights = self._masked_softmax(-logits, mask)
        supportive = torch.bmm(supportive_weights.unsqueeze(1), values).squeeze(1)
        background = torch.bmm(background_weights.unsqueeze(1), values).squeeze(1)
        contrast = torch.sigmoid(self.contrast_strength[source_index])
        evidence = self.evidence_norms[source_index](
            supportive - contrast * background
        )

        supportive_similarity = F.cosine_similarity(supportive, anchor, dim=-1)
        background_similarity = F.cosine_similarity(background, anchor, dim=-1)
        informative = mask.to(dtype=torch.long).sum(-1) > 1
        ranking_loss = F.relu(
            self.contrastive_margin
            + background_similarity
            - supportive_similarity
        )
        ranking_loss = ranking_loss * informative.to(ranking_loss.dtype)
        return evidence, ranking_loss, informative

    def forward(
        self,
        head_feature,
        tail_feature,
        relation_tokens,
        relation_mask,
        phrase_tokens,
        phrase_mask,
        caption_tokens,
        caption_mask,
        visual_fine_feature=None,
        visual_object_feature=None,
    ):
        pair_input = torch.cat(
            [
                head_feature,
                tail_feature,
                head_feature - tail_feature,
                head_feature * tail_feature,
            ],
            dim=-1,
        )
        anchor = self.anchor_projection(pair_input)
        if visual_fine_feature is None or visual_object_feature is None:
            raise ValueError("Full EAT requires both visual evidence streams")
        visual_pair = torch.cat(
            [visual_fine_feature, visual_object_feature], dim=-1
        )
        visual_anchor = self.visual_anchor_projection(visual_pair)
        anchor = anchor + torch.tanh(self.visual_anchor_scale) * visual_anchor
        sources = (
            (relation_tokens, relation_mask),
            (phrase_tokens, phrase_mask),
            (caption_tokens, caption_mask),
        )
        extracted = [
            self._extract_source(tokens, mask, anchor, source_index)
            for source_index, (tokens, mask) in enumerate(sources)
        ]
        evidence = torch.stack([item[0] for item in extracted], dim=1)

        source_context, _ = self.cross_source_attention(
            evidence,
            evidence,
            evidence,
            need_weights=False,
        )
        evidence = self.cross_source_norm(evidence + self.dropout(source_context))

        normalized_evidence = F.normalize(evidence, dim=-1)
        normalized_anchor = F.normalize(anchor, dim=-1)
        anchor_compatibility = torch.bmm(
            normalized_evidence, normalized_anchor.unsqueeze(-1)
        ).squeeze(-1)
        agreement = torch.bmm(
            normalized_evidence, normalized_evidence.transpose(1, 2)
        )
        identity = torch.eye(
            agreement.size(1), device=agreement.device, dtype=agreement.dtype
        ).unsqueeze(0)
        cross_source_consensus = (
            agreement * (1.0 - identity)
        ).sum(-1) / (agreement.size(1) - 1)
        consensus_scale = torch.sigmoid(self.consensus_strength)
        routing_logits = (
            anchor_compatibility + consensus_scale * cross_source_consensus
        ) / self.routing_temperature
        source_weights = torch.softmax(routing_logits, dim=-1)
        routed_evidence = torch.bmm(
            source_weights.unsqueeze(1), evidence
        ).squeeze(1)

        interaction = torch.cat(
            [
                anchor,
                routed_evidence,
                anchor * routed_evidence,
                torch.abs(anchor - routed_evidence),
            ],
            dim=-1,
        )
        residual = self.residual_scale * self.residual_projection(interaction)
        # Restore the grounding-phrase evidence as a second residual path.
        # The phrase-free base has no phrase branch; only full (with this
        # module) sees phrase evidence, keeping full = base + modules.
        phrase_summary = masked_mean(phrase_tokens, phrase_mask)
        residual = residual + self.phrase_scale * self.phrase_projection(
            phrase_summary
        )

        ranking_losses = torch.stack([item[1] for item in extracted], dim=1)
        informative = torch.stack([item[2] for item in extracted], dim=1)
        informative_count = informative.sum().clamp_min(1)
        evidence_loss = ranking_losses.sum() / informative_count
        diagnostics = {
            "source_weights": source_weights,
            "anchor_compatibility": anchor_compatibility,
            "cross_source_consensus": cross_source_consensus,
        }
        return residual, evidence_loss, diagnostics


class FullOTER(nn.Module):
    """Fixed Full model: Sinkhorn OTA + paired attention + EAT router."""

    hidden_size = HIDDEN * 2

    def __init__(
        self,
        max_length,
        pretrain_path,
        blank_padding=True,
        mask_entity=False,
        vision_pretrained_path=None,
        freeze_vision_bn=True,
        attention_residual_scale=0.0,
        ota_residual_scale=0.0,
        text_router_residual_scale=0.05,
        text_evidence_loss_weight=0.01,
        text_router_temperature=0.7,
        text_router_margin=0.1,
        text_router_phrase_scale=0.5,
        phrase_feature_scale=0.80,
        text_feature_scale=0.85,
        visual_feature_scale=0.90,
        phrase_mode="legacy",
        phrase_max_length=24,
        relation_context_length=40,
        bert_trainable_layers=3,
    ):
        nn.Module.__init__(self)
        self.max_length = max_length
        self.blank_padding = blank_padding
        self.mask_entity = mask_entity
        self.freeze_vision_bn = freeze_vision_bn
        if phrase_mode not in {"legacy", "entity_context", "entity_compact"}:
            raise ValueError(f"Unknown phrase_mode: {phrase_mode}")
        if phrase_max_length < 4:
            raise ValueError("phrase_max_length must be at least 4")
        if not 8 <= relation_context_length <= max_length:
            raise ValueError(
                "relation_context_length must be in [8, max_length]"
            )
        if not 0 <= bert_trainable_layers <= 12:
            raise ValueError("bert_trainable_layers must be in [0, 12]")
        self.phrase_mode = phrase_mode
        self.phrase_max_length = phrase_max_length
        # Keep the MNRE attention dataflow unchanged while making its relation
        # evidence window explicit.  FewRel sentences are longer than MNRE in
        # a non-negligible number of cases, so a hard-coded 40-token prefix can
        # hide an entity from both paired attention and the text router.
        self.relation_context_length = relation_context_length
        self.bert_trainable_layers = bert_trainable_layers

        self.model_resnet50 = timm.create_model("resnet50", pretrained=False)
        if vision_pretrained_path:
            if not os.path.exists(vision_pretrained_path):
                raise FileNotFoundError(vision_pretrained_path)
            state = torch.load(
                vision_pretrained_path, map_location="cpu", weights_only=True
            )
            state = state.get("state_dict", state)
            self.model_resnet50.load_state_dict(state, strict=True)
            logging.info("Loaded ImageNet ResNet-50 from %s", vision_pretrained_path)
        for parameter_name, parameter in self.model_resnet50.named_parameters():
            parameter.requires_grad = "layer4" in parameter_name

        self.bert = transformers.BertModel.from_pretrained(pretrain_path)
        self.bert2 = self.bert
        first_trainable_bert_layer = 12 - bert_trainable_layers
        for parameter_name, parameter in self.bert.named_parameters():
            if "embeddings" in parameter_name:
                parameter.requires_grad = False
            else:
                layer_index = next(
                    (
                        layer
                        for layer in range(12)
                        if f"encoder.layer.{layer}." in parameter_name
                    ),
                    None,
                )
                parameter.requires_grad = (
                    layer_index is None or layer_index >= first_trainable_bert_layer
                )
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

        self.spatial_aligner = SpatialSequenceEncoder_Image(2048, HIDDEN)
        self.ota_bridge = BidirectionalOTABridge(
            HIDDEN, residual_scale_init=ota_residual_scale
        )
        self.cross_attention = PairedSymmetricCrossAttention(
            HIDDEN,
            num_heads=8,
            dropout=0.2,
            residual_scale_init=attention_residual_scale,
        )
        self.text_evidence_router = EntityAnchoredTextEvidenceRouter(
            HIDDEN,
            routing_temperature=text_router_temperature,
            contrastive_margin=text_router_margin,
            residual_scale_init=text_router_residual_scale,
            phrase_scale_init=text_router_phrase_scale,
        )
        self.linear_phrases = nn.Linear(HIDDEN, self.hidden_size)
        self.linear_final = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout_linear = nn.Dropout(0.2)

        self.text_feature_scale = float(text_feature_scale)
        self.phrase_feature_scale = phrase_feature_scale
        self.visual_feature_scale = float(visual_feature_scale)
        self.ot_loss_weight = 0.1
        self.text_evidence_loss_weight = text_evidence_loss_weight
        # Runtime-only switch used by visualization scripts. It is deliberately
        # false by default so training memory/compute and checkpoints remain
        # unchanged.
        self.capture_diagnostics = False

    def enable_diagnostics(self, enabled=True):
        """Capture OTA/paired-attention maps for inspection or plotting."""
        self.capture_diagnostics = bool(enabled)
        self.ota_bridge.capture_diagnostics = bool(enabled)
        self.cross_attention.capture_diagnostics = bool(enabled)
        if not enabled:
            self.ota_bridge.last_diagnostics = None
            self.cross_attention.last_attention = None
        return self

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_vision_bn:
            for module in self.model_resnet50.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self,
        token,
        att_mask,
        pos1,
        pos2,
        token_phrase,
        att_mask_phrase,
        image_ori,
        image_ori_objects,
        caption_input_ids,
        caption_token_type_ids,
        caption_attention_mask,
        object_slot_mask=None,
        return_modal_features=False,
    ):
        if object_slot_mask is not None and object_slot_mask.size(-1) >= 5:
            visual_available = object_slot_mask[:, 4:5].to(
                device=image_ori.device, dtype=image_ori.dtype
            ).clamp(0.0, 1.0)
        else:
            # Native MNRE examples always contain the original image; its
            # three-value mask only describes which object slots are valid.
            visual_available = image_ori.new_ones((image_ori.size(0), 1))
        caption_tokens = self.bert2(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            token_type_ids=caption_token_type_ids,
        ).last_hidden_state
        relation_limit = self.relation_context_length
        relation_mask = att_mask[:, :relation_limit]
        relation_tokens = self.bert2(
            token[:, :relation_limit], attention_mask=relation_mask
        ).last_hidden_state
        fine_map = self.model_resnet50.forward_features(image_ori)
        fine_tokens, _ = self.spatial_aligner(fine_map)
        batch = fine_tokens.size(0)
        object_map = self.model_resnet50.forward_features(image_ori_objects)
        object_tokens, _ = self.spatial_aligner(object_map)
        objects = object_tokens.size(0) // batch
        (
            fine_context_tokens,
            object_context_tokens,
            _,
            raw_object_tokens,
            ot_loss,
        ) = self.ota_bridge(
            fine_tokens,
            object_tokens,
            objects,
            object_slot_mask=None,
            sample_validity=visual_available,
        )
        fine_feature, object_feature = self.cross_attention(
            fine_context_tokens,
            object_context_tokens,
            caption_tokens,
            relation_tokens,
            caption_attention_mask,
            relation_mask,
            raw_object_tokens=raw_object_tokens,
        )
        fine_feature = torch.tanh(fine_feature)
        object_feature = torch.tanh(object_feature)

        # Official FewRel text augmentation has no paired visual record.  A
        # zero image otherwise becomes a non-zero ResNet/BN bias feature and
        # is mistakenly consumed by OTA, paired attention and the router.
        # Mask only those explicitly text-only records; native MNRE and all
        # ordinary multimodal FewRel records retain exactly the old function.
        fine_feature = fine_feature * visual_available
        object_feature = object_feature * visual_available

        hidden_text = self.bert(token, attention_mask=att_mask).last_hidden_state
        head_mask = torch.zeros(
            hidden_text.shape[:2], device=hidden_text.device
        ).scatter_(1, pos1, 1.0)
        tail_mask = torch.zeros(
            hidden_text.shape[:2], device=hidden_text.device
        ).scatter_(1, pos2, 1.0)
        head_hidden = (head_mask.unsqueeze(-1) * hidden_text).sum(1)
        tail_hidden = (tail_mask.unsqueeze(-1) * hidden_text).sum(1)
        text_feature = self.text_feature_scale * torch.cat(
            [head_hidden, tail_hidden], dim=-1
        )

        phrase_tokens = self.bert(
            token_phrase, attention_mask=att_mask_phrase
        ).last_hidden_state
        phrase_feature = self.phrase_feature_scale * self.linear_phrases(
            masked_mean(phrase_tokens, att_mask_phrase)
        )
        visual_feature = self.visual_feature_scale * torch.tanh(
            torch.cat([fine_feature, object_feature], dim=-1)
        )
        representation = self.linear_final(
            self.dropout_linear(
                torch.cat([text_feature, phrase_feature, visual_feature], dim=-1)
            )
        )
        text_residual, text_evidence_loss, text_evidence_diagnostics = (
            self.text_evidence_router(
                head_hidden,
                tail_hidden,
                relation_tokens,
                relation_mask,
                phrase_tokens,
                att_mask_phrase,
                caption_tokens,
                caption_attention_mask,
                visual_fine_feature=fine_feature,
                visual_object_feature=object_feature,
            )
        )
        representation = representation + text_residual
        auxiliary_loss = (
            self.ot_loss_weight * ot_loss
            + self.text_evidence_loss_weight * text_evidence_loss
        )
        if return_modal_features:
            modal_features = {
                "visual": visual_feature,
                "visual_available": visual_available.to(visual_feature.dtype),
            }
            if text_evidence_diagnostics is not None:
                modal_features.update(
                    {
                        "text_source_weights": text_evidence_diagnostics[
                            "source_weights"
                        ],
                        "text_anchor_compatibility": text_evidence_diagnostics[
                            "anchor_compatibility"
                        ],
                        "text_cross_source_consensus": text_evidence_diagnostics[
                            "cross_source_consensus"
                        ],
                    }
                )
            return representation, auxiliary_loss, modal_features
        return representation, auxiliary_loss

    def tokenize(self, item):
        """Encode entity-marked relation text and its grounding phrase locally."""
        if len(item) == 5 and not isinstance(item, dict):
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(item)
            return torch.tensor(indexed_tokens).long().unsqueeze(0)

        sentence = item["text"] if "text" in item else item["token"]
        is_token = "token" in item
        pos_head, pos_tail = item["h"]["pos"], item["t"]["pos"]
        pos_min, pos_max, rev = (
            (pos_tail, pos_head, True)
            if pos_head[0] > pos_tail[0]
            else (pos_head, pos_tail, False)
        )

        if not is_token:
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
        else:
            sent0 = self.tokenizer.tokenize(" ".join(sentence[:pos_min[0]]))
            ent0 = self.tokenizer.tokenize(" ".join(sentence[pos_min[0]:pos_min[1]]))
            sent1 = self.tokenizer.tokenize(" ".join(sentence[pos_min[1]:pos_max[0]]))
            ent1 = self.tokenizer.tokenize(" ".join(sentence[pos_max[0]:pos_max[1]]))
            sent2 = self.tokenizer.tokenize(" ".join(sentence[pos_max[1]:]))

        if self.mask_entity:
            ent0 = ["[unused4]"] if not rev else ["[unused5]"]
            ent1 = ["[unused5]"] if not rev else ["[unused4]"]
        else:
            ent0 = (
                ["[unused0]"] + ent0 + ["[unused1]"]
                if not rev else ["[unused2]"] + ent0 + ["[unused3]"]
            )
            ent1 = (
                ["[unused2]"] + ent1 + ["[unused3]"]
                if not rev else ["[unused0]"] + ent1 + ["[unused1]"]
            )

        relation_tokens = ["[CLS]"] + sent0 + ent0 + sent1 + ent1 + sent2 + ["[SEP]"]
        pos1 = 1 + len(sent0) if not rev else 1 + len(sent0 + ent0 + sent1)
        pos2 = 1 + len(sent0 + ent0 + sent1) if not rev else 1 + len(sent0)
        pos1, pos2 = min(self.max_length - 1, pos1), min(self.max_length - 1, pos2)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(relation_tokens)
        available_length = len(indexed_tokens)
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)
        att_mask = torch.zeros(indexed_tokens.size()).long()
        att_mask[0, :available_length] = 1

        if self.phrase_mode == "legacy":
            # Kept byte-for-byte compatible with the historical FewRel setup.
            phrases = item["grounding"]
            token_phrases = self.tokenizer.convert_tokens_to_ids(phrases.split(" "))
            phrase_length = min(len(phrases.split(" ")), 6)
            phrase_limit = 6
        elif self.phrase_mode == "entity_context":
            # Build a short, entity-marked local relation context.  The former
            # implementation only encoded the first six *raw* sentence words,
            # which usually omitted both entities and bypassed WordPiece.
            source_tokens = sentence if is_token else sentence.split()
            head_start, head_end = pos_head
            tail_start, tail_end = pos_tail
            left = max(0, min(head_start, tail_start) - 4)
            right = min(len(source_tokens), max(head_end, tail_end) + 4)
            phrase_tokens = ["[CLS]"]
            for index in range(left, right):
                if index == head_start:
                    phrase_tokens.append("[unused0]")
                if index == tail_start:
                    phrase_tokens.append("[unused2]")
                phrase_tokens.extend(self.tokenizer.tokenize(str(source_tokens[index])))
                if index + 1 == head_end:
                    phrase_tokens.append("[unused1]")
                if index + 1 == tail_end:
                    phrase_tokens.append("[unused3]")
            phrase_tokens.append("[SEP]")
            phrase_limit = self.phrase_max_length
            phrase_tokens = phrase_tokens[:phrase_limit]
            phrase_length = len(phrase_tokens)
            token_phrases = self.tokenizer.convert_tokens_to_ids(phrase_tokens)
        else:
            # Evidence-budgeted entity bridge for FewRel.  A contiguous
            # 24-token window drops a complete entity marker pair in roughly
            # one quarter of the current data, while simply enlarging the
            # window introduces unrelated context.  Keep both marked mentions
            # unconditionally and spend the remaining budget on relation words
            # between them, retaining both ends if the bridge is long.
            source_tokens = sentence if is_token else sentence.split()
            head_start, head_end = pos_head
            tail_start, tail_end = pos_tail

            def marked_mention(start, end, is_head):
                result = ["[unused0]" if is_head else "[unused2]"]
                for source_token in source_tokens[start:end]:
                    result.extend(self.tokenizer.tokenize(str(source_token)))
                result.append("[unused1]" if is_head else "[unused3]")
                return result

            if head_start <= tail_start:
                left = marked_mention(head_start, head_end, True)
                right = marked_mention(tail_start, tail_end, False)
                between_source = source_tokens[head_end:tail_start]
            else:
                left = marked_mention(tail_start, tail_end, False)
                right = marked_mention(head_start, head_end, True)
                between_source = source_tokens[tail_end:head_start]
            bridge = []
            for source_token in between_source:
                bridge.extend(self.tokenizer.tokenize(str(source_token)))

            phrase_limit = self.phrase_max_length
            bridge_budget = max(0, phrase_limit - len(left) - len(right) - 2)
            if len(bridge) > bridge_budget:
                if bridge_budget >= 3:
                    content_budget = bridge_budget - 1
                    left_budget = (content_budget + 1) // 2
                    right_budget = content_budget - left_budget
                    bridge = (
                        bridge[:left_budget]
                        + ["[unused6]"]
                        + (bridge[-right_budget:] if right_budget else [])
                    )
                else:
                    bridge = bridge[:bridge_budget]
            phrase_tokens = ["[CLS]"] + left + bridge + right + ["[SEP]"]
            # Extremely long entity names are rare; truncate mention content,
            # never the four boundary markers, in that case.
            if len(phrase_tokens) > phrase_limit:
                left_content = left[1:-1]
                right_content = right[1:-1]
                content_budget = max(0, phrase_limit - 6)
                left_budget = min(
                    len(left_content), (content_budget + 1) // 2
                )
                right_budget = min(
                    len(right_content), content_budget - left_budget
                )
                remaining = content_budget - left_budget - right_budget
                left_budget += min(
                    remaining, len(left_content) - left_budget
                )
                remaining = content_budget - left_budget - right_budget
                right_budget += min(
                    remaining, len(right_content) - right_budget
                )
                phrase_tokens = (
                    ["[CLS]", left[0]]
                    + left_content[:left_budget]
                    + [left[-1], right[0]]
                    + right_content[:right_budget]
                    + [right[-1], "[SEP]"]
                )
            phrase_length = len(phrase_tokens)
            token_phrases = self.tokenizer.convert_tokens_to_ids(phrase_tokens)

        while len(token_phrases) < phrase_limit:
            token_phrases.append(0)
        token_phrases = torch.tensor(token_phrases[:phrase_limit]).long().unsqueeze(0)
        att_mask_phrases = torch.zeros(token_phrases.size()).long()
        att_mask_phrases[0, :phrase_length] = 1
        return indexed_tokens, att_mask, pos1, pos2, token_phrases, att_mask_phrases
