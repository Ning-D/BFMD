# models/videomae_caption.py

import torch
import torch.nn as nn
from transformers import VideoMAEModel

MAX_LEN = 120

class MemoryRefiner(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory):
        refined, _ = self.attn(memory, memory, memory)
        return self.norm(memory + refined)
# =========================================================
# Structured Semantic Label Builder
# =========================================================
def build_reflection_labels(text):
    text = text.lower()

    shot_keywords = {
        "serve": ["serve"],
        "flick_serve": ["flick serve"],
        "smash": ["smash", "full-power", "downward shot"],
        "clear": ["clear"],
        "drop": ["drop shot", "drop"],
        "lift": ["lift"],
        "drive": ["drive"],
        "block": ["block"],
        "net_shot": ["net shot", "net_shot", "spinning net"],
        "net_kill": ["net kill", "finish the rally"],
    }

    shot_vec = []
    for key, words in shot_keywords.items():
        shot_vec.append(int(any(w in text for w in words)))

    high_arc = int("high" in text or "arc" in text)
    downward = int("downward" in text or "steep" in text)
    flat = int("flat" in text or "horizontal" in text)

    frontcourt = int("front" in text or "near the net" in text)
    midcourt = int("mid" in text)
    backcourt = int("backcourt" in text or "rear court" in text)

    offensive = int("attack" in text or "aggressive" in text or "full-power" in text)
    defensive = int("defensive" in text or "recover" in text)
    pressure = int("pressure" in text or "maintain pressure" in text)

    full_vec = (
        shot_vec
        + [high_arc, downward, flat]
        + [frontcourt, midcourt, backcourt]
        + [offensive, defensive, pressure]
    )

    return torch.tensor(full_vec)


# =========================================================
# Encoder
# =========================================================
class VideoMAEEncoder(nn.Module):
    def __init__(self, name="MCG-NJU/videomae-base"):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(name)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, frames):
        out = self.model(pixel_values=frames)
        return out.last_hidden_state  # (B, T, D)


# =========================================================
# Decoder
# =========================================================
class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=6, nhead=8):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(MAX_LEN, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, memory):
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)

        tgt = self.token_emb(tokens) + self.pos_emb(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tokens.device)

        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
        )

        logits = self.fc_out(hidden)
        return logits, hidden


class VideoCaptionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.encoder = VideoMAEEncoder()
        D = self.encoder.hidden_dim

        self.decoder = CaptionDecoder(
            vocab_size=vocab_size,
            d_model=D,
        )

        # ===== Semantic Critic =====
        self.reflection_classifier = nn.Linear(D, 19)

        # error → correction vector
        self.error_to_bias = nn.Sequential(
            nn.Linear(19, D),
            nn.Tanh(),
        )

        self.gate_proj = nn.Linear(D, 1)

        # Freeze encoder except last 2 layers
        for p in self.encoder.model.parameters():
            p.requires_grad = False

        for p in self.encoder.model.encoder.layer[-2:].parameters():
            p.requires_grad = True

    def forward(self, x, tokens, gt_sem=None, enable_reflection=True):

        memory = self.encoder(x)

        # -------------------------
        # PASS 1 — Base Decode
        # -------------------------
        logits1, hidden1 = self.decoder(tokens, memory)
        # hidden1: (B, L, D)

        # =========================
        # TOKEN-LEVEL REFLECTION
        # =========================
        reflection_logits = self.reflection_classifier(hidden1)
        reflection_probs = torch.sigmoid(reflection_logits)
        # (B, L, 19)

        if enable_reflection:

            if gt_sem is not None:
                # gt_sem should be (B, L, 19)
                error = gt_sem - reflection_probs
            else:
                error = -reflection_probs

            correction = self.error_to_bias(error)
            gate = torch.sigmoid(self.gate_proj(hidden1))

            global_correction = (gate * correction).mean(dim=1)
            memory_refined = memory + global_correction.unsqueeze(1)

            logits2, _ = self.decoder(tokens, memory_refined)

        else:
            logits2 = logits1

        return logits1, logits2, reflection_logits

    def generate(self, x, input_ids, max_length, eos_token_id):

        memory = self.encoder(x)
        generated = input_ids.clone()  # (B, 1)

        # =========================
        # PASS 1 — Autoregressive
        # =========================
        while generated.size(1) < max_length:

            logits, hidden = self.decoder(generated, memory)
            next_token = logits[:, -1].argmax(dim=-1)

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)],
                dim=1
            )

            if (next_token == eos_token_id).all():
                break

        # =========================
        # TOKEN-LEVEL REFLECTION
        # =========================
        logits1, hidden1 = self.decoder(generated, memory)

        reflection_probs = torch.sigmoid(
            self.reflection_classifier(hidden1)
        )

        error = -reflection_probs
        correction = self.error_to_bias(error)
        gate = torch.sigmoid(self.gate_proj(hidden1))

        global_correction = (gate * correction).mean(dim=1)
        memory_refined = memory + global_correction.unsqueeze(1)

        logits2, _ = self.decoder(generated, memory_refined)
        refined_tokens = logits2.argmax(dim=-1)

        return refined_tokens

