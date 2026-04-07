# train_ddp.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from models.videomae_caption import VideoCaptionModel
import torch.distributed as dist
from models.videomae_caption import build_reflection_labels
LR = 1e-4


class CaptionLightningModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        tokenizer,
        model_type: str = "videomae",
        num_val_samples: int = 2,
    ):
        super().__init__()

        self.model_type = model_type
        if model_type == "videomae":
            from models.videomae_caption import VideoCaptionModel
            self.model = VideoCaptionModel(len(tokenizer))
            # self.model = VideoCaptionModel(vocab_size)
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        elif model_type == "gpt":
            from models.model_gpt import GPTCaptionModel
            self.model = GPTCaptionModel()
            self.loss_fn = None  # GPT 不需要 loss

        else:
            raise ValueError("Unknown model_type")

        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.tokenizer = tokenizer

        self.bos_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.sep_token_id

        self.val_preds = []
        self.val_gts = []

        self.test_preds = []
        self.test_gts = []

        # ⭐ 现在是合法的了
        self.num_val_samples = num_val_samples
        self._val_debug_samples = []

        self.reflection_weight = 0.1
        self.warmup_epochs = 3

    # -------------------------
    # Train
    # -------------------------
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,   # ⭐ DDP 必须
        )
        return loss


    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        frames = batch["frames"]
        gt_ids = batch["input_ids"]

        input_ids = torch.full(
            (frames.size(0), 1),
            self.bos_token_id,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            pred_ids = self.model.generate(
                frames,
                input_ids,
                max_length=120,  # 可以稍微放大一点
                eos_token_id=self.eos_token_id
            )

        preds = [
            self.tokenizer.decode(p, skip_special_tokens=True)
            for p in pred_ids
        ]
        gts = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in gt_ids
        ]

        # ⭐ 所有 GPU 都存，用于 gather
        self.val_preds.extend(preds)
        self.val_gts.extend([[g] for g in gts])

        # ⭐ 只缓存少量 sample 用于打印
        if len(self._val_debug_samples) < self.num_val_samples:
            for gt, pred in zip(gts, preds):
                if len(self._val_debug_samples) < self.num_val_samples:
                    self._val_debug_samples.append((gt, pred))

    def on_validation_epoch_end(self):

        # ===============================
        # 1️⃣ Gather across GPUs
        # ===============================
        if dist.is_available() and dist.is_initialized():

            world_size = dist.get_world_size()

            gathered_preds = [None for _ in range(world_size)]
            gathered_gts = [None for _ in range(world_size)]

            dist.all_gather_object(gathered_preds, self.val_preds)
            dist.all_gather_object(gathered_gts, self.val_gts)

            all_preds = []
            all_gts = []

            for rank_preds in gathered_preds:
                all_preds.extend(rank_preds)

            for rank_gts in gathered_gts:
                all_gts.extend(rank_gts)

        else:
            all_preds = self.val_preds
            all_gts = self.val_gts

        # ===============================
        # 2️⃣ Only rank0 handles output
        # ===============================
        if self.trainer.is_global_zero:

            # 🔍 打印 sample
            for i, (gt, pred) in enumerate(self._val_debug_samples):
                print("\n" + "=" * 60)
                print(f"Epoch {self.current_epoch} | Sample {i}")
                print("\n[GT]")
                print(gt)
                print("\n[PRED]")
                print(pred)
                print("=" * 60)

            from caption_metrics import compute_caption_metrics

            scores = compute_caption_metrics(all_preds, all_gts)

            for k, v in scores.items():
                self.log(
                    f"val/{k}",
                    v,
                    prog_bar=True,
                    sync_dist=False,  # ❗必须 False
                )

        # ===============================
        # 3️⃣ Clear buffers
        # ===============================
        self.val_preds.clear()
        self.val_gts.clear()
        self._val_debug_samples.clear()


    def test_step(self, batch, batch_idx):

        # ======================================
        # 1️⃣ 如果是 VideoMAE
        # ======================================
        if self.model_type == "videomae":

            loss = self._shared_step(batch)

            self.log(
                "test/loss",
                loss,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            frames = batch["frames"]
            gt_ids = batch["input_ids"]

            input_ids = torch.full(
                (frames.size(0), 1),
                self.bos_token_id,
                dtype=torch.long,
                device=self.device,
            )

            with torch.no_grad():
                pred_ids = self.model.generate(
                    frames,
                    input_ids,
                    max_length=120,  # 可以稍微放大一点
                    eos_token_id=self.eos_token_id
                )

            preds = [
                self.tokenizer.decode(p, skip_special_tokens=True)
                for p in pred_ids
            ]

        # ======================================
        # 2️⃣ 如果是 GPT-4.1
        # ======================================
        elif self.model_type == "gpt":

            # ⚠️ GPT 不需要 loss
            preds = []

            video_paths = batch["video_path"]  # ⭐ dataset 必须返回

            for path in video_paths:
                caption = self.model.generate(path)
                preds.append(caption)

            gt_ids = batch["input_ids"]

        else:
            raise ValueError("Unknown model_type")

        # ======================================
        # 3️⃣ Ground Truth
        # ======================================
        gts = [
            [self.tokenizer.decode(t, skip_special_tokens=True)]
            for t in gt_ids
        ]

        # ======================================
        # 4️⃣ 存结果（用于 epoch end gather）
        # ======================================
        self.test_preds.extend(preds)
        self.test_gts.extend(gts)

    def on_test_epoch_end(self):

        # =========================================================
        # 1️⃣ Gather across GPUs
        # =========================================================
        if dist.is_available() and dist.is_initialized():

            world_size = dist.get_world_size()

            gathered_preds = [None for _ in range(world_size)]
            gathered_gts = [None for _ in range(world_size)]

            dist.all_gather_object(gathered_preds, self.test_preds)
            dist.all_gather_object(gathered_gts, self.test_gts)

            all_preds = []
            all_gts = []

            for rank_preds in gathered_preds:
                all_preds.extend(rank_preds)

            for rank_gts in gathered_gts:
                all_gts.extend(rank_gts)

        else:
            all_preds = self.test_preds
            all_gts = self.test_gts

        # =========================================================
        # 2️⃣ Only rank 0 computes metrics
        # =========================================================
        if self.trainer.is_global_zero:

            from caption_metrics import compute_caption_metrics

            scores = compute_caption_metrics(all_preds, all_gts)

            for k, v in scores.items():
                self.log(
                    f"test/{k}",
                    v,
                    prog_bar=True,
                    sync_dist=False,  # ❗必须 False
                )

        self.test_preds.clear()
        self.test_gts.clear()

    def _shared_step(self, batch):

        frames = batch["frames"]
        ids = batch["input_ids"]
        gt_texts = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in ids
        ]

        # =========================
        # Build sentence-level semantic
        # =========================
        gt_sem_sentence = torch.stack([
            build_reflection_labels(t)
            for t in gt_texts
        ]).float().to(self.device)  # (B, 19)

        L = ids.size(1) - 1
        gt_sem = gt_sem_sentence.unsqueeze(1).expand(-1, L, -1)
        # (B, L, 19)

        enable_reflection = self.current_epoch >= self.warmup_epochs

        logits1, logits2, reflection_logits = self.model(
            frames,
            ids[:, :-1],
            gt_sem=gt_sem if enable_reflection else None,
            enable_reflection=enable_reflection
        )

        # =========================
        # CE Loss
        # =========================
        loss_ce1 = self.loss_fn(
            logits1.reshape(-1, logits1.size(-1)),
            ids[:, 1:].reshape(-1),
        )

        loss_ce2 = self.loss_fn(
            logits2.reshape(-1, logits2.size(-1)),
            ids[:, 1:].reshape(-1),
        )

        loss_ce = 0.5 * loss_ce1 + 0.5 * loss_ce2

        # =========================
        # Semantic Loss (token-level)
        # =========================
        if enable_reflection:

            bce = nn.BCEWithLogitsLoss(reduction="none")

            loss_sem_all = bce(reflection_logits, gt_sem)
            # (B, L, 19)

            token_mask = (ids[:, 1:] != 0).unsqueeze(-1)
            # (B, L, 1)

            loss_sem = (loss_sem_all * token_mask).sum() / token_mask.sum()

            loss = loss_ce + 0.03 * loss_sem

        else:
            loss = loss_ce

        return loss



    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=LR)



