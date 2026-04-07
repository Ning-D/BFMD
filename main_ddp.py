# =======================
# MUST be first
# =======================
import comet_ml
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =======================
# NCCL stability
# =======================
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "0"
os.environ["NCCL_SOCKET_IFNAME"] = "lo"
os.environ["NCCL_DEBUG"] = "WARN"

# =======================
# Normal imports
# =======================
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets.ls_caption_dataset import LSCaptionDataset, build_splits
from train_ddp import CaptionLightningModule

# =======================
# Config
# =======================
TOKENIZER_NAME = "bert-base-uncased"
BATCH_SIZE = 32
EPOCHS = 10
NUM_GPUS = 2
# MODEL_TYPE = "gpt"
MODEL_TYPE = "videomae"


# =======================
# Config via argparse
# =======================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        default="videomae",
        choices=["videomae", "gpt"],
        help="Model type to use",
    )

    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument(
        "--use_features",
        action="store_true",
        help="Use pre-extracted features instead of raw video"
    )

    return parser.parse_args()





def main():

    args = parse_args()

    print(f"🚀 Model type: {args.model_type}")
    print(f"🖥️ GPUs: {args.gpus}")
    print(f"📦 Batch size: {args.batch_size}")

    print("🚀 Initializing tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    special_tokens = {
        "additional_special_tokens": ["[PLAYER]"]
    }
    tokenizer.add_special_tokens(special_tokens)

    print("🧪 Initializing Comet")
    comet = CometLogger(
        project_name="racket-video-caption",
        experiment_name=f"{args.model_type}_run_reflection",
    )

    print("📦 Loading dataset")

    if args.model_type == "gpt":
        dataset = LSCaptionDataset(return_frames=False)

    else:
        if args.use_features:
            print("⚡ Using pre-extracted features")
            from datasets.ls_feature_dataset import LSCaptionFeatureDataset
            FEATURE_ROOT = "./features"

            dataset = LSCaptionFeatureDataset(
                feature_root=FEATURE_ROOT,
                tokenizer=tokenizer,
            )


        else:
            print("🎥 Using raw video frames")
            dataset = LSCaptionDataset()

    print(f"🎬 Data mode: {'feature' if args.use_features else 'raw'}")

    train_set, val_set, test_set = build_splits(dataset)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print("🧠 Initializing model")
    model = CaptionLightningModule(
        vocab_size=len(tokenizer),
        tokenizer=tokenizer,
        model_type=args.model_type,
        num_val_samples=2,
    )

    print("⚡ Initializing Trainer")

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val/loss",  # 你 log 的名字
    #     mode="min",
    #     save_top_k=1,
    #     filename="best",
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best",
        dirpath="checkpoints",  # ⭐ 明确保存路径
    )

    # GPT 不用 DDP
    if args.model_type == "gpt":
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=comet,
            enable_checkpointing=False,
            enable_model_summary=False,
        )

        print("🧪 GPT Testing start")
        trainer.test(model, test_loader)

    else:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=args.gpus,
            strategy="ddp_find_unused_parameters_true",
            precision="bf16-mixed",
            max_epochs=args.epochs,
            logger=comet,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            # enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[checkpoint_callback],
        )

        print("🏃 Training start")
        trainer.fit(model, train_loader, val_loader)

        print("🧪 Testing start")
        trainer.test(
            model=None,
            dataloaders=test_loader,
            ckpt_path="best"  # ⭐ 关键
        )
        # trainer.test(model, test_loader)





# def main():
#     print("🚀 Initializing tokenizer")
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
#
#     special_tokens = {
#         "additional_special_tokens": ["[PLAYER]"]
#     }
#
#     tokenizer.add_special_tokens(special_tokens)
#
#     print("🧪 Initializing Comet")
#     comet = CometLogger(
#         project_name="racket-video-caption",
#         experiment_name="videomae_caption_ddp",
#     )
#     print("🧪 Tokenizer test:")
#     print(tokenizer.tokenize("[PLAYER] hits a net shot"))
#
#     print("📦 Loading dataset")
#     dataset = LSCaptionDataset()
#     train_set, val_set, test_set = build_splits(dataset)
#     # =======================
#     # Dataset statistics
#     # =======================
#     print("📊 Dataset split sizes:")
#     print(f"  - Train samples: {len(train_set)}")
#     print(f"  - Val samples:   {len(val_set)}")
#     print(f"  - Test samples:  {len(test_set)}")
#
#     # ⭐ 调试阶段必须 num_workers=0
#     train_loader = DataLoader(
#         train_set,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=8,
#         pin_memory=True,
#     )
#
#     val_loader = DataLoader(
#         val_set,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=True,
#     )
#
#     test_loader = DataLoader(
#         test_set,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=8,
#         pin_memory=True,
#     )
#
#     print("🧠 Initializing model")
#     model = CaptionLightningModule(
#         vocab_size=len(tokenizer),
#         tokenizer=tokenizer,
#         model_type=MODEL_TYPE,  # ⭐ 加這行
#         num_val_samples=2,
#     )
#
#     print("⚡ Initializing Trainer")
#     trainer = pl.Trainer(
#         accelerator="gpu",
#         devices=NUM_GPUS,
#         strategy="ddp_find_unused_parameters_false",
#         precision="bf16-mixed",
#         max_epochs=EPOCHS,
#         logger=comet,
#         log_every_n_steps=1,
#
#         # ⭐⭐⭐ 关键三行 ⭐⭐⭐
#         check_val_every_n_epoch=1,  # 每个 epoch 都跑 validation
#         enable_checkpointing=False,  # 防止小数据集卡住
#         enable_model_summary=False,  # 日志更干净
#     )
#
#
#
#     print("🏃 Training start")
#     trainer.fit(model, train_loader, val_loader)
#
#     print("🧪 Testing start")
#     trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
