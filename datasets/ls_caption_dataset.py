# datasets/ls_caption_dataset.py
import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import AutoTokenizer

# JSON_ROOT = "/mnt/HDD12TB-1/ding/RacketSports/single-modified"
JSON_ROOT = "/mnt/HDD12TB-1/ding/RacketSports/single-modified-/raw/normalize"
VIDEO_ROOT = "/mnt/HDD12TB-1/ding/2026_Videos"

NUM_FRAMES = 16
IMG_SIZE = 224
MAX_LEN = 120
TOKENIZER_NAME = "bert-base-uncased"
from torch.utils.data import random_split



import random
from torch.utils.data import Subset

def build_splits(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    n = len(dataset)
    assert n > 0, "Dataset must contain at least 1 sample"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # 计算每个部分大小
    train_size = int(n * train_ratio)
    val_size   = int(n * val_ratio)
    test_size  = n - train_size - val_size  # 保证总数等于 n

    # 切分
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]
    test_idx  = indices[train_size + val_size:]

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    return train_set, val_set, test_set





def find_video(json_name: str) -> str:
    base = os.path.splitext(json_name)[0]
    for root, _, files in os.walk(VIDEO_ROOT):
        for f in files:
            if f.startswith(base) and f.endswith(".mp4"):
                return os.path.join(root, f)
    raise FileNotFoundError(base)


def group_by_frame(ls_json):
    m = defaultdict(list)
    for ann in ls_json.get("annotations", []):
        for r in ann.get("result", []):
            for rg in r.get("value", {}).get("ranges", []):
                m[int(rg["start"])].append(r)
    return m



# def sample_frames(center):
#     half = NUM_FRAMES // 2
#     return list(range(center - half, center + half))

def sample_frames(center):
    start = center - 3
    end = center + 12
    return list(range(start, end + 1))  # +1 因为 range 不包含右边界






class LSCaptionDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        for fname in os.listdir(JSON_ROOT):
            if not fname.endswith(".json"):
                continue

            with open(os.path.join(JSON_ROOT, fname)) as f:
                ls_json = json.load(f)

            video = find_video(fname)
            frame_map = group_by_frame(ls_json)

            for frame, results in frame_map.items():
                caption = select_final_caption(results)
                shot_type = select_shot_type(results)

                if caption and shot_type:
                    self.samples.append({
                        "video": video,
                        "frame": frame,
                        "caption": caption,
                        "shot_type": shot_type,
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frames = self._load_frames(s["video"], s["frame"])

        tokens = self.tokenizer(
            s["caption"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        return {
            "frames": frames,
            "input_ids": tokens["input_ids"][0],
            "shot_type": s["shot_type"],
            "video_path": s["video"],
        }
    def _load_frames(self, video, center):
        cap = cv2.VideoCapture(video)
        imgs = []

        for f in sample_frames(center):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if ok:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                imgs.append(img)

        cap.release()

        imgs = torch.from_numpy(
            torch.stack([torch.from_numpy(i) for i in imgs]).numpy()
        )
        imgs = imgs.permute(0, 3, 1, 2).float() / 255.0
        return imgs


def select_final_caption(results):
    """
    Final caption selection logic:
      1) shotDescriptionRefined  (人工修改后的最终版本)
      2) shotDescriptionClean    (匿名版本)
    """

    # 1️⃣ refined version（人工改过）
    for r in results:
        if (
            r.get("type") == "textarea"
            and r.get("from_name") == "shotDescriptionRefined"
        ):
            return r["value"]["text"][0]

    # 2️⃣ clean anonymized version
    for r in results:
        if (
            r.get("type") == "textarea"
            and r.get("from_name") == "shotDescriptionClean"
        ):
            return r["value"]["text"][0]

    return None

def select_shot_type(results):
    for r in results:
        if (
            r.get("type") == "timelinelabels"
            and r.get("from_name") == "shotType"
        ):
            return r["value"]["timelinelabels"][0]

    return None



# if __name__ == "__main__":
#     print("🔍 Analyzing caption length distribution")
#
#     dataset = LSCaptionDataset()
#     tokenizer = dataset.tokenizer
#
#     lengths = []
#
#     for s in dataset.samples:
#         tokens = tokenizer(
#             s["caption"],
#             truncation=False,
#             add_special_tokens=True,
#         )
#         lengths.append(len(tokens["input_ids"]))
#
#     lengths = sorted(lengths)
#
#     print(f"Total samples: {len(lengths)}")
#     print(f"Min length: {min(lengths)}")
#     print(f"Max length: {max(lengths)}")
#
#     def percentile(p):
#         idx = int(len(lengths) * p)
#         return lengths[idx]
#
#     print(f"50% percentile: {percentile(0.5)}")
#     print(f"90% percentile: {percentile(0.9)}")
#     print(f"95% percentile: {percentile(0.95)}")
#     print(f"99% percentile: {percentile(0.99)}")
# if __name__ == "__main__":
#     print("🔍 Debugging LSCaptionDataset")
#
#     dataset = LSCaptionDataset()
#     print(f"📦 Total samples: {len(dataset)}")
#
#     # 打印前几个样本看看
#     NUM_SHOW = min(5, len(dataset))
#
#     for i in range(NUM_SHOW):
#         sample = dataset.samples[i]
#         caption = sample["caption"]
#
#         print("\n==============================")
#         print(f"Sample #{i}")
#         print(f"Video: {os.path.basename(sample['video'])}")
#         print(f"Frame: {sample['frame']}")
#         print(f"Shot type: {sample['shot_type']}")
#         print(f"Caption:\n{caption}")
#
#         # ⭐ 关键检查：是否匿名
#         if "[PLAYER]" not in caption:
#             print("⚠️ WARNING: NOT ANONYMIZED")
#         else:
#             print("✅ OK: anonymized")
