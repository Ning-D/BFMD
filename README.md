# BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning

[![Conference](https://img.shields.io/badge/CVSports-2026-blue)]()
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)]()

Official repository for **BFMD: A Full-Match Badminton Dense Dataset for Dense Shot Captioning**.

## Overview

Understanding tactical dynamics in badminton requires analyzing entire matches rather than isolated clips. However, existing badminton datasets mainly focus on short clips or task-specific annotations and rarely provide full-match data with dense multimodal annotations. This limitation makes it difficult to generate accurate shot captions and perform match-level analysis.

To address this limitation, we introduce **BFMD**, the first full-match badminton dense dataset, containing:

- **19 broadcast matches**
- **20+ hours** of play
- **1,687 rallies**
- **16,751 hit events**
- both **singles and doubles**
- a shot caption for each hit event

BFMD provides hierarchical annotations including:

- match segments
- rally events
- shot types
- shuttle trajectories
- player pose keypoints
- shot captions

In addition, we develop a **VideoMAE-based multimodal captioning framework with Semantic Feedback**, which leverages shot semantics to guide caption generation and improve semantic consistency.

Experimental results show that multimodal modeling and semantic feedback improve shot caption quality over RGB-only baselines. We also demonstrate the potential of BFMD for analyzing the temporal evolution of tactical patterns across full matches.

---

## Dataset Statistics

| Item | Value |
|------|-------|
| Matches | 19 |
| Total duration | 20+ hours |
| Rallies | 1,687 |
| Hit events | 16,751 |
| Match types | Singles and Doubles |

---

## Features

- Full-match badminton dataset with dense annotations
- Hierarchical annotation structure from match level to shot level
- Multimodal annotations for each event
- Dense shot captioning benchmark
- Baseline multimodal captioning framework
- Tactical pattern analysis across full matches

---

## Annotation Contents

BFMD includes the following annotations:

- **Match-level segments**
  - rally boundaries
  - replay intervals
  - game-related segments

- **Rally-level annotations**
  - rally start and end
  - rally structure

- **Shot-level annotations**
  - hit frame
  - shot type
  - shuttle trajectory
  - player pose keypoints
  - shot caption

---

## Method

We provide a **VideoMAE-based multimodal captioning framework** with a **Semantic Feedback** mechanism.

### Main components

- **Visual encoder** based on VideoMAE
- **Multimodal fusion** of visual and structured cues
- **Transformer-based caption decoder**
- **Semantic Feedback module** for improving semantic consistency

This framework serves as a baseline for dense shot captioning on BFMD.

---

## Repository Structure

You can adjust this part based on your actual project files.

```bash
BFMD/
├── data/                 # dataset files or links / metadata
├── annotations/          # annotation files
├── checkpoints/          # trained model checkpoints
├── scripts/              # preprocessing / training / evaluation scripts
├── src/                  # model source code
├── README.md
├── requirements.txt
└── environment.yml