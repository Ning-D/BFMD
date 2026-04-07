from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice


def compute_caption_metrics(preds, gts):
    """
    preds: List[str]
    gts:   List[List[str]]
    """

    # COCO-style dict
    res = {i: [p] for i, p in enumerate(preds)}
    ref = {i: gts[i] for i in range(len(preds))}

    scores = {}

    scorers = [
        ("BLEU", Bleu(4)),
        ("METEOR", Meteor()),
        ("CIDEr", Cider()),
        ("ROUGE_L", Rouge()),
    
    ]

    for name, scorer in scorers:
        score, _ = scorer.compute_score(ref, res)
        if isinstance(score, list):
            for i, s in enumerate(score):
                scores[f"{name}_{i+1}"] = s
        else:
            scores[name] = score

    return scores
