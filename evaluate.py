"""
evaluate.py
테스트셋 평가 + TTA 적용 + 이상 스코어 리포트
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from dataset import TTA_TRANSFORMS, CLASS_NAMES, NUM_CLASSES, get_dataloaders
from model import build_model
from gradcam import batch_visualize


@torch.no_grad()
def tta_predict(model, img_pil, device):
    all_probs = []
    for tfm in TTA_TRANSFORMS:
        tensor = tfm(img_pil).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        all_probs.append(probs)
    mean_probs = np.mean(all_probs, axis=0)
    return mean_probs, 1.0 - mean_probs[0]


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"모델 로드: epoch {ckpt['epoch']}")

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=1, num_workers=0)

    all_labels, all_preds, all_scores = [], [], []
    per_image_results = []

    print("\nTTA 추론 중...")
    for i, (_, label, img_path) in enumerate(test_loader.dataset):
        img_pil = Image.open(img_path).convert("RGB")
        mean_probs, anomaly_score = tta_predict(model, img_pil, device)
        pred_class = int(np.argmax(mean_probs))

        all_labels.append(int(label))
        all_preds.append(pred_class)
        all_scores.append(anomaly_score)
        per_image_results.append({
            "image": img_path,
            "true_class": CLASS_NAMES[int(label)],
            "pred_class": CLASS_NAMES[pred_class],
            "correct": pred_class == int(label),
            "anomaly_score": round(float(anomaly_score), 4),
            "probs": {CLASS_NAMES[j]: round(float(mean_probs[j]), 4) for j in range(NUM_CLASSES)},
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1} / {len(test_loader.dataset)}")

    print("\n" + "=" * 60)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("혼동 행렬")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:<12}", row)

    binary_labels = [0 if l == 0 else 1 for l in all_labels]
    try:
        auc = roc_auc_score(binary_labels, all_scores)
        print(f"\nROC-AUC: {auc:.4f}")
    except Exception:
        pass

    scores_arr = np.array(all_scores)
    print(f"\n이상 스코어  평균={scores_arr.mean():.4f}  std={scores_arr.std():.4f}")

    if args.gradcam:
        wrong = [r["image"] for r in per_image_results if not r["correct"]]
        print(f"\nGrad-CAM 생성: 오분류 {len(wrong)}장")
        batch_visualize(model, wrong, str(Path(args.save_dir) / "gradcam_errors"), device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {save_dir / 'eval_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--save_dir", default="../runs/eval")
    parser.add_argument("--gradcam", action="store_true")
    evaluate(parser.parse_args())
