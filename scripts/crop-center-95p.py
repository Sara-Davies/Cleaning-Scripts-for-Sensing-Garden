import os
import numpy as np
from PIL import Image
import shutil
from scipy.spatial.distance import euclidean

def compute_center_rgb(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        crop_box = (w * 0.3, h * 0.3, w * 0.7, h * 0.7)
        center_crop = img.crop(crop_box)
        pixels = np.array(center_crop).reshape(-1, 3)
        if pixels.size == 0:
            print(f"No pixels found in crop for {img_path}")
            return None
        return np.mean(pixels, axis=0)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None

def filter_by_species_outliers(train_dir, output_dir, keep_percent):
    kept_dir = os.path.join(output_dir, "kept")
    removed_dir = os.path.join(output_dir, "removed")
    os.makedirs(kept_dir, exist_ok=True)
    os.makedirs(removed_dir, exist_ok=True)

    total_tp = total_fp = total_fn = total_tn = 0
    species_results = {}

    print("Processing each species separately...")

    for species in os.listdir(train_dir):
        species_path = os.path.join(train_dir, species)
        if not os.path.isdir(species_path):
            continue

        all_images = []
        for quality_label in ["good", "bad"]:
            q_path = os.path.join(species_path, quality_label)
            if not os.path.isdir(q_path):
                continue
            for img_name in os.listdir(q_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_path = os.path.join(q_path, img_name)
                color = compute_center_rgb(img_path)
                if color is not None:
                    all_images.append((color, img_name, quality_label, q_path))

        if len(all_images) < 5:
            print(f"  {species}: Only {len(all_images)} images, skipping")
            continue

        print(f"  {species}: {len(all_images)} images")

        vectors = np.array([x[0] for x in all_images])
        species_median = np.median(vectors, axis=0)
        distances = [euclidean(x[0], species_median) for x in all_images]
        threshold = np.percentile(distances, keep_percent)

        tp = fp = fn = tn = 0  # true positives, etc.

        for i, (color, img_name, label, src_folder) in enumerate(all_images):
            src = os.path.join(src_folder, img_name)
            dist = distances[i]

            is_removed = dist > threshold
            dst_folder = os.path.join(removed_dir if is_removed else kept_dir, species.replace(" ", "_"))
            os.makedirs(dst_folder, exist_ok=True)
            shutil.copy(src, os.path.join(dst_folder, f"{dist:.2f}_" + img_name))

            # Metrics calculation
            # Correctly removed a bad image
            if label == "bad" and is_removed:
                tp += 1
            # Missed bad image
            elif label == "bad" and not is_removed:
                fn += 1
            # Wrongly removed a good one
            elif label == "good" and is_removed:
                fp += 1
            # Correctly kept
            elif label == "good" and not is_removed:
                tn += 1

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

        species_results[species] = {
            'total': len(all_images),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        print(f"    Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}")

    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (total_tp + total_fn + 1e-9)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-9)
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-9)

    print("\n=== Overall Metrics ===")
    print(f"Precision: {overall_precision:.2f}")
    print(f"Recall: {overall_recall:.2f}")
    print(f"F1 Score: {overall_f1:.2f}")
    print(f"Accuracy: {overall_accuracy:.2f}")

filter_by_species_outliers(
    train_dir="/Users/scl/PycharmProjects/Filter-Metrics/data/train-gb copy",
    output_dir="/Users/scl/PycharmProjects/Filter-Metrics/data/cleaned-95",
    keep_percent=95
)
