import os
import numpy as np
import cv2
from PIL import Image
import shutil
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
import hashlib
from collections import defaultdict


class InsectDatasetCleaner:
    def __init__(self):
        self.duplicate_hashes = set()
        self.image_features = []

    def compute_center_rgb(self, img_path):
        """Original color-based filtering method"""
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

    def check_brightness_exposure(self, img_path, dark_threshold=30, bright_threshold=220):
        """Filter too dark or overexposed images"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return False, "corrupted"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)

            if mean_brightness < dark_threshold:
                return False, "too_dark"
            elif mean_brightness > bright_threshold:
                return False, "overexposed"

            return True, "good_exposure"
        except Exception as e:
            print(f"Error checking brightness for {img_path}: {e}")
            return False, "error"

    def check_contrast(self, img_path, min_contrast=10):
        """Filter low contrast images"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "corrupted"

            contrast = np.std(img)

            if contrast < min_contrast:
                return False, f"low_contrast_{contrast:.1f}"

            return True, f"good_contrast_{contrast:.1f}"
        except Exception as e:
            print(f"Error checking contrast for {img_path}: {e}")
            return False, "error"

    def check_blur(self, img_path, blur_threshold=50):
        """Detect blurry images using Laplacian variance"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, "corrupted"

            laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

            if laplacian_var < blur_threshold:
                return False, f"blurry_{laplacian_var:.1f}"

            return True, f"sharp_{laplacian_var:.1f}"
        except Exception as e:
            print(f"Error checking blur for {img_path}: {e}")
            return False, "error"

    def compute_exact_hash(self, img_path):
        """Compute exact duplicate hash"""
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            print(f"Error computing hash for {img_path}: {e}")
            return None

    def compute_perceptual_hash(self, img_path):
        """Compute perceptual hash for near-duplicate detection"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Resize to 8x8 for pHash
            resized = cv2.resize(img, (8, 8))

            # Compute DCT
            dct = cv2.dct(np.float32(resized))

            # Take top-left 8x8 and flatten
            dct_low = dct[:8, :8].flatten()

            # Compute median
            median = np.median(dct_low)

            # Create hash
            hash_bits = dct_low > median
            return ''.join(['1' if b else '0' for b in hash_bits])
        except Exception as e:
            print(f"Error computing perceptual hash for {img_path}: {e}")
            return None

    def hamming_distance(self, hash1, hash2):
        """Calculate Hamming distance between two hashes"""
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def check_duplicates(self, img_path, similarity_threshold=5):
        """Check for exact and near duplicates"""
        # Check exact duplicates
        exact_hash = self.compute_exact_hash(img_path)
        if exact_hash in self.duplicate_hashes:
            return False, "exact_duplicate"

        # Check near duplicates
        perceptual_hash = self.compute_perceptual_hash(img_path)
        if perceptual_hash:
            for stored_hash, stored_path in self.image_features:
                if self.hamming_distance(perceptual_hash, stored_hash) <= similarity_threshold:
                    return False, "near_duplicate"

            # Store this image's features
            self.image_features.append((perceptual_hash, img_path))

        if exact_hash:
            self.duplicate_hashes.add(exact_hash)

        return True, "unique"

    def apply_all_filters(self, img_path):
        """Apply all filters to an image"""
        filters_results = {}

        # Brightness/exposure check
        pass_brightness, brightness_reason = self.check_brightness_exposure(img_path)
        filters_results['brightness'] = (pass_brightness, brightness_reason)

        # Contrast check
        pass_contrast, contrast_reason = self.check_contrast(img_path)
        filters_results['contrast'] = (pass_contrast, contrast_reason)

        # Blur check
        pass_blur, blur_reason = self.check_blur(img_path)
        filters_results['blur'] = (pass_blur, blur_reason)


        # Duplicate check
        pass_duplicates, duplicate_reason = self.check_duplicates(img_path)
        filters_results['duplicates'] = (pass_duplicates, duplicate_reason)

        # Overall pass/fail
        overall_pass = all(result[0] for result in filters_results.values())

        return overall_pass, filters_results

    def filter_dataset_comprehensive(self, train_dir, output_dir, keep_percent=85):
        """Enhanced filtering with OpenCV filters + original color filtering"""
        kept_dir = os.path.join(output_dir, "kept")
        removed_dir = os.path.join(output_dir, "removed")
        os.makedirs(kept_dir, exist_ok=True)
        os.makedirs(removed_dir, exist_ok=True)

        # Create subdirectories for different removal reasons
        removal_reasons = ["brightness", "contrast", "blur", "duplicates", "color_outlier"]
        for reason in removal_reasons:
            os.makedirs(os.path.join(removed_dir, reason), exist_ok=True)

        total_tp = total_fp = total_fn = total_tn = 0
        species_results = {}
        filter_stats = defaultdict(int)

        print("Processing each species with comprehensive filtering...")

        for species in os.listdir(train_dir):
            species_path = os.path.join(train_dir, species)
            if not os.path.isdir(species_path):
                continue

            all_images = []
            opencv_passed_images = []

            # First pass: OpenCV filters
            for quality_label in ["good", "bad"]:
                q_path = os.path.join(species_path, quality_label)
                if not os.path.isdir(q_path):
                    continue

                for img_name in os.listdir(q_path):
                    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue

                    img_path = os.path.join(q_path, img_name)

                    # Apply OpenCV filters
                    passes_opencv, filter_results = self.apply_all_filters(img_path)

                    if not passes_opencv:
                        # Find the first failing filter for categorization
                        failing_filter = next(filter_name for filter_name, (passed, reason)
                                              in filter_results.items() if not passed)

                        dst_folder = os.path.join(removed_dir, failing_filter, species.replace(" ", "_"))
                        os.makedirs(dst_folder, exist_ok=True)

                        reason = filter_results[failing_filter][1]
                        shutil.copy(img_path, os.path.join(dst_folder, f"{reason}_{img_name}"))
                        filter_stats[failing_filter] += 1

                        # Update metrics
                        if quality_label == "bad":
                            total_tp += 1  # Correctly removed bad image
                        else:
                            total_fp += 1  # Wrongly removed good image
                    else:
                        # Compute color for second pass
                        color = self.compute_center_rgb(img_path)
                        if color is not None:
                            opencv_passed_images.append((color, img_name, quality_label, q_path))
                            all_images.append((color, img_name, quality_label, q_path))

            if len(opencv_passed_images) < 5:
                print(
                    f"  {species}: Only {len(opencv_passed_images)} images passed OpenCV filters, skipping color filtering")
                continue

            print(f"  {species}: {len(all_images)} total images, {len(opencv_passed_images)} passed OpenCV filters")

            # Second pass: Color-based outlier detection (your original method)
            vectors = np.array([x[0] for x in opencv_passed_images])
            species_median = np.median(vectors, axis=0)
            distances = [euclidean(x[0], species_median) for x in opencv_passed_images]
            threshold = np.percentile(distances, keep_percent)

            tp = fp = fn = tn = 0

            for i, (color, img_name, label, src_folder) in enumerate(opencv_passed_images):
                src = os.path.join(src_folder, img_name)
                dist = distances[i]

                is_color_outlier = dist > threshold

                if is_color_outlier:
                    dst_folder = os.path.join(removed_dir, "color_outlier", species.replace(" ", "_"))
                    os.makedirs(dst_folder, exist_ok=True)
                    shutil.copy(src, os.path.join(dst_folder, f"{dist:.2f}_{img_name}"))
                    filter_stats["color_outlier"] += 1

                    if label == "bad":
                        tp += 1
                    else:
                        fp += 1
                else:
                    # Keep the image
                    dst_folder = os.path.join(kept_dir, species.replace(" ", "_"))
                    os.makedirs(dst_folder, exist_ok=True)
                    shutil.copy(src, os.path.join(dst_folder, f"{dist:.2f}_{img_name}"))

                    if label == "bad":
                        fn += 1
                    else:
                        tn += 1

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            # Calculate metrics for this species
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-9)

            species_results[species] = {
                'total': len(all_images),
                'opencv_passed': len(opencv_passed_images),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
            }

            print(f"    OpenCV passed: {len(opencv_passed_images)}/{len(all_images)}")
            print(f"    Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}")

        # Print overall results
        overall_precision = total_tp / (total_tp + total_fp + 1e-9)
        overall_recall = total_tp / (total_tp + total_fn + 1e-9)
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-9)
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-9)

        print("\n=== Overall Metrics ===")
        print(f"Precision: {overall_precision:.2f}")
        print(f"Recall: {overall_recall:.2f}")
        print(f"F1 Score: {overall_f1:.2f}")
        print(f"Accuracy: {overall_accuracy:.2f}")

        print("\n=== Filter Statistics ===")
        for filter_name, count in filter_stats.items():
            print(f"{filter_name}: {count} images removed")


# Usage with more lenient settings
cleaner = InsectDatasetCleaner()


# You can also create a test run to see the actual values in your images:
def analyze_sample_images(train_dir, num_samples=10):
    """Analyze a few images to see typical values for calibrating thresholds"""
    print("=== ANALYZING SAMPLE IMAGES FOR CALIBRATION ===")
    count = 0

    for species in os.listdir(train_dir):
        if count >= num_samples:
            break
        species_path = os.path.join(train_dir, species)
        if not os.path.isdir(species_path):
            continue

        for quality in ["good", "bad"]:
            quality_path = os.path.join(species_path, quality)
            if not os.path.isdir(quality_path):
                continue

            for img_name in os.listdir(quality_path):
                if count >= num_samples:
                    break
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(quality_path, img_name)

                try:
                    # Analyze this image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    brightness = np.mean(img)
                    contrast = np.std(img)
                    blur = cv2.Laplacian(img, cv2.CV_64F).var()

                    print(f"{species}/{quality}/{img_name}:")
                    print(f"  Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
                    print(f"  Blur: {blur:.1f}")
                    print()

                    count += 1
                except Exception as e:
                    print(f"Error analyzing {img_path}: {e}")


# Run analysis first to see your data characteristics
# analyze_sample_images("/Users/scl/PycharmProjects/Filter-Metrics/.venv/data/train-gb copy")

# Then run the cleaner with appropriate settings
cleaner.filter_dataset_comprehensive(
    train_dir="/Users/scl/PycharmProjects/Filter-Metrics/.venv/data/train-gb copy",
    output_dir="/Users/scl/PycharmProjects/Filter-Metrics/.venv/data/cleaned-comprehensive",
    keep_percent=85
)