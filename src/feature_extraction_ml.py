import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from skimage.measure import shannon_entropy
from skimage.feature import local_binary_pattern, hog

def extract_stat_features(image):
    """
    Statistical feature extraction on ROIs.
    Returns a 15-dimensional vector.
    """
    pixels = image[image > 0]

    # Return array of zeros to prevent matrix stacking errors later
    if len(pixels) == 0:
        return np.zeros(15)

    # Calcul des gradients
    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_pixels = grad_mag[image > 0]

    features = np.array([
        pixels.mean(),
        pixels.std(),
        pixels.min(),
        pixels.max(),
        np.median(pixels),
        np.percentile(pixels, 5),
        np.percentile(pixels, 25),
        np.percentile(pixels, 75),
        np.percentile(pixels, 95),
        skew(pixels),
        kurtosis(pixels),
        shannon_entropy(pixels),
        cv2.Laplacian(image.astype(np.float32), cv2.CV_32F)[image > 0].var(),
        grad_pixels.mean(),
        grad_pixels.std()
    ])

    return features


def extract_lbp_features(image, radius=3, method="uniform"):
    """
    Local Binary Pattern feature extraction on ROIs.
    Returns a (8*radius + 2)-dimensional vector.
    """
    n_points = 8 * radius

    lbp = local_binary_pattern(
        image,
        P=n_points,
        R=radius,
        method=method
    )

    mask = image > 0
    lbp_pixels = lbp[mask]
    
    n_bins = n_points + 2  # pour method='uniform'

    # Safe return for empty masks
    if len(lbp_pixels) == 0:
        return np.zeros(n_bins)

    hist, _ = np.histogram(
        lbp_pixels,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )
    
    return hist


def extract_hog_features(image, 
                         hog_image_size=(160, 160),
                         orientations=9,
                         pixels_per_cell=(32, 32),
                         cells_per_block=(2, 2),
                         block_norm='L2-Hys'):
    """
    HOG features extraction. 
    (Masking handled naturally since black background yields 0 gradients)
    """
    if image.shape[:2] != hog_image_size:
        image = cv2.resize(image, hog_image_size, interpolation=cv2.INTER_LINEAR)

    features = hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=False,
        feature_vector=True
    )
    
    return features


from data_loader import DatasetLoader 

def process_and_save_features(X_images, y_labels, split_name, output_dir):
    """
    Extracts features independently and saves them as separate .npy files.
    split_name: 'train' or 'test'
    """
    print(f"\n--- Extracting features for {split_name.upper()} set ({len(X_images)} images) ---")
    
    # Initialize empty lists for each feature type
    all_stats, all_lbp, all_hog = [], [], []
    
    for img in tqdm(X_images, desc=f"Processing {split_name} images"):
        # Extract and append
        all_stats.append(extract_stat_features(img))
        all_lbp.append(extract_lbp_features(img))
        all_hog.append(extract_hog_features(img))
        
    # Convert lists to NumPy matrices
    all_stats = np.array(all_stats)
    all_lbp = np.array(all_lbp)
    all_hog = np.array(all_hog)
    
    # Save independently as raw binary (.npy)
    np.save(output_dir / f"X_{split_name}_stats.npy", all_stats)
    np.save(output_dir / f"X_{split_name}_lbp.npy", all_lbp)
    np.save(output_dir / f"X_{split_name}_hog.npy", all_hog)
    np.save(output_dir / f"y_{split_name}.npy", y_labels)
    
    print(f"Saved {split_name} features to {output_dir}/")

def main():
    # 1. Setup
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    FEATURES_DIR = PROJECT_ROOT / "data" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Load the dataset (Memory efficient)
    loader = DatasetLoader(data_dir=DATA_DIR)
    X_train, X_test, y_train, y_test, label_encoder = loader.get_train_test_split()
    
    # 3. Save label encoder classes to know what 0,1,2,3 map to later
    np.save(FEATURES_DIR / "classes.npy", label_encoder.classes_)
    
    # 4. Extract and Save
    process_and_save_features(X_train, y_train, split_name="train", output_dir=FEATURES_DIR)
    process_and_save_features(X_test, y_test, split_name="test", output_dir=FEATURES_DIR)
    
    print("\nAll feature extraction completed successfully!")

if __name__ == "__main__":
    main()