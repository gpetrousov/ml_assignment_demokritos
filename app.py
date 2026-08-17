import gradio as gr
import numpy as np
import joblib
from PIL import Image
from skimage.transform import resize
from huggingface_hub import hf_hub_download
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2hsv
import numpy as np


# 0. Inject feature extraction functions

def extract_hog_features_from_list(X_images_rgb):
    """
    Takes a list of pre-resized RGB images and returns HOG features.
    
    INPUTS:
        - X_images_rgb: list of images in memory
    
    OUTPUTS:
        - np.array(X_hog_features): Numpy array of HOG features
    """
    
    X_hog_features = []
    
    for img_rgb in X_images_rgb:
        # 1. Convert RGB to Gray using skimage
        img_gray = rgb2gray(img_rgb)

        # 2. Extract HOG
        hog_features = hog(
            img_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        X_hog_features.append(hog_features)
        
    return np.array(X_hog_features)


def extract_lbp_features_from_list(X_images_rgb):
    """
    Standardized LBP extraction using pre-resized RGB images.
    
    INPUTS
        - X_images_rgb: list of images in memory
        
    OUTPUTS:
        - np.array(X_lbp_features): numpy array of LBP features for each image
    """
    
    X_lbp_features = []
    
    for img_rgb in X_images_rgb:
        # 1. Convert RGB to Gray
        img_gray = rgb2gray(img_rgb)

        # 2. Extract LBP
        lbp = local_binary_pattern(img_gray.astype(dtype="int32"), P=8, R=1, method='uniform')
        
        # 3. Create Histogram (10 bins for P=8 uniform)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

        X_lbp_features.append(lbp_hist)
        
    return np.array(X_lbp_features)


def extract_hsv_features_from_list(X_images_rgb, hue_lower_bound=0.2, hue_upper_bound=0.45, sat_lower=0.25, bins=10):
    """
    Extracts masked HSV histograms from a list of RGB images.
    Uses color-based thresholding to isolate plant matter from the background.
    
    INPUTS:
        - X_images_rgb: List of RGB images in memory
        - hue_lower_bound: Lower threshold for green hue (default 0.2)
        - hue_upper_bound: Upper threshold for green hue (default 0.45)
        - sat_lower: Minimum saturation to filter out gray/background (default 0.25)
        - bins: Number of histogram bins per channel
        
    OUTPUTS:
        - np.array(X_hsv_features): Numpy array of concatenated H-S-V histograms
    """
    
    X_hsv_features = []
    
    for img_rgb in X_images_rgb:
        
        # 1. Extract channels
        hsv_img = rgb2hsv(img_rgb)
        hue_chan = hsv_img[:, :, 0]
        sat_chan = hsv_img[:, :, 1]
        val_chan = hsv_img[:, :, 2]

        # 2. Build mask
        hsv_mask = (hue_chan > hue_lower_bound) & (hue_chan < hue_upper_bound) & (sat_chan > sat_lower)
        
        # 3. Apply mask
        plant_hue = hue_chan[hsv_mask]
        plant_sat = sat_chan[hsv_mask]
        plant_val = val_chan[hsv_mask]

        # 4. Create Histogram for each channel
        if plant_hue.size > 0:
            hue_hist, _ = np.histogram(plant_hue, bins=bins, range=(0, 1), density=True)
            sat_hist, _ = np.histogram(plant_sat, bins=bins, range=(0, 1), density=True)
            val_hist, _ = np.histogram(plant_val, bins=bins, range=(0, 1), density=True)
        else:
#             print("Zeroed image")
            hue_hist = np.zeros(bins)
            sat_hist = np.zeros(bins)
            val_hist = np.zeros(bins)

        # 5. Combine channels into one vector
        hsv_vector = np.concatenate([hue_hist, sat_hist, val_hist]) # Flat
        
        # 6. Append to feature list
        X_hsv_features.append(hsv_vector)
        
    return np.array(X_hsv_features)


def feature_fusion(super_matrix:np.ndarray=None, feature_list:list =[]):
    """
    Fuses multiple feature sets into a single matrix via horizontal stacking.
    Used to combine HOG, LBP, and HSV features into a unified feature vector.
    
    INPUTS:
        - super_matrix: Existing numpy matrix of features (None for first fusion)
        - feature_list: List of feature arrays to be appended to the super_matrix
        
    OUTPUTS:
        - super_matrix: Numpy matrix with features stacked horizontally per image
    """
    
    new_features = [np.array(f) for f in feature_list]
    
    if super_matrix is None:
        super_matrix = np.hstack([ np.array(f) for f in feature_list ])
        
    else:
        super_matrix = np.hstack([super_matrix] + new_features)
        
    return super_matrix


# 1. Load artifacts
REPO_ID = "ipetrousov/weedcrop_svm_classifier"

model_file = hf_hub_download(repo_id=REPO_ID, filename="svc_opt.joblib")
scaler_file = hf_hub_download(repo_id=REPO_ID, filename="scaler.joblib")
pca_file = hf_hub_download(repo_id=REPO_ID, filename="pca.joblib")

model = joblib.load(model_file)
scaler = joblib.load(scaler_file)
pca = joblib.load(pca_file)

# 2. Optimized threshold
OPT_THRESHOLD = 0.023699424100647747

def classify_plant(image_input):
    if image_input is None:
        return "No image provided", "0.00%"

    # Convert to NumPy array and resize to target dimension (256, 256)
    img_np = np.array(image_input)
    img_resized = resize(img_np, (256, 256), anti_aliasing=True)
    img_list = [img_resized]

    # Feature extraction
    img_hog = extract_hog_features_from_list(img_list)
    img_lbp = extract_lbp_features_from_list(img_list)
    img_hsv = extract_hsv_features_from_list(img_list)

    # Dimensionality reduction on HOG features
    img_hog_scaled = scaler.transform(img_hog)
    img_hog_reduced = pca.transform(img_hog_scaled)

    # Feature fusion
    X_combined = feature_fusion(
        feature_list=[img_lbp, img_hog_reduced, img_hsv]
    )

    # Probabilistic inference
    probs = model.predict_proba(X_combined)
    crop_prob = probs[0, 0]

    # Apply threshold logic
    if crop_prob >= OPT_THRESHOLD:
        prediction = "CROP"
        confidence = crop_prob
    else:
        prediction = "WEED"
        confidence = probs[0, 1]

    confidence_str = f"{confidence:.4%}"
    detailed_metrics = (
        f"Decision: {prediction}\n"
        f"Confidence: {confidence_str}\n"
        f"Raw Crop Probability: {crop_prob:.6f}\n"
        f"Operating Threshold: {OPT_THRESHOLD:.6f}"
    )

    return prediction, detailed_metrics

citation_markdown = """
---
### 📖 Citation & Reference

If you use this work, model, or code in your research, please cite it as:

```bibtex
@software{petrousov2026autofarm,
  author = {Petrousov, Ioannis},
  title = {Autofarming: Surgical Weed Control via Multi-modal Feature Fusion},
  month = {February},
  year = {2026},
  institution = {NCSR Demokritos & University of Piraeus},
  url = {[https://github.com/gpetrousov/ml_assignment_demokritos](https://github.com/gpetrousov/ml_assignment_demokritos)},
  note = {MSc in Artificial Intelligence - Machine Learning}
}
"""

# 3. Gradio Interface Construction
demo = gr.Interface(
    fn=classify_plant,
    inputs=gr.Image(type="pil", label="Upload Seedling Image"),
    outputs=[
        gr.Textbox(label="Predicted Class"),
        gr.Textbox(label="Evaluation Breakdown")
    ],
    title="Autofarm: Surgical Weed & Crop Classifier",
    description="Drop a plant or weed image to evaluate it using an optimized SVM classifier with multi-modal feature fusion (HOG + LBP + HSV).",
    examples=[],
    article=citation_markdown
)

if __name__ == "__main__":
    demo.launch()