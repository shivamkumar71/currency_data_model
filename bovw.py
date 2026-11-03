import numpy as np
import cv2
import mahotas
import sys
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Setup filepath and parameters
IMDIR=sys.argv[1] if len(sys.argv) > 1 else "data_filtered"
BOVW="model/bovw_codebook_6class.pickle"
DICT_SIZE=600
DATA='model/data_6class.npy'
LABEL='model/label_6class.npy'
SCALER='model/feature_scaler_6class.joblib'

# Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture
def fd_haralick(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick

# Color Histogram
def fd_histogram(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins=8
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# Sift Bag of Visual Words
def feature_extract(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(DICT_SIZE, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_L2).match(descriptors, dictionary)
    histogram = np.bincount(
        [match.trainIdx for match in matches], minlength=DICT_SIZE
    )
    return histogram.astype(np.float32) / max(len(descriptors), 1)

# Directory containing images
base= Path(IMDIR).resolve()
print(base)

# Initialize SIFT and BOW trainer
sift = cv2.SIFT_create()
dictionarySize = DICT_SIZE
descriptors = []

# Generate the vocabulary for BOVW
for file in base.glob('**/*.*'):
    fpath=Path(file).resolve()
    print(str(fpath))
    image = cv2.imread(str(fpath))
    print(fpath.parent.name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, dsc = sift.detectAndCompute(gray, None)
    if dsc is not None:
        descriptors.append(dsc)

# Cluster the visual vocabulary and generate codebook
print("Clustering...")
if not descriptors:
    raise RuntimeError("No SIFT descriptors found in the dataset.")
all_descriptors = np.vstack(descriptors).astype(np.float32)
max_descriptors = 20_000
if len(all_descriptors) > max_descriptors:
    rng = np.random.default_rng(42)
    selected = rng.choice(len(all_descriptors), max_descriptors, replace=False)
    all_descriptors = all_descriptors[selected]
print("Descriptors used for clustering:", len(all_descriptors))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
_, _, dictionary = cv2.kmeans(
    all_descriptors, dictionarySize, None, criteria, 3, cv2.KMEANS_PP_CENTERS
)
print((dictionary.shape))

# Save the codebook with pickle (for later use)
pickle_out = open(BOVW,"wb")
pickle.dump(dictionary, pickle_out)
pickle_out.close()


# Load the bovw codebook (verify)
pickle_in = open(BOVW,"rb")
dictionary = pickle.load(pickle_in)

# Initialize SIFT BOW image descriptor extractor
sift = cv2.SIFT_create()

print("Feature Extraction...")

x_data=[]
x_label=[]
# Combine global features with SIFT-BOVW
for file in base.glob('**/*.*'):
    fpath=Path(file).resolve()
    image = cv2.imread(str(fpath))

    Humo=fd_hu_moments(image)
    Harl=fd_haralick(image)
    Hist=fd_histogram(image)
    Bovw=feature_extract(image)

    print(fpath.parent.name)
    mfeature= np.hstack([Humo, Harl, Hist, Bovw])
    print(mfeature.shape)
    x_data.append(mfeature)
    x_label.append(int(fpath.parent.name))

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
x_data = scaler.fit_transform(x_data)

# Encode labels
encoder  = LabelEncoder()
x_label  = encoder.fit_transform(x_label)

# Save the data, labels, and the fitted scaler. Inference must use the same
# feature ranges as training.
np.save(DATA,np.array(x_data))
np.save(LABEL,np.array(x_label))
import joblib
joblib.dump(scaler, SCALER)

'''
Sample run: python bovw.py data
'''
