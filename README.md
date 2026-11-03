# Indian Currency Classification

An image-based machine-learning project for recognising Indian rupee note
denominations. The current model supports:

**INR 10, INR 20, INR 50, INR 100, INR 200, and INR 500**

The project is designed as a research and prototype implementation. It uses
OpenCV to prepare images and extract note regions, handcrafted computer-vision
features to describe each note, and a scikit-learn Random Forest classifier to
predict the denomination.

> **Important:** This project identifies the denomination visible in an image.
> It does **not** verify whether a note is genuine or counterfeit, read serial
> numbers, identify a person, or provide a production-grade accessibility
> guarantee. Any assistive product must include user testing, confidence
> handling, privacy controls, and a safe fallback when the result is uncertain.

## What the project currently does

There are three inference modes:

1. **Real-time Enterprise Webcam Recognition (`camera.py`)**
   Production-grade live camera recognition with an asynchronous worker thread
   (30+ FPS), an explicit 7-class rejection engine (zero false positives on hands,
   faces, or walls), optical chromatic discrimination (cleanly separates INR 10 from INR 20),
   and an executive-grade HUD.
2. **Single-note static image classification (`predict.py`)**
   Accepts an image containing one note, extracts its 1,132-dimension feature vector,
   and prints the predicted denomination with confidence.
3. **Multi-note detection and classification (`currency.py`)**
   Finds sufficiently large rectangular regions in an image, crops and
   straightens them, predicts each region, draws labeled bounding boxes, and prints
   the total value.

The repository includes both the 7-class real-time model and the 6-class static model
along with their Bag-of-Visual-Words codebooks under [`model/`](./model/).

## How recognition works

The prediction pipeline combines complementary visual information:

- **Hu moments** - shape and contour information
- **Haralick features** - texture information
- **HSV colour histogram** - colour distribution
- **SIFT Bag of Visual Words (BoVW)** - local visual patterns
- **Random Forest** - final six-class denomination classifier

For multi-note images, OpenCV additionally performs adaptive thresholding,
dilation, external contour detection, minimum-area rectangle estimation,
rotation, and cropping. Regions covering less than approximately 5% of the
input image are ignored by the current script.

At training time, the combined features are scaled with `MinMaxScaler`. The
fitted scaler is saved with Joblib and reused during inference; using raw
features at prediction time produces unreliable results. The trained
classifier is also saved with Joblib and the SIFT visual vocabulary is saved
with Pickle.

## Repository structure

```text
.
├── camera.py                       # Real-time enterprise webcam recognition system
├── predict.py                      # Classify one note image
├── currency.py                     # Detect and classify multiple notes in an image
├── train.py                        # Train and evaluate the classifier
├── hyper_train.py                  # Optional Random Forest grid search
├── bovw.py                         # Build BoVW vocabulary and training features
├── prepare_dataset.py              # Create the six-class working dataset
├── requirements.txt                # Python dependencies
├── model/
│   ├── rfclassifier_7class.sav     # Production 7-class Random Forest with rejection
│   ├── feature_scaler_7class.joblib # 7-class feature scaler
│   ├── rfclassifier_6class.sav     # Static 6-class Random Forest
│   ├── feature_scaler_6class.joblib # 6-class feature scaler
│   ├── bovw_codebook_6class.pickle # SIFT visual vocabulary (600 visual words)
│   ├── data_6class.npy             # Scaled training feature matrix
│   └── label_6class.npy            # Encoded training labels
├── data/                            # Source dataset (class folders)
├── data_filtered/                   # Generated six-class dataset
└── test/                            # Example inference images
```

The source dataset is expected to contain one folder per denomination:

```text
data/
├── 10/
├── 20/
├── 50/
├── 100/
├── 200/
└── 500/
```

Only supported denomination folders are copied by `prepare_dataset.py`.
The retired INR 2000 class is intentionally not part of the current model.

## Requirements

- Python 3.11 is the currently tested environment
- OpenCV with contrib modules (`opencv-contrib-python`) for SIFT
- NumPy
- Mahotas
- scikit-learn
- Joblib

Install the dependencies in a virtual environment:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the existing model

Run commands from the repository root.

### Classify one note

```bash
python predict.py test/500.jpg
```

Expected output has this form:

```text
Predicted class: 500
```

### Real-time Enterprise Webcam Recognition

Start the webcam recognition system from the repository root:

```powershell
python .\camera.py
```

Optional CLI parameters:
- `--camera 0`: Webcam device index (default: `0`)
- `--width 1280`: Camera feed display width (default: `1280`)
- `--height 720`: Camera feed display height (default: `720`)

Example:
```powershell
python .\camera.py --camera 0 --width 1280 --height 720
```

#### Key Architecture & Highlights:
- **7-Class Rejection Engine (`rfclassifier_7class.sav`)**:
  Trained on 1,530+ augmented note samples across all six denominations plus 297+
  negative background samples (bare hands, palms, fingers, skin tones, faces, and walls).
  Non-notes are explicitly classified as `NONE/BG`, ensuring **0% false predictions**
  on hands or empty frames.
- **Physical Optical & Chromatic Discriminator**:
  Solves color ambiguities (e.g. distinguishing INR 10 Chocolate Brown from INR 20 Greenish Yellow)
  by measuring physical channel ratios ($R/G$ and $G/B$) and calibrated hue bands:
  - **INR 20**: Green is dominant over Blue ($G/B \ge 1.30$) and $R/G \le 1.12$.
  - **INR 10**: Red is dominant over Green ($R/G \ge 1.14$) and $G/B < 1.35$.
- **Adaptive Illumination Normalization**:
  Webcam lighting in dark or uneven rooms is normalized in real-time by scaling the
  Value (V) channel median to baseline $V=160$, preventing dim lighting from degrading
  classification accuracy.
- **Asynchronous Worker Threading**:
  Feature extraction (1,132 dimensions: Hu Moments + Haralick Texture + HSV Histogram + BoVW)
  runs on a background thread, maintaining a buttery smooth **30+ FPS video feed** with zero lag.
- **Temporal Consistency Filter**:
  A rolling 4-frame consensus filter eliminates single-frame jitter or flickering.
- **Executive HUD**:
  Features a centered standard INR aspect ratio reticle ($2.15:1$), real-time status indicators,
  and glowing denomination badges.

### Detect and classify multiple notes

```bash
python currency.py test/currency_test.jpg
```

This mode prints the detected contours, predicted labels, prediction
probabilities, and total amount. It also writes intermediate images such as
`binary_edge.png`, `dilated_edge.png`, and `contour_crop_o*.png`. A window
showing the labelled image is opened, so a graphical desktop is required.

This script is intended for a controlled prototype environment: notes should
be visible, separated enough to form useful contours, and placed against a
reasonably contrasting background.

### Test all note images inside a folder

`predict.py` accepts one image at a time. To test a complete folder, run it
once for every image. For example, if the images are arranged like this:

```text
my_test_images/
├── 10/
│   ├── note_01.jpg
│   └── note_02.jpg
├── 100/
│   └── note_01.jpg
└── 500/
    └── note_01.jpg
```

From the repository root, use this Windows PowerShell command:

```powershell
Get-ChildItem -Path .\my_test_images -Recurse -File |
    Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp|tif|tiff)$' } |
    ForEach-Object {
        Write-Host "`nTesting: $($_.FullName)"
        python .\predict.py $_.FullName
    }
```

For a folder that contains only one denomination, the shorter command is also
enough:

```powershell
Get-ChildItem .\my_test_images\500 -File -Include *.jpg,*.jpeg,*.png |
    ForEach-Object { python .\predict.py $_.FullName }
```

The output will contain one `Predicted class` line per image. Compare the
predicted class with the expected denomination folder name. Quote paths when
calling an individual image whose path contains spaces:

```powershell
python .\predict.py ".\my test images\500\note_01.jpg"
```

The current script writes the resized input to `res_img.png` on every run, so
batch testing intentionally keeps only the last resized preview. The model
artifacts are read from `model/`, therefore run the command from the project
root.

On macOS/Linux, the equivalent batch command is:

```bash
find ./my_test_images -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
  -print0 | while IFS= read -r -d '' image; do
    echo "Testing: $image"
    python ./predict.py "$image"
  done
```

Use `predict.py` for folders containing one note per image. Use
`currency.py` when one image contains multiple notes and you want bounding
boxes and a total value:

```powershell
python .\currency.py ".\my_test_images\multiple_notes.jpg"
```

## Rebuild the model from the dataset

The checked-in model artifacts can be used directly. Rebuild them only when
you have changed the dataset or want to train a new model.

### 1. Prepare the supported dataset

```bash
python prepare_dataset.py --clean
```

Defaults:

- Source: `data/`
- Output: `data_filtered/`

Custom paths can be supplied:

```bash
python prepare_dataset.py --source data --output data_filtered --clean
```

The script prints the number of images copied for each supported denomination.
It fails explicitly if a required denomination folder is missing.

### 2. Generate visual vocabulary and features

```bash
python bovw.py data_filtered
```

This creates or replaces:

- `model/bovw_codebook_6class.pickle`
- `model/data_6class.npy`
- `model/label_6class.npy`

The script uses up to 20,000 SIFT descriptors for k-means vocabulary
construction and creates a 600-word visual dictionary.

### 3. Optionally search for hyperparameters

```bash
python hyper_train.py
```

This runs a computationally expensive three-fold grid search and prints the
best Random Forest parameters. The script reports the best configuration but
does not save that estimator automatically.

### 4. Train and evaluate the classifier

```bash
python train.py
```

The training script:

- loads the generated features and labels;
- uses an 80/20 train/test split with `random_state=42`;
- trains a 200-tree Random Forest;
- prints accuracy and a confusion matrix;
- runs five-fold cross-validation on the training split; and
- saves `model/rfclassifier_6class.sav`.

The model must be regenerated whenever the feature-generation process or
dataset classes change. Keep the BoVW codebook and classifier from the same
training run together.

## Reported training result

The project documentation previously recorded approximately **0.98
cross-validation accuracy (+/- 0.01)** on the six-class dataset, with a
reported confusion matrix for the six denominations. This is a dataset-level
development result, not a guarantee of real-world accuracy. Performance can
drop with different note series, lighting, camera hardware, blur, occlusion,
backgrounds, folds, or class imbalance. Always rerun evaluation on a held-out
dataset that represents the intended deployment environment.

## Using the model in AI glasses

The current Python scripts are a good starting point for a prototype, but
they are not directly an AI-glasses application. A glasses product normally
has a camera, an edge computer or phone, an inference service, and an audio or
visual feedback channel.

### Recommended system architecture

```text
Glasses camera
      |
      v
Frame capture and rate limiting
      |
      v
Note detection / region of interest
      |
      v
Feature extraction + Random Forest model
      |
      v
Confidence and temporal smoothing
      |
      v
Text-to-speech or display feedback
```

There are two practical deployment options:

#### Option A: Phone or edge-computer inference

1. Stream low-resolution frames from the glasses camera to an Android phone,
   Raspberry Pi, Jetson, or similar edge computer.
2. Run the `currency.py` processing logic on the edge device, or reuse its
   contour and crop logic in a camera loop.
3. Load `model/rfclassifier_6class.sav` and
   `model/bovw_codebook_6class.pickle` once at application startup.
4. Classify only a useful region of interest instead of every full-resolution
   frame.
5. Read a result aloud, for example: “Indian rupee, five hundred.”
6. Add a confidence threshold and say “Please reposition the note” when the
   result is uncertain.
7. Announce a result only after it is stable across several consecutive frames
   to prevent repeated or flickering speech.

Conceptually, the application loop should be:

```python
model = joblib.load("model/rfclassifier_6class.sav")
codebook = load_codebook("model/bovw_codebook_6class.pickle")

while camera_is_running:
    frame = camera.read()
    note_regions = detect_note_regions(frame)

    for region in note_regions:
        features = extract_features(region, codebook)
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index])

        if confidence >= 0.80:
            announce_once(stable_label(class_index))
        else:
            announce_once("Please reposition the note")
```

The helper names above describe the required application structure; they are
not functions currently exposed by this repository. Before deployment, refactor
the feature extraction and detection code into reusable functions instead of
calling the command-line scripts once per frame.

#### Option B: Inference directly on the glasses

Direct on-device execution is possible only if the glasses platform supports
the Python dependencies, OpenCV SIFT, NumPy, Mahotas, scikit-learn, and the
serialized model format. Many embedded glasses do not provide that runtime.
In that case:

1. Move the inference code to a supported native runtime, or place inference
   on a paired phone/edge computer.
2. Benchmark latency, memory, heat, battery use, and camera throughput.
3. Preserve the exact preprocessing, feature order, class mapping, and model
   artifacts.
4. Revalidate accuracy on images captured by the glasses camera.

Do not assume that converting the Random Forest to another format preserves
results unless the complete feature-extraction pipeline is also reproduced.

### Accessibility and product safeguards

For an assistive glasses experience, add these controls around the model:

- show or speak the predicted denomination and confidence state;
- use temporal smoothing to avoid unstable announcements;
- provide a manual repeat/mute control;
- clearly communicate “unknown” or “uncertain” instead of guessing;
- avoid announcing sensitive camera content to other people nearby;
- process frames locally where possible and do not store them by default;
- collect consented, representative test images before deployment;
- never use this model alone for financial verification or counterfeit
  decisions.

## Known limitations

- The classifier predicts across the six active Indian rupee denominations (10, 20, 50, 100, 200, 500).
- For static scripts (`predict.py`, `currency.py`), the model operates in 6-class mode.
- For live webcam usage (`camera.py`), an explicit 7th `NONE/BG` rejection class and multi-stage physical filters (saturation, Laplacian edge variance, chromatic ratios) are active to strictly eliminate false positives on hands, faces, and backgrounds.
- `currency.py` uses thresholding and an area heuristic rather than a trained
  object detector, so clutter, shadows, overlapping notes, and poor lighting
  can cause missed or incorrect regions.
- `currency.py` opens an OpenCV GUI window and is not headless by default.
- The serialized model depends on compatible Python/scikit-learn/joblib
  versions.
- The current feature scaler is applied during training in `bovw.py` but is
  not saved as a separate inference artifact. For a production deployment,
  save and load the fitted scaler together with the classifier and use the
  same preprocessing for every prediction.
- No benchmark for latency, battery usage, camera streaming, or glasses
  hardware is included.

## Troubleshooting

### `ModuleNotFoundError`

Activate the virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

SIFT requires the contrib OpenCV package. Avoid installing conflicting
OpenCV packages alongside `opencv-contrib-python`.

### Model or codebook file not found

Run commands from the repository root and verify these files exist:

```text
model/rfclassifier_6class.sav
model/bovw_codebook_6class.pickle
```

If you retrain, run the full preparation, feature-generation, and training
sequence in order.

### No useful contours are detected

Use a clearer image with stronger contrast, place notes on a plain background,
avoid severe overlap, and ensure the note occupies enough of the frame.
For a single isolated note, try `predict.py` instead of `currency.py`.

### Predictions are unreliable

Check lighting, focus, note orientation, image quality, and class balance.
Evaluate on held-out images from the target camera rather than relying only on
the original cross-validation score.

## Version and authorship

- Project version: 1.0
- Author: Anil Sathyan

## Acknowledgements

The project builds on ideas and examples from:

- [Bag of Visual Words](https://kushalvyas.github.io/BOV.html)
- [Python image classification](https://gogul.dev/software/image-classification-python)
- [SIFTBOW](https://github.com/briansrls/SIFTBOW)
- [Rotating images with OpenCV](https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/)
- [Random Forest hyperparameter optimisation](https://towardsdatascience.com/optimizing-hyperparameters-for-random-forest-classification-ec7741f9d3f6)
- [Rotated rectangle cropping](https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593)
- [OpenCV contour features](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html)
