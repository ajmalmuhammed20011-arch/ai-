# SpaceSnap AI — NASA EPIC Feature Detection

A command-line computer-vision demo that segments a NASA EPIC image and marks cloud systems and a possible smoke plume. It uses OpenCV, scikit-learn, NumPy, and Matplotlib.

## Run the demo

Python 3.9 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python detect_features.py NASA_EPIC_Smoke_Over_Canada.jpg
```

The sample image is included in this repository. The script writes `feature_detection_result.png`, `segmentation_only.png`, `annotated_result.png`, and `image_explanation.txt` to the current directory. A sample explanation is included; each run replaces it with a summary containing that run's coverage percentages.

## Demo outputs

- `feature_detection_result.png` — original, segmentation, edge map, and annotated detections.
- `annotated_result.png` — cloud outlines and the possible smoke-plume box.
- `segmentation_only.png` — colour-coded pixel classes.
- `image_explanation.txt` — plain-language summary generated from the measured class coverage and the detector's labels.

These PNG files are sample outputs from the command-line demo; no video is included.

## Approach (100–150 words)

I built a classical computer-vision pipeline for detecting features in NASA EPIC satellite imagery. First, I masked the black background to isolate the Earth's disk. Then I applied K-Means clustering in Lab colour space to find dominant colour groups. Because land, ocean and haze overlap in colour, I classified each pixel using calibrated HSV and blue-minus-red rules into ocean, cloud, land, coastal water and smoke. Canny edge detection traced cloud boundaries, and morphological operations with contour detection outlined large cloud systems and wildfire smoke plumes. Finally, the program calculates the percentage coverage of each class. This approach needs no labelled training data, is fully explainable, and runs in seconds. It could be extended with a U-Net trained on labelled satellite data for higher accuracy.

## Sample image and source

The sample is NASA EPIC imagery acquired on May 30, 2019, showing wildfire smoke streaming east across Alberta, Saskatchewan, and Manitoba. This repository uses one sample image, not a labelled training dataset; K-Means is used for unsupervised colour clustering.

- Image file: `NASA_EPIC_Smoke_Over_Canada.jpg`
- Source and image context: [NASA Earth Observatory — An EPIC View of Smoke Over Canada](https://www.earthobservatory.nasa.gov/images/145119/an-epic-view-of-smoke-over-canada)
- Instrument: EPIC aboard NOAA's DSCOVR satellite.
- Credit: NASA Earth Observatory / DSCOVR EPIC.

## Limitations and AI assistance

The final class labels are based mainly on hand-set colour thresholds, and smoke detection uses a manually selected image region. The detections are approximate and are not validated scientific measurements; performance may not generalize to other images. The explanation file is a rule-based template using the script's measured coverage, not a trained language model.

AI assistance was used to organize this README and add the rule-based explanation output. The detector itself should be described and presented according to the project author's actual contributions and the challenge's AI-use rules.
