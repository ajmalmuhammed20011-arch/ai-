# Earth Feature Detection – NASA EPIC "Smoke Over Canada"

Classical computer-vision + unsupervised ML pipeline (OpenCV, K-Means, Canny).

## Run
    pip install -r requirements.txt
    python detect_features.py NASA_EPIC_Smoke_Over_Canada.jpg

## Outputs
- feature_detection_result.png  (4-panel figure: original, segmentation, edges, detections)
- segmentation_only.png, annotated_result.png

## Image source
NASA EPIC (Earth Polychromatic Imaging Camera) on NOAA's DSCOVR satellite, 1.5 million km from Earth.
https://epic.gsfc.nasa.gov  |  https://science.nasa.gov/earth/earth-observatory/
NASA imagery is public domain (credit: NASA/NOAA/DSCOVR EPIC).
