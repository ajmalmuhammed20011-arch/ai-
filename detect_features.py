"""
Earth Feature Detection from NASA EPIC imagery (Smoke over Canada)
Classifies clouds, ocean, land and smoke haze using K-Means clustering
+ HSV colour rules, and detects cloud-system edges/vortices with OpenCV.

Usage:  python detect_features.py NASA_EPIC_Smoke_Over_Canada.jpg
"""
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans

path = sys.argv[1] if len(sys.argv) > 1 else "NASA_EPIC_Smoke_Over_Canada.jpg"
bgr = cv2.imread(path)
if bgr is None:
    sys.exit(f"Cannot read {path}")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]

# ---------- 1. Isolate the Earth disk (remove black space) ----------
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
disk = (gray > 18).astype(np.uint8)
disk = cv2.morphologyEx(disk, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
disk = cv2.erode(disk, np.ones((15, 15), np.uint8))   # drop limb-darkened edge
disk = disk.astype(bool)

# ---------- 2. Unsupervised segmentation (K-Means in Lab space) ----------
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
pix = lab[disk].astype(np.float32)
sample = pix[np.random.RandomState(0).choice(len(pix), 60000, replace=False)]
km = KMeans(n_clusters=6, n_init=5, random_state=0).fit(sample)
clusters = np.full((h, w), -1, int)
clusters[disk] = km.predict(pix)

# ---------- 3. Classify every pixel with calibrated colour rules ----------
# (K-Means clusters above give the dominant palette; per-pixel rules give the
#  physical classes, because land/ocean/haze overlap in a single cluster.)
CLASSES = {0: "Ocean", 1: "Cloud", 2: "Land", 3: "Smoke / haze", 4: "Coastal / hazy water"}
COLORS = {0: (20, 70, 190), 1: (245, 245, 245), 2: (60, 160, 60),
          3: (255, 140, 0), 4: (0, 220, 220), -1: (0, 0, 0)}
sm_rgb = cv2.GaussianBlur(rgb, (0, 0), 4).astype(int)
hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (0, 0), 4), cv2.COLOR_BGR2HSV).astype(int)
R, G, B = sm_rgb[..., 0], sm_rgb[..., 1], sm_rgb[..., 2]
S, V = hsv[..., 1], hsv[..., 2]
yy, xx = np.mgrid[0:h, 0:w]
bmr = B - R                                       # blue-minus-red index

label_map = np.full((h, w), -1, int)
label_map[disk] = 2                               # default: land
label_map[disk & (bmr >= 40)] = 0                 # strongly blue  -> ocean
label_map[disk & (bmr >= 20) & (bmr < 40)] = 4    # mildly blue    -> hazy / coastal water
label_map[disk & (V > 185) & (S < 60)] = 1        # bright + grey  -> cloud
# smoke: slightly blue-grey/warm, mid-brightness, low saturation, over N. America
smoke = (disk & (V > 105) & (V <= 185) & (S < 70) & (bmr > 5) & (bmr < 35)
         & (yy > h * 0.27) & (yy < h * 0.42) & (xx > w * 0.46) & (xx < w * 0.68)
         & (bmr > 8) & (bmr < 28))   # region of interest: central Canada
smoke = cv2.morphologyEx(smoke.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
# keep only the 2 largest smoke blobs (removes speckle)
n, lab_cc, st, _ = cv2.connectedComponentsWithStats(smoke.astype(np.uint8))
keep = 1 + np.argsort(st[1:, cv2.CC_STAT_AREA])[::-1][:2]
smoke = np.isin(lab_cc, keep[st[keep, cv2.CC_STAT_AREA] > 1500])
label_map[smoke] = 3

seg = np.zeros((h, w, 3), np.uint8)
for k, c in COLORS.items():
    seg[label_map == k] = c

# ---------- 4. Edge / cloud-structure detection ----------
blur = cv2.GaussianBlur(gray, (0, 0), 3)
edges = cv2.Canny(blur, 30, 90)
edges[~disk] = 0
cloud_mask = (label_map == 1).astype(np.uint8)
cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big = [c for c in contours if cv2.contourArea(c) > 8000]
annot = rgb.copy()
cv2.drawContours(annot, big, -1, (255, 255, 0), 3)

# Smoke plume bounding boxes
sm = cv2.morphologyEx((label_map == 3).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
sc, _ = cv2.findContours(sm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in sc:
    if cv2.contourArea(c) > 6000:
        x, y, bw, bh = cv2.boundingRect(c)
        cv2.rectangle(annot, (x, y), (x + bw, y + bh), (255, 0, 0), 4)
        cv2.putText(annot, "Smoke plume", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 0), 3)

# ---------- 5. Coverage statistics ----------
total = disk.sum()
stats = {CLASSES[k]: 100 * (label_map == k).sum() / total for k in CLASSES}
print("Surface / atmosphere coverage (% of visible disk):")
for k, v in stats.items():
    print(f"  {k:<15}{v:6.2f}%")

# A short, rule-based explanation from the measured class coverage. This is
# generated from this run's output; it is not a trained vision-language model.
ranked_classes = sorted(stats.items(), key=lambda item: item[1], reverse=True)
summary_lines = [
    "This NASA EPIC image shows the visible disk of Earth against black space.",
    "The image was segmented into approximate ocean, cloud, land, smoke/haze, "
    "and coastal/hazy-water classes using colour rules.",
    "Estimated coverage of the visible Earth disk:",
]
summary_lines.extend(f"- {name}: {value:.1f}%" for name, value in ranked_classes)
summary_lines.extend([
    "The red box in annotated_result.png marks a region classified as a "
    "possible smoke plume; yellow outlines mark large cloud regions.",
    "These are approximate image-based labels, not validated scientific "
    "measurements. Smoke detection uses a manually selected region and "
    "colour thresholds, so results may not transfer to other images.",
])
with open("image_explanation.txt", "w", encoding="utf-8") as explanation_file:
    explanation_file.write("\n".join(summary_lines) + "\n")

# ---------- 6. Save figure ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 14))
ax[0, 0].imshow(rgb);        ax[0, 0].set_title("1. Original NASA EPIC image")
ax[0, 1].imshow(seg);        ax[0, 1].set_title("2. Segmentation (K-Means + colour rules)")
ax[1, 0].imshow(edges, cmap="gray"); ax[1, 0].set_title("3. Canny edge detection")
ax[1, 1].imshow(annot);      ax[1, 1].set_title("4. Detected cloud systems (yellow) & smoke (red)")
for a in ax.ravel():
    a.axis("off")
fig.legend(handles=[Patch(color=np.array(COLORS[k]) / 255, label=CLASSES[k]) for k in CLASSES],
           loc="lower center", ncol=5, fontsize=12)
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("feature_detection_result.png", dpi=110)
cv2.imwrite("segmentation_only.png", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
cv2.imwrite("annotated_result.png", cv2.cvtColor(annot, cv2.COLOR_RGB2BGR))
print("Saved: feature_detection_result.png, segmentation_only.png, annotated_result.png")
print("Saved: image_explanation.txt")
