#.......................................part: 1................................................

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace, gaussian_filter, sobel
from heapq import heappush, heappop

#.......................................Setup and Image Loading................................

image_path = 'image/input.jpg'                           # input image path
output_dir = 'image'                                     # folder to store outputs
os.makedirs(output_dir, exist_ok=True)

image_bgr = cv2.imread(image_path)                       # read image using opencv
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)   # bgr to rgb for display
gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)       # grayscale conversion
gray = gray.astype(np.float32) / 255.0                   # normalize pixel intensities to [0, 1]

# scale down image so dijkstra runs in reasonable time
MAX_SIZE = 300
h, w = gray.shape
scale = min(1.0, MAX_SIZE / max(h, w))

if scale < 1.0:
    new_w = int(w * scale)
    new_h = int(h * scale)
    gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

H, W = gray.shape                                        # final dimensions of resized image

#.......................................1.1: Marr-Hildreth Edge Detector.......................

def marr_hildreth(image, sigma=1.5, slope_threshold=0.03):
    log_response = gaussian_laplace(image, sigma=sigma)  # laplacian of gaussian filter
    h_img, w_img = image.shape
    zero_crossings = np.zeros((h_img, w_img), dtype=np.uint8)

    # scan 8-connected neighbors for sign changes
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            y1 = max(0, -dy)
            y2 = min(h_img, h_img - dy)
            x1 = max(0, -dx)
            x2 = min(w_img, w_img - dx)

            current = log_response[y1:y2, x1:x2]
            neighbour = log_response[y1 + dy:y2 + dy, x1 + dx:x2 + dx]

            crossing = (current * neighbour < 0) & (np.abs(current - neighbour) >= slope_threshold)
            zero_crossings[y1:y2, x1:x2] |= crossing.astype(np.uint8)

    return log_response, zero_crossings

# run marr-hildreth with default parameters
sigma = 1.5
slope_threshold = 0.03
log_response, edges = marr_hildreth(gray, sigma, slope_threshold)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(gray, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(log_response, cmap="gray")
axes[1].set_title("Laplacian of Gaussian")
axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Marr-Hildreth Edges")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "marr_hildreth.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................1.2: Effect of Gaussian Smoothing Radius (sigma).......

sigmas = [0.5, 1.5, 2.5]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, s in zip(axes, sigmas):
    _, result = marr_hildreth(gray, sigma=s, slope_threshold=0.015)
    ax.imshow(result, cmap="gray")
    ax.set_title(f"Sigma = {s}")
    ax.axis("off")

plt.suptitle("Effect of Gaussian Smoothing Radius")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gaussian_radius_effect.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................1.3: Effect of Slope Threshold.........................

thresholds = [0.005, 0.015, 0.03]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, t in zip(axes, thresholds):
    _, result = marr_hildreth(gray, sigma=1.5, slope_threshold=t)
    ax.imshow(result, cmap="gray")
    ax.set_title(f"Slope Threshold = {t}")
    ax.axis("off")

plt.suptitle("Effect of Slope Threshold")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "slope_threshold_effect.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................2.1: Image Feature Maps................................

def compute_features(image, sigma=1.5, slope_threshold=0.03):
    log_response, zero_crossings = marr_hildreth(image, sigma, slope_threshold)

    smoothed = gaussian_filter(image, sigma=1.0)         # smooth slightly before derivatives
    gx = sobel(smoothed, axis=1)                         # horizontal gradient
    gy = sobel(smoothed, axis=0)                         # vertical gradient
    magnitude = np.sqrt(gx**2 + gy**2)                   # gradient magnitude
    orientation = np.arctan2(gy, gx)                     # gradient angle

    # compute continuous zero-crossing strength
    h_img, w_img = image.shape
    zero_strength = np.zeros_like(image)

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            y1 = max(0, -dy)
            y2 = min(h_img, h_img - dy)
            x1 = max(0, -dx)
            x2 = min(w_img, w_img - dx)

            curr = log_response[y1:y2, x1:x2]
            nbr = log_response[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
            crossing = curr * nbr < 0
            diff = np.abs(curr - nbr)
            zero_strength[y1:y2, x1:x2] = np.maximum(zero_strength[y1:y2, x1:x2], diff * crossing)

    # 99th percentile normalization
    def normalize(x):
        scale = np.percentile(x, 99)
        if scale < 1e-12:
            return np.zeros_like(x)
        return np.clip(x / scale, 0, 1)

    zero_strength = normalize(zero_strength)
    magnitude = normalize(magnitude)

    return {
        "log": log_response,
        "zero_crossings": zero_crossings,
        "zero_strength": zero_strength,
        "magnitude": magnitude,
        "orientation": orientation,
        "gx": gx,
        "gy": gy
    }

features = compute_features(gray)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(features["zero_strength"], cmap="gray")
axes[0].set_title("Zero Crossing Strength")
axes[1].imshow(features["magnitude"], cmap="gray")
axes[1].set_title("Gradient Magnitude")
axes[2].imshow(features["orientation"], cmap="hsv")
axes[2].set_title("Gradient Orientation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "gradient.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................2.2: 8-Connected Edge Cost Function....................

NEIGHBOURS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]

def get_edge_cost(y, x, ny, nx, features, mode="combined", weights=(1/3, 1/3, 1/3)):
    # zero-crossing cost (low cost on strong zero crossings)
    cz = 1.0 - 0.5 * (features["zero_strength"][y, x] + features["zero_strength"][ny, nx])

    # gradient magnitude cost (low cost on high gradient)
    cg = 1.0 - 0.5 * (features["magnitude"][y, x] + features["magnitude"][ny, nx])

    # unit vector of link direction
    dx = nx - x
    dy = ny - y
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length

    # orientation cost (low when link direction is perpendicular to gradient)
    orientation_costs = []
    for py, px in [(y, x), (ny, nx)]:
        gx = features["gx"][py, px]
        gy = features["gy"][py, px]
        norm = np.sqrt(gx**2 + gy**2)
        if norm < 1e-10:
            co = 0.5
        else:
            co = abs((gx / norm) * dx + (gy / norm) * dy)
        orientation_costs.append(co)
    co = np.mean(orientation_costs)

    if mode == "zero":
        cost = cz
    elif mode == "magnitude":
        cost = cg
    elif mode == "orientation":
        cost = co
    elif mode == "combined":
        wz, wg, wo = weights
        cost = wz * cz + wg * cg + wo * co
    else:
        raise ValueError("Invalid mode")

    return length * (cost + 1e-4)                       # weight scaled by euclidean step length

#.......................................2.3: Dijkstra Algorithm................................

def dijkstra(features, seed, mode="combined", weights=(1/3, 1/3, 1/3), target=None):
    h_img, w_img = features["magnitude"].shape
    sy, sx = seed

    distances = np.full((h_img, w_img), np.inf)
    predecessors = np.full((h_img, w_img, 2), -1, dtype=np.int32)
    visited = np.zeros((h_img, w_img), dtype=bool)

    distances[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]                                # priority queue min-heap

    while pq:
        current_dist, y, x = heappop(pq)

        if visited[y, x]:
            continue
        visited[y, x] = True

        if target is not None and (y, x) == target:     # stop early if target reached
            break

        for dy, dx in NEIGHBOURS:
            ny = y + dy
            nx = x + dx

            if not (0 <= ny < h_img and 0 <= nx < w_img):
                continue
            if visited[ny, nx]:
                continue

            edge_weight = get_edge_cost(y, x, ny, nx, features, mode=mode, weights=weights)
            new_dist = current_dist + edge_weight

            if new_dist < distances[ny, nx]:
                distances[ny, nx] = new_dist
                predecessors[ny, nx] = [y, x]
                heappush(pq, (new_dist, ny, nx))

    return distances, predecessors

# single seed visualization
seed = (150, 150)                                       # seed pixel (row, col)

plt.figure(figsize=(8, 6))
plt.imshow(image_rgb)
plt.scatter(seed[1], seed[0], color="cyan", s=100, edgecolors="black")
plt.title(f"Selected Seed: {seed}")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "seed.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................2.4: Distance Function Maps............................

modes = [
    ("zero", "Zero Crossing Only"),
    ("magnitude", "Gradient Magnitude Only"),
    ("orientation", "Gradient Orientation Only"),
    ("combined", "Combined Weight")
]

distance_maps = {}
predecessor_maps = {}

for mode, title in modes:
    distances, predecessors = dijkstra(features, seed, mode=mode)
    distance_maps[mode] = distances
    predecessor_maps[mode] = predecessors

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (mode, title) in zip(axes.ravel(), modes):
    ax.imshow(np.log1p(distance_maps[mode]), cmap="inferno")
    ax.scatter(seed[1], seed[0], c="cyan", s=60, edgecolors="black")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("Dijkstra Distance Maps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distance_map.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................3.1: Seed Selection and Intelligent Scissors...........

# seeds placed along flower boundary
seeds = [
    (44, 180), (43, 208), (56, 234), (75, 257),
    (99, 266), (120, 259), (142, 240), (166, 216),
    (179, 195), (168, 173), (151, 150), (129, 135),
    (108, 129), (88, 133), (67, 146), (52, 163)
]

plt.figure(figsize=(9, 7))
plt.imshow(image_rgb)
plt.scatter([x for y, x in seeds], [y for y, x in seeds], color="red", s=60)
plt.plot([x for y, x in seeds] + [seeds[0][1]], [y for y, x in seeds] + [seeds[0][0]], "r--")
plt.title("Manually Selected Boundary Seeds")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "boundary_seed.png"), bbox_inches="tight", dpi=150)
plt.show()

def reconstruct_path(predecessors, start, end):
    path = []
    current = end
    while current != start:
        path.append(current)
        py, px = predecessors[current[0], current[1]]
        if py == -1 or px == -1:
            return []
        current = (int(py), int(px))
    path.append(start)
    path.reverse()
    return path

def intelligent_scissors(features, seeds, weights=(1/3, 1/3, 1/3)):
    complete_path = []
    for i in range(len(seeds)):
        start = seeds[i]
        end = seeds[(i + 1) % len(seeds)]
        _, predecessors = dijkstra(features, start, mode="combined", weights=weights, target=end)
        segment = reconstruct_path(predecessors, start, end)
        if not segment:
            raise RuntimeError(f"Could not connect {start} to {end}")
        if complete_path:
            complete_path.extend(segment[1:])
        else:
            complete_path.extend(segment)
    return complete_path

def create_mask(boundary, shape):
    result = np.zeros(shape, dtype=np.uint8)
    polygon = np.array([(x, y) for y, x in boundary], dtype=np.int32)
    cv2.fillPoly(result, [polygon], 1)
    return result

def get_segmented_image(image, mask):
    result = image.copy()
    result[mask == 0] = 0
    return result

# compute original boundary and segmentation
weights_original = (1/3, 1/3, 1/3)
boundary_original = intelligent_scissors(features, seeds, weights_original)
mask_original = create_mask(boundary_original, (H, W))
segmented = get_segmented_image(image_rgb, mask_original)

fig, axes = plt.subplots(1, 3, figsize=(17, 6))
axes[0].imshow(image_rgb)
axes[0].scatter([x for y, x in seeds], [y for y, x in seeds], color="red", s=35)
axes[0].set_title("Original Image and Seeds")
axes[1].imshow(image_rgb)
axes[1].plot([x for y, x in boundary_original], [y for y, x in boundary_original], color="lime", linewidth=1.5)
axes[1].scatter([x for y, x in seeds], [y for y, x in seeds], color="red", s=30)
axes[1].set_title("Final Closed Boundary")
axes[2].imshow(segmented)
axes[2].set_title("Final Segmentation")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "segmentation_marr.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................4.1: Modification 1 - Canny Edge Detector..............

canny_edges = cv2.Canny((gray * 255).astype(np.uint8), threshold1=50, threshold2=150)
canny_strength = canny_edges.astype(np.float32) / 255.0

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].imshow(features["zero_strength"], cmap="gray")
axes[0].set_title("Marr-Hildreth Zero-Crossing Strength")
axes[1].imshow(canny_strength, cmap="gray")
axes[1].set_title("Canny Edge Map")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "canny_vs_marr.png"), bbox_inches="tight", dpi=150)
plt.show()

# run intelligent scissors using canny feature map
canny_features = features.copy()
canny_features["zero_strength"] = canny_strength
canny_boundary = intelligent_scissors(canny_features, seeds, weights_original)
canny_mask = create_mask(canny_boundary, (H, W))
canny_segmented = get_segmented_image(image_rgb, canny_mask)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(segmented)
axes[0].set_title("Marr-Hildreth Segmentation")
axes[1].imshow(canny_segmented)
axes[1].set_title("Canny-Based Segmentation")
axes[2].imshow(image_rgb)
axes[2].plot([x for y, x in canny_boundary], [y for y, x in canny_boundary], color="lime")
axes[2].scatter([x for y, x in seeds], [y for y, x in seeds], color="red", s=25)
axes[2].set_title("Canny-Based Boundary")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "segmentation_canny.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................4.2: Modification 2 - Emphasize Zero Crossings.........

weights_modified = (0.8, 0.1, 0.1)                      # heavier weight on edge crossings
boundary_modified = intelligent_scissors(features, seeds, weights_modified)
mask_modified = create_mask(boundary_modified, (H, W))
seg_modified = get_segmented_image(image_rgb, mask_modified)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(segmented)
axes[0].set_title("Equal Weights (1/3, 1/3, 1/3)")
axes[1].imshow(seg_modified)
axes[1].set_title("Modified Weights (0.8, 0.1, 0.1)")
axes[2].imshow(image_rgb)
axes[2].plot([x for y, x in boundary_original], [y for y, x in boundary_original], color="yellow", label="Equal Weights")
axes[2].plot([x for y, x in boundary_modified], [y for y, x in boundary_modified], color="red", label="Zero-Crossing Emphasized")
axes[2].legend()
axes[2].set_title("Boundary Comparison")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "og_vs_modified_weight.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................5.1: Save Masks and Segments...........................

cv2.imwrite(os.path.join(output_dir, "marr_hildreth_edges.png"), edges * 255)
cv2.imwrite(os.path.join(output_dir, "original_segmentation_mask.png"), mask_original * 255)
cv2.imwrite(os.path.join(output_dir, "canny_segmentation_mask.png"), canny_mask * 255)
cv2.imwrite(os.path.join(output_dir, "modified_weights_mask.png"), mask_modified * 255)
cv2.imwrite(os.path.join(output_dir, "final_segmentation.png"), cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
