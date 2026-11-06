#.......................................part: 2................................................

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from scipy.ndimage import binary_erosion
import warnings
warnings.filterwarnings("ignore")

#.......................................Setup and Image Loading................................

image_path = 'image2/input.jpg'                          # path to input image
output_dir = 'image2'                                    # output folder for figures
os.makedirs(output_dir, exist_ok=True)

# load input image
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# resize image for manageable processing
MAX_SIDE = 512
h, w = img_rgb.shape[:2]
scale = min(1.0, MAX_SIDE / max(h, w))
img = cv2.resize(img_rgb, (int(w * scale), int(h * scale))).astype(np.float64)

H, W = img.shape[:2]                                     # dimensions of scaled image

#.......................................User Seeds and Bounding Rectangles.....................

# foreground seed rectangles on horses (x1, y1, x2, y2)
FG_BOXES = [
    (150, 125, 250, 175),
    (350, 150, 400, 175)
]

# background seed rectangles on grass (x1, y1, x2, y2)
BG_BOXES = [
    (25, 25, 75, 75),
    (400, 25, 450, 75),
    (25, 225, 100, 300),
    (275, 225, 325, 275)
]

# bounding rectangle enclosing both horses for GrabCut
GRAB_RECT = (95, 15, 505, 315)

# graph cut hyperparameters
LAMBDA = 3.5
SIGMA = 25.0

KMEANS_ITER = 15
GRABCUT_ITER = 6

GRAPH_MAX_SIDE = 180                                     # smaller resolution for graph cut max-flow

#.......................................Helper Functions.......................................

def rectangle_mask(shape, boxes):
    h_m, w_m = shape
    mask = np.zeros((h_m, w_m), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(w_m, int(x1)))
        x2 = max(0, min(w_m, int(x2)))
        y1 = max(0, min(h_m, int(y1)))
        y2 = max(0, min(h_m, int(y2)))
        mask[y1:y2, x1:x2] = True
    return mask

def get_boundary(mask):
    return mask ^ binary_erosion(mask)                   # 1-pixel boundary via morphological erosion

def show_segmentation(image, mask, title, save_path=None):
    mask = mask.astype(bool)
    if mask.sum() == 0 or (~mask).sum() == 0:
        return

    fg_mean = image[mask].mean(axis=0)                   # mean color of foreground pixels
    bg_mean = image[~mask].mean(axis=0)                  # mean color of background pixels

    result = np.zeros_like(image)
    result[mask] = fg_mean                               # replace foreground with fg mean
    result[~mask] = bg_mean                              # replace background with bg mean

    boundary = get_boundary(mask)
    result[boundary] = [255, 0, 0]                       # highlight class boundary in red

    plt.figure(figsize=(9, 6))
    plt.imshow(np.clip(result, 0, 255).astype(np.uint8))
    plt.title(title)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

def fit_gaussian(X):
    X = np.asarray(X).reshape(-1, 3)
    mean = X.mean(axis=0)                                # sample mean in RGB space
    cov = np.cov(X, rowvar=False)                        # 3x3 covariance matrix
    cov += np.eye(3) * 25.0                              # ridge regularization to avoid singularity
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    return mean, inv, logdet

def gaussian_log_likelihood(image, gaussian):
    mean, inv, logdet = gaussian
    diff = image - mean
    mahal = np.einsum("...i,ij,...j->...", diff, inv, diff)   # mahalanobis distance squared
    return -0.5 * (mahal + logdet + 3 * np.log(2 * np.pi))

def fit_two_gaussians(image, fg_mask, bg_mask):
    fg = fit_gaussian(image[fg_mask])
    bg = fit_gaussian(image[bg_mask])
    return fg, bg

# generate seed masks
fg_seed = rectangle_mask((H, W), FG_BOXES)
bg_seed = rectangle_mask((H, W), BG_BOXES)

fg_gaussian, bg_gaussian = fit_two_gaussians(img, fg_seed, bg_seed)

# display seed selection
plt.figure(figsize=(10, 6))
plt.imshow(img.astype(np.uint8))
plt.contour(fg_seed, levels=[0.5], colors="lime", linewidths=2)
plt.contour(bg_seed, levels=[0.5], colors="red", linewidths=2)
plt.title("Foreground seeds (green) and Background seeds (red)")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "seed_selection.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................a: Maximum Likelihood Classification...................

log_fg = gaussian_log_likelihood(img, fg_gaussian)
log_bg = gaussian_log_likelihood(img, bg_gaussian)

# normalize log-likelihoods for visualization
lik_fg = np.exp(np.clip(log_fg - np.max(log_fg), -40, 0))
lik_bg = np.exp(np.clip(log_bg - np.max(log_bg), -40, 0))

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].imshow(lik_fg, cmap="gray")
ax[0].set_title("Foreground Likelihood")
ax[0].axis("off")
ax[1].imshow(lik_bg, cmap="gray")
ax[1].set_title("Background Likelihood")
ax[1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "likelihood.png"), bbox_inches="tight", dpi=150)
plt.show()

# classify each pixel by maximum likelihood
ml_mask = log_fg > log_bg
ml_mask[fg_seed] = True                                  # enforce marked seeds
ml_mask[bg_seed] = False

show_segmentation(
    img, ml_mask,
    "Part (a): Maximum Likelihood Segmentation",
    save_path=os.path.join(output_dir, "likelihood_segmentation.png")
)

#.......................................b: K-means Clustering..................................

pixels = img.reshape(-1, 3)
fg_flat = fg_seed.flatten()
bg_flat = bg_seed.flatten()

# initialize centers using marked seed means
centers = np.array([
    img[fg_seed].mean(axis=0),
    img[bg_seed].mean(axis=0)
])

labels = np.zeros(len(pixels), dtype=np.int32)
wcss_values = []
snapshots = []

for iteration in range(KMEANS_ITER):
    # compute squared distance to both cluster means
    d_fg = np.sum((pixels - centers[0])**2, axis=1)
    d_bg = np.sum((pixels - centers[1])**2, axis=1)

    new_labels = (d_bg < d_fg).astype(np.int32)

    # preserve marked seeds in respective classes
    new_labels[fg_flat] = 0
    new_labels[bg_flat] = 1

    # update cluster means
    for k in range(2):
        selected = pixels[new_labels == k]
        if len(selected):
            centers[k] = selected.mean(axis=0)

    labels = new_labels

    # compute within-cluster sum of squares
    wcss = 0.0
    for k in range(2):
        selected = pixels[labels == k]
        if len(selected):
            wcss += np.sum((selected - centers[k])**2)
    wcss_values.append(wcss)

    if iteration in [0, 1, 2, 4, 9, KMEANS_ITER - 1]:
        snapshots.append((iteration + 1, labels.reshape(H, W).copy()))

# plot intermediate iterations
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (it, mask) in zip(axes.ravel(), snapshots):
    ax.imshow(mask == 0, cmap="gray")
    ax.set_title(f"Iteration {it}")
    ax.axis("off")
plt.suptitle("K-means: Intermediate Segmentations")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "intermediate_segmentation.png"), bbox_inches="tight", dpi=150)
plt.show()

# plot wcss convergence
plt.figure(figsize=(8, 4))
plt.plot(range(1, KMEANS_ITER + 1), wcss_values, marker="o")
plt.xlabel("Iteration")
plt.ylabel("WCSS")
plt.title("K-means: WCSS vs Iteration")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "K_means_graph.png"), bbox_inches="tight", dpi=150)
plt.show()

kmeans_mask = (labels.reshape(H, W) == 0)

show_segmentation(
    img, kmeans_mask,
    "Part (b): Converged K-means Segmentation",
    save_path=os.path.join(output_dir, "converged_K_mean_segmentation.png")
)

#.......................................c: Graph Cut using SciPy Maximum Flow..................

def graph_cut(image, fg_gauss, bg_gauss, fg_fixed, bg_fixed, lam=3.5, sigma=25.0):
    h_c, w_c = image.shape[:2]
    n = h_c * w_c

    # unary regional costs: negative log-likelihood
    lf = gaussian_log_likelihood(image, fg_gauss)
    lb = gaussian_log_likelihood(image, bg_gauss)
    Df = -lf
    Db = -lb

    # subtract minimum for numerical stability
    minimum = np.minimum(Df, Db)
    Df -= minimum
    Db -= minimum

    # hard terminal penalties for marked seeds
    INF = 1e6
    Df[bg_fixed] = INF
    Db[fg_fixed] = INF

    source = n
    sink = n + 1
    nodes = np.arange(n)

    rows = [np.full(n, source), nodes]
    cols = [nodes, np.full(n, sink)]
    vals = [Db.flatten(), Df.flatten()]

    # pairwise boundary penalties: 4-connected grid
    for dy, dx in [(0, 1), (1, 0)]:
        diff = image[:h_c - dy or None, :w_c - dx or None] - image[dy:, dx:]
        dist2 = np.sum(diff**2, axis=2)
        weight = lam * np.exp(-dist2 / (2 * sigma**2))

        p = (np.arange(h_c - dy)[:, None] * w_c + np.arange(w_c - dx)[None, :]).flatten()
        q = ((np.arange(h_c - dy)[:, None] + dy) * w_c + (np.arange(w_c - dx)[None, :] + dx)).flatten()
        capacities = weight.flatten()

        rows.extend([p, q])
        cols.extend([q, p])
        vals.extend([capacities, capacities])

    # build sparse adjacency matrix
    rows = np.concatenate([np.asarray(x).ravel() for x in rows])
    cols = np.concatenate([np.asarray(x).ravel() for x in cols])
    vals = np.concatenate([np.asarray(x).ravel() for x in vals])

    SCALE = 1000
    capacities = np.clip(np.rint(vals * SCALE), 0, 2_000_000_000).astype(np.int32)
    graph = sparse.coo_matrix((capacities, (rows, cols)), shape=(n + 2, n + 2), dtype=np.int32).tocsr()
    graph.sum_duplicates()

    # run maximum flow algorithm
    result = maximum_flow(graph, source, sink)

    # find reachable nodes from source in residual graph
    residual = graph - result.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()

    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    mask = np.zeros(n, dtype=bool)
    mask[reachable[reachable < n]] = True

    return mask.reshape(h_c, w_c), result.flow_value / SCALE

# downscale image to make graph cut fast
gscale = min(1.0, GRAPH_MAX_SIDE / max(H, W))
gh = max(1, int(H * gscale))
gw = max(1, int(W * gscale))
graph_img = cv2.resize(img, (gw, gh)).astype(np.float64)

# resize seed masks using nearest neighbors
fg_small = cv2.resize(fg_seed.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)
bg_small = cv2.resize(bg_seed.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(bool)

fg_g_small, bg_g_small = fit_two_gaussians(graph_img, fg_small, bg_small)

# default graph cut
gc_mask, flow = graph_cut(graph_img, fg_g_small, bg_g_small, fg_small, bg_small, LAMBDA, SIGMA)

show_segmentation(
    graph_img, gc_mask,
    f"Part (c): Graph Cut (lambda={LAMBDA}, sigma={SIGMA})",
    save_path=os.path.join(output_dir, "graph_cut.png")
)

# parameter study for lambda
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, lam_val, title in zip(axes, [LAMBDA / 10, LAMBDA * 10], ["Small lambda (lambda=0.35)", "Large lambda (lambda=35.00)"]):
    m, _ = graph_cut(graph_img, fg_g_small, bg_g_small, fg_small, bg_small, lam=lam_val, sigma=SIGMA)
    ax.imshow(m, cmap="gray")
    ax.set_title(f"{title}, sigma={SIGMA:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph_cut_lambda.png"), bbox_inches="tight", dpi=150)
plt.show()

# parameter study for sigma
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, sig_val, title in zip(axes, [SIGMA / 3, SIGMA * 3], ["Small sigma (sigma=8.33)", "Large sigma (sigma=75.00)"]):
    m, _ = graph_cut(graph_img, fg_g_small, bg_g_small, fg_small, bg_small, lam=LAMBDA, sigma=sig_val)
    ax.imshow(m, cmap="gray")
    ax.set_title(f"lambda={LAMBDA:.2f}, {title}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph_cut_sigma.png"), bbox_inches="tight", dpi=150)
plt.show()

#.......................................d: GrabCut-like Iterative Approach.....................

# convert bounding rectangle to graph image coordinates
rx1, ry1, rx2, ry2 = GRAB_RECT
x1 = int(np.clip(rx1 * gw / W, 0, gw - 1))
x2 = int(np.clip(rx2 * gw / W, 1, gw))
y1 = int(np.clip(ry1 * gh / H, 0, gh - 1))
y2 = int(np.clip(ry2 * gh / H, 1, gh))

# create bounding rectangle mask
rect_mask = np.zeros((gh, gw), dtype=bool)
rect_mask[y1:y2, x1:x2] = True

# rectangle interior initialized as tentative foreground
interior = binary_erosion(rect_mask, iterations=3)
if interior.sum() < 10:
    interior = rect_mask.copy()

initial_fg = graph_img[interior]
initial_bg = graph_img[~rect_mask]

fg_g = fit_gaussian(initial_fg)
bg_g = fit_gaussian(initial_bg)

# outside bounding box is definite background
fixed_bg = ~rect_mask
fixed_fg = np.zeros((gh, gw), dtype=bool)                # no fixed foreground seeds

# visualize initial bounding box
plt.figure(figsize=(9, 6))
plt.imshow(graph_img.astype(np.uint8))
plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
plt.title("Part (d): Initial GrabCut Rectangle")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "grab_cut_rect.png"), bbox_inches="tight", dpi=150)
plt.show()

# alternate between graph cut and Gaussian refitting
grab_snapshots = []
grab_mask = None

for iteration in range(GRABCUT_ITER):
    # E-step: assign pixels via graph cut
    new_mask, flow = graph_cut(graph_img, fg_g, bg_g, fixed_fg, fixed_bg, LAMBDA, SIGMA)

    # M-step: refit Gaussians from current segmentation
    if new_mask.sum() > 10 and (~new_mask).sum() > 10:
        fg_g = fit_gaussian(graph_img[new_mask])
        bg_g = fit_gaussian(graph_img[~new_mask])

    grab_mask = new_mask
    grab_snapshots.append(grab_mask.copy())

# display intermediate GrabCut iterations
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for i, ax in enumerate(axes.ravel()):
    if i < len(grab_snapshots):
        ax.imshow(grab_snapshots[i], cmap="gray")
        ax.set_title(f"Iteration {i + 1}")
    ax.axis("off")
plt.suptitle("Part (d): GrabCut-like Intermediate Results")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "graph_cut_intermediate.png"), bbox_inches="tight", dpi=150)
plt.show()

# final GrabCut segmentation
show_segmentation(
    graph_img, grab_mask,
    "Part (d): Final GrabCut-like Segmentation",
    save_path=os.path.join(output_dir, "grab_cut_segmentation.png")
)

# final binary mask
plt.figure(figsize=(9, 6))
plt.imshow(grab_mask, cmap="gray")
plt.title("Part (d): Final Binary Segmentation Mask")
plt.axis("off")
plt.savefig(os.path.join(output_dir, "binary_seg.png"), bbox_inches="tight", dpi=150)
plt.show()
