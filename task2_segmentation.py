"""
Task 2 — Image Segmentation (HNRS)

Splits a Task 1 mask (white foreground on black background, one image
containing a whole handwritten number or expression) into individual
per-character segments, in left-to-right reading order — matching how
MNIST samples are structured: one square image per character.

Approach: connected-component analysis. Each separate blob of
foreground pixels is a candidate character, character fragment, or
piece of a multi-part symbol.

1. Label connected components.
2. Rescue undersized fragments — reassign broken-stroke pieces to the
   nearest full-sized component so they merge back into the character
   they split from (see rescue_small_fragments for the distance/threshold
   logic).
3. Merge components that are vertically stacked at the same horizontal
   position (e.g. a colon's two dots, a division sign's dash and dots)
   into one segment — see merge_stacked_components for the exact
   gap/overlap criteria that distinguish this from two side-by-side
   characters.
4. Drop remaining components below the noise threshold, judged by true
   pixel count rather than bounding-box area (see filter_noise_components).
5. Sort the remaining boxes left to right — the reading order of the
   number or expression. To mitigate misordering in slanted handwriting,
   the sort key evaluates the horizontal center of each box (x + w/2)
   rather than the strict left edge. Ties are broken by top-to-bottom
   y-center position, ensuring fully deterministic output.
6. Crop each character, pad to a centered square, then resize to a
   fixed size (28x28 by default, matching MNIST) — see extract_segments
   for why padding happens before resizing.

Known limitation: this is gap-based segmentation — it can only
separate characters that don't touch. Characters that are physically
connected in the ink (overlapping digits, or cursive strokes running
from one letter into the next) have no pixel gap to cut along, so they
stay one segment. That's a fundamental limit of this technique, not a
bug to tune away; handling connected handwriting would need a
different approach entirely (e.g. a sliding-window classifier or
object detection). Label-aware cropping doesn't help here either:
touching ink is one connected component to begin with, so there's only
one label to keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # (x, y, w, h)


# Configuration

@dataclass
class SegmentationConfig:
    """Tunable parameters for the segmentation pipeline.

    Attributes:
        merge_stacked: whether to merge vertically stacked components
            into one segment (see merge_stacked_components). Disable if
            the input is known to have no multi-part symbols.
        x_overlap_min_ratio: minimum required horizontal overlap, as a
            fraction of the narrower box's width, for two components to
            count as aligned at the same position. Lower this if a
            symbol's pieces aren't merging because they're not quite
            lined up on the x-axis.
        max_gap_ratio: maximum vertical gap allowed between two stacked
            pieces, as a fraction of the tallest character present in
            the image. Raise this if a symbol's pieces are spaced
            further apart than expected; lower it if two unrelated
            components are being merged by mistake.
        min_component_area: pixel-count threshold for both fragment
            rescue and noise filtering (see rescue_small_fragments,
            filter_noise_components). Set to None to disable both.
        fragment_bridge_dist: max distance for reassigning a fragment to
             a full-sized component (see rescue_small_fragments).
        pad_ratio: extra margin added around each cropped character,
            relative to its longer side, so it doesn't touch the edge
            of its square canvas.
        target_size: final width/height (in pixels) each segment is
            resized to after padding, so it matches a model's expected
            input size. Set to None to skip resizing and keep each
            segment at its own natural (padded) size instead.
    """

    merge_stacked: bool = True
    x_overlap_min_ratio: float = 0.3
    max_gap_ratio: float = 0.8
    min_component_area: int | None = 40
    fragment_bridge_dist: float = 25.0
    pad_ratio: float = 0.2
    target_size: int | None = 28


DEFAULT_CONFIG = SegmentationConfig()


# Segment representation

@dataclass
class Segment:
    """One detected character/symbol, before cropping.

    A bounding box alone can't tell a cropping step which pixels inside
    that rectangle actually belong to this character — label_ids
    carries that information forward from the original
    connected-component labeling, through fragment rescue, merging,
    and filtering, to the final crop.

    Attributes:
        box: (x, y, w, h) bounding box, in reading-order-independent
            image coordinates.
        label_ids: ids (from cv2.connectedComponentsWithStats) of every
            original component this segment is made of. A single,
            unbroken character has exactly one id; a character rebuilt
            from a rescued fragment, or a multi-part symbol (division
            sign, colon, "i" with its dot), has one id per original
            piece folded into it.
    """

    box: Box
    label_ids: frozenset[int] = field(default_factory=frozenset)


# Geometry helpers

def x_overlap_ratio(box_a: Box, box_b: Box) -> float:
    """Fraction of the narrower box's width that overlaps the other box, on the x-axis."""
    ax, aw = box_a[0], box_a[2]
    bx, bw = box_b[0], box_b[2]
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    narrower_width = min(aw, bw)
    return overlap / narrower_width if narrower_width > 0 else 0.0


def vertical_gap(box_a: Box, box_b: Box) -> int:
    """Vertical gap between two boxes.

    Positive: the boxes are genuinely stacked one above the other, with
    real empty space between them (e.g. a dot sitting above a dash).
    Zero or negative: the boxes' y-ranges touch or overlap — they
    occupy the same vertical band, as two characters sitting side by
    side on a line would.
    """
    ay, ah = box_a[1], box_a[3]
    by, bh = box_b[1], box_b[3]
    return max(ay, by) - min(ay + ah, by + bh)


# Fragment rescue

def rescue_small_fragments(
    mask_uint8: np.ndarray,
    min_component_area: int,
    max_bridge_dist: float,
) -> tuple[np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    """Reassign undersized components to the nearest full-sized component.

    A broken stroke (faint ink, a dry brush stroke, a scan artifact)
    can split one character into several connected components, each
    too small on its own to pass as a real character. Left alone,
    every one of those pieces would either be dropped by the noise
    filter or survive as its own malformed, partial segment.

    This can only ever grow existing full-sized components, never
    merge two of them into each other: only components already below
    min_component_area are eligible to be reassigned, so two components
    that are each already full-sized are never touched here, no matter
    how close together they sit.

    Args:
        mask_uint8: binary mask to label (white foreground on black
            background).
        min_component_area: components with fewer pixels than this are
            rescue candidates.
        max_bridge_dist: maximum distance-transform distance, in
            pixels, at which a fragment is still reassigned to its
            nearest full-sized neighbor.

    Returns:
        labels: the raw (H, W) label map from
            connectedComponentsWithStats.
        stats: the per-label stats array from
            connectedComponentsWithStats (box + area per original id).
        parent: {label id: label id}, mapping every component to
            itself by default, or to the full-sized component it was
            reassigned to. Callers group components by their parent to
            get the final, rescue-aware set of pieces.
        areas: {label id: integer pixel count} mapping each valid label
            id strictly bound to the allocated stats matrix to prevent
            index out-of-bounds in downstream processing.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    areas = {label: int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, stats.shape[0])}

    small = [label for label, area in areas.items() if area < min_component_area]
    large = [label for label, area in areas.items() if area >= min_component_area]

    parent = {label: label for label in areas.keys()}
    if small and large:
        for frag_label in small:
            frag_mask = labels == frag_label
            best_label, best_dist = None, None
            for large_label in large:
                outside_large = np.where(labels == large_label, 0, 255).astype(np.uint8)
                dist_to_large = cv2.distanceTransform(outside_large, cv2.DIST_L2, 5)
                nearest = dist_to_large[frag_mask].min()
                if best_dist is None or nearest < best_dist:
                    best_dist, best_label = nearest, large_label
            if best_dist is not None and best_dist <= max_bridge_dist:
                parent[frag_label] = best_label

    return labels, stats, parent, areas


# Merging

def merge_stacked_components(
    segments: list[Segment],
    x_overlap_min_ratio: float,
    max_gap_ratio: float,
) -> list[Segment]:
    """Merge components that are vertically stacked at the same horizontal position.

    This is a purely geometric test — it never compares the two
    components' sizes to each other — so it correctly groups multi-part
    symbols whether their pieces are different sizes (a division sign's
    dash + a smaller dot) or the same size (a colon's two equal dots).

    Merges only when there's a genuine vertical gap (via vertical_gap),
    the gap is small relative to the tallest character, and the boxes
    overlap substantially on the x-axis — see the checks below.

    Chains of more than two pieces (e.g. a division sign's dot-dash-dot)
    are grouped correctly via union-find, not just merged pairwise.
    When a group merges, its members' label_ids are unioned together,
    so the merged Segment still knows every original component it's
    built from — this is what lets extract_segments crop it
    label-aware later.

    Args:
        segments: Segments to test for stacking — typically one per
            rescue-aware group, from segment_characters.
        x_overlap_min_ratio: minimum required x-axis overlap ratio.
        max_gap_ratio: maximum allowed vertical gap, as a fraction of
            the tallest box in the image.

    Returns:
        List of merged Segments — boxes combined by min/max extent,
        label_ids combined by union.
    """
    if not segments:
        return []

    boxes = [s.box for s in segments]

    # Reference scale = the tallest component in the image (a full-size
    # character). Using each pair's own height would break down
    # whenever one of the two pieces is naturally very short (a dash),
    # making the allowed gap unrealistically tiny.
    reference_height = max(h for (_, _, _, h) in boxes)

    n = len(segments)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            box_a, box_b = boxes[i], boxes[j]
            gap = vertical_gap(box_a, box_b)
            if gap <= 0 or reference_height <= 0:
                continue  # same vertical band -> side by side, not stacked
            if gap > max_gap_ratio * reference_height:
                continue  # too far apart to be pieces of one symbol
            if x_overlap_ratio(box_a, box_b) < x_overlap_min_ratio:
                continue  # not aligned at the same horizontal position
            union(i, j)

    groups: dict[int, list[Segment]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(segments[i])

    merged = []
    for group in groups.values():
        x1 = min(s.box[0] for s in group)
        y1 = min(s.box[1] for s in group)
        x2 = max(s.box[0] + s.box[2] for s in group)
        y2 = max(s.box[1] + s.box[3] for s in group)
        combined_ids = frozenset().union(*(s.label_ids for s in group))
        merged.append(Segment(box=(x1, y1, x2 - x1, y2 - y1), label_ids=combined_ids))
    return merged


def filter_noise_components(
    segments: list[Segment],
    component_areas: dict[int, int],
    min_area: int,
) -> list[Segment]:
    """Drop segments whose total foreground pixel count is below min_area.

    Intended to run last, after fragment rescue and stacked-component
    merging: a fragment that legitimately belongs to a broken stroke or
    a stacked symbol has already been absorbed into its group by that
    point, so it's judged by the group's pixel count rather than its
    own — it survives. Only pieces that stayed isolated through both
    prior steps are candidates for removal here, which is exactly the
    set that plausible noise falls into.

    Args:
        segments: list of Segments, post-rescue and post-merge.
        component_areas: {original component id: pixel count}, produced
            alongside the initial labeling.
        min_area: minimum total foreground pixel count to keep.

    Returns:
        Filtered list of segments.
    """
    kept = []
    for seg in segments:
        area = sum(component_areas[label_id] for label_id in seg.label_ids)
        if area >= min_area:
            kept.append(seg)
    return kept


# Cropping and normalization

def pad_to_square(segment: np.ndarray, pad_ratio: float) -> np.ndarray:
    """Pad a cropped character mask into a centered square canvas.

    Args:
        segment: cropped uint8 mask of a single character (tight
            bounding box).
        pad_ratio: extra margin added around the character, relative to
            its longer side.

    Returns:
        uint8 square array with the character centered.
    """
    h, w = segment.shape
    side = int(max(h, w) * (1 + pad_ratio))

    square = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    square[y_off:y_off + h, x_off:x_off + w] = segment
    return square


def resize_square(square: np.ndarray, target_size: int) -> np.ndarray:
    """Resize an already-square segment to target_size x target_size.

    Uses INTER_AREA, the recommended OpenCV interpolation for shrinking
    an image (the typical case here, since character crops are usually
    larger than the 28x28 a model expects) — it averages pixels into
    each output pixel instead of just sampling, which avoids the
    aliasing/broken-up-looking strokes that nearest/bilinear resizing
    can produce on thin lines.

    Args:
        square: uint8 square array (equal width and height).
        target_size: desired output width/height in pixels.

    Returns:
        uint8 array of shape (target_size, target_size).
    """
    return cv2.resize(square, (target_size, target_size), interpolation=cv2.INTER_AREA)


# Public API

def segment_characters(
    mask_uint8: np.ndarray,
    config: SegmentationConfig = DEFAULT_CONFIG,
) -> tuple[list[Segment], np.ndarray]:
    """Locate individual character/symbol segments in a Task 1 mask.

    Runs fragment rescue, then stacked-component merge, then noise
    filtering, in that order (see module docstring).

    Args:
        mask_uint8: binary mask from Task 1 preprocessing (white
            foreground on black background).
        config: tunable pipeline parameters. See SegmentationConfig.

    Returns:
        segments: list of Segments (box + originating label_ids),
            sorted left to right (reading order).
        labels: the raw (H, W) label map from
            connectedComponentsWithStats — each pixel's value is the id
            of the component it belongs to (0 = background). Callers
            need this alongside segments to crop label-aware in
            extract_segments.
    """
    min_area = config.min_component_area if config.min_component_area is not None else 0
    labels, stats, parent, component_areas = rescue_small_fragments(
        mask_uint8, min_area, config.fragment_bridge_dist
    )

    # Group every valid component under its rescue parent.
    groups: dict[int, set[int]] = {}
    for label in component_areas.keys():
        groups.setdefault(parent[label], set()).add(label)

    segments = []
    for members in groups.values():
        x1 = min(int(stats[m, 0]) for m in members)
        y1 = min(int(stats[m, 1]) for m in members)
        x2 = max(int(stats[m, 0]) + int(stats[m, 2]) for m in members)
        y2 = max(int(stats[m, 1]) + int(stats[m, 3]) for m in members)
        segments.append(Segment(box=(x1, y1, x2 - x1, y2 - y1), label_ids=frozenset(members)))

    if config.merge_stacked:
        segments = merge_stacked_components(segments, config.x_overlap_min_ratio, config.max_gap_ratio)

    if config.min_component_area is not None:
        segments = filter_noise_components(segments, component_areas, config.min_component_area)

    # Sort segments by center X to accommodate slanted handwriting overlap.
    # Ties resolve top-to-bottom by center Y.
    segments.sort(key=lambda seg: (
        seg.box[0] + seg.box[2] // 2,
        seg.box[1] + seg.box[3] // 2
    ))
    
    return segments, labels


def extract_segments(
    labels: np.ndarray,
    segments: list[Segment],
    config: SegmentationConfig = DEFAULT_CONFIG,
) -> list[np.ndarray]:
    """Crop each segment, pad it to a centered square, then resize.

    Label-aware cropping: within a segment's bounding rectangle, only
    pixels whose label is one of that segment's label_ids are kept —
    everything else (e.g. a neighboring character's stroke poking into
    the corner of this rectangle) is zeroed out. This is what stops a
    comma tucked under a "7" from surviving in the "7" crop, and vice
    versa, without changing either box's shape or position.

    Padding happens before resizing, not after: resizing a tight,
    non-square crop directly would stretch each axis by a different
    factor (distorting the character's shape), while resizing an
    already-square padded crop scales both axes equally.

    Args:
        labels: the (H, W) label map returned by segment_characters.
        segments: list of Segments, e.g. from segment_characters.
        config: tunable pipeline parameters. See SegmentationConfig. If
            config.target_size is None, segments are returned at their
            own padded size instead of being resized to a fixed size.

    Returns:
        List of uint8 arrays, one per segment, in the same order as
        segments — each of shape (target_size, target_size) if
        target_size is set, otherwise each at its own (different)
        padded size.
    """
    outputs = []
    for seg in segments:
        x, y, w, h = seg.box
        region_labels = labels[y:y + h, x:x + w]
        cleaned = np.where(np.isin(region_labels, list(seg.label_ids)), 255, 0).astype(np.uint8)

        square = pad_to_square(cleaned, config.pad_ratio)
        if config.target_size is not None:
            square = resize_square(square, config.target_size)
        outputs.append(square)
    return outputs


def segments_to_model_input(
    segments: list[np.ndarray],
    add_channel_dim: bool = True,
) -> np.ndarray:
    """Stack segments into a single normalized batch, ready for a Task 3 model.

    All segments must already be the same size — i.e. produced with a
    fixed config.target_size, not target_size=None.

    Args:
        segments: list of uint8 arrays, e.g. from extract_segments.
        add_channel_dim: if True (default), adds a trailing channel
            axis of size 1, giving shape (N, H, W, 1) — the input shape
            most Keras/TensorFlow CNNs trained on MNIST expect. Set to
            False for a model that wants (N, H, W) instead.

    Returns:
        float32 array of shape (N, H, W, 1) or (N, H, W), values in
        [0.0, 1.0].

    Raises:
        ValueError: if segments is empty, or the segments aren't all
            the same shape (e.g. because target_size was None when they
            were created).
    """
    if not segments:
        raise ValueError("segments is empty — nothing to stack into a batch")

    shapes = {seg.shape for seg in segments}
    if len(shapes) > 1:
        raise ValueError(
            f"segments have inconsistent shapes {shapes} — make sure they were "
            "created with a fixed config.target_size, not target_size=None"
        )

    batch = np.stack(segments).astype(np.float32) / 255.0
    if add_channel_dim:
        batch = batch[..., np.newaxis]
    return batch


def segment_and_extract(
    mask_uint8: np.ndarray,
    config: SegmentationConfig = DEFAULT_CONFIG,
) -> tuple[list[Box], list[np.ndarray]]:
    """Convenience wrapper: locate segments and crop them in one call.

    Args:
        mask_uint8: binary mask from Task 1 preprocessing.
        config: tunable pipeline parameters. See SegmentationConfig.

    Returns:
        A tuple (boxes, crops):
            boxes: list of (x, y, w, h), one per detected character, in
                reading order — for callers that only care about
                position (e.g. drawing boxes on the original image).
            crops: list of cropped/padded/resized arrays, same order as
                boxes, e.g. ready for segments_to_model_input.
    """
    segments, labels = segment_characters(mask_uint8, config)
    crops = extract_segments(labels, segments, config)
    boxes = [seg.box for seg in segments]
    return boxes, crops


def draw_segmentation(mask_uint8: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Draw detected bounding boxes over the mask, for a visual sanity check.

    Args:
        mask_uint8: binary mask the boxes were detected on.
        boxes: list of (x, y, w, h), e.g. from segment_and_extract.

    Returns:
        BGR uint8 image with red bounding boxes drawn over the mask,
        ready to display.
    """
    vis = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis