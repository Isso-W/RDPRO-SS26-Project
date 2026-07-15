# `proposals.md`: suggested KB changes (five-tier decision process)

> These are suggestions only. Review them before committing any KB data changes.

## Tier 0: confirmed (5 items)

- **[classification · backbone] efficientnet** × `class_imbalance` **[DOMINANCE]**
  - EfficientNet is already the top-1 result for the `class_imbalance` prototype query. No change is required.
- **[classification · backbone] efficientnet** × `fine_grained` **[DOMINANCE]**
  - EfficientNet is already the top-1 result for the `fine_grained` prototype query. No change is required.
- **[classification · loss] cross_entropy_loss** × `fine_grained` **[DOMINANCE]**
  - Cross-entropy is already the top-1 loss for the `fine_grained` prototype query. No change is required.
- **[classification · loss] cross_entropy_loss** × `medical` **[DOMINANCE]**
  - Cross-entropy is already the top-1 loss for the `medical` prototype query. No change is required.
- **[image_segmentation · backbone] unet** × `medical` **[DOMINANCE]**
  - U-Net is already the top-1 result for the `medical` prototype query. No change is required.

## Tier 3: new edge (1 item)

- **[classification · loss] cross_entropy_loss** × `class_imbalance` **[CONFLICT] [DOMINANCE]**
  - Proposed edge: `('cross_entropy_loss', 'focal_loss', preferred_when)` with `{'any': ['class_imbalance=True']}`. The target is the current top-1 result; scoring does not consume it.
  - Conflict: the reverse edge `focal_loss→cross_entropy_loss` already exists under an overlapping condition. Run the short A/B arbitration experiment before applying this proposal.

## Tier 5: cross-role, no matching RAG slot (1 item)

- **[object_detection · backbone] efficientnet** × `medical`
  - EfficientNet has the `engine` role, while the current top-1 YOLOv8 component has the `frame` role. Object-detection retrieval selects only a frame and does not model a separate engine slot, so no edge is proposed.

## Findings (3 items; record only)

- **[object_detection · backbone] unet** × `class_imbalance`
  - U-Net appears in the object-detection consensus (`support=0.6056`, `breadth=2`). The KB cannot represent this cross-task pattern, so record it without changing segmentation edges.
- **[object_detection · loss] cross_entropy_loss** × `class_imbalance` **[DOMINANCE]**
  - The consensus (`support=0.7105`) comes from compound losses. Flattening them to a single vote creates an artifact and does not establish a preference between cross-entropy and focal loss.
- **[object_detection · loss] cross_entropy_loss** × `medical`
  - The consensus (`support=0.7838`) also comes from compound losses. Record the finding without proposing an edge.
