"""
VQA Pipeline: CNN Perception + Symbolic Compositional Reasoning
===============================================================
A hybrid system for Visual Question Answering on a synthetic shapes dataset.

Architecture:
  Stage 1 — Visual Perception (CNN):
      Split each 30x30 image into a 3x3 grid of 10x10 cells.
      A shallow CNN classifies each non-empty cell into
      shape in {circle, square, triangle} and color in {red, green, blue}.
      Training labels are generated via a calibrated heuristic and then
      refined through self-training with query-answer feedback.

  Stage 2 — Query Parsing:
      Recursive-descent parser that converts parenthesised prefix queries
      into nested-tuple ASTs.

  Stage 3 — Compositional Reasoning:
      A rule-based recursive executor evaluates the AST over the detected
      objects, implementing spatial relations (left_of, right_of, above,
      below) and the logical operator 'is'.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Constants
# ============================================================

SHAPES = ['circle', 'square', 'triangle']
COLORS = ['red', 'green', 'blue']
SHAPE_TO_IDX = {s: i for i, s in enumerate(SHAPES)}
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

# Empirically-verified square profiles that the simple fill-ratio
# heuristic would misclassify as circles.  Discovered by measuring
# the net accuracy impact of flipping each profile's label on the
# full training set.
_SQUARE_OVERRIDE_PROFILES = frozenset({
    (4, 8, 8, 8, 8, 8, 8, 4),
    (6, 8, 8, 10, 10, 8, 8, 6),
    (4, 7, 8, 8, 8, 8, 7, 4),
    (4, 8, 8, 8, 8, 8, 7, 4),
    (2, 6, 8, 8, 10, 10, 8, 8, 6, 2),
    (2, 6, 8, 8, 10, 10, 8, 8, 6),
    (6, 8, 8, 9, 9, 8, 8, 6),
    (4, 7, 8, 8, 8, 8, 6, 4),
    (5, 8, 8, 8, 8, 8, 8, 4),
    (1, 6, 8, 8, 10, 10, 8, 8, 6),
})

# ============================================================
# Stage 1: Visual Perception
# ============================================================

# ---- Cell extraction ----

def extract_cells(image):
    """Split a 30x30 image into 9 cells of 10x10."""
    cells = []
    for row in range(3):
        for col in range(3):
            cell = image[row * 10:(row + 1) * 10, col * 10:(col + 1) * 10, :]
            cells.append(cell)
    return cells


# ---- Heuristic labeler (generates CNN training data) ----

def heuristic_label_cell(cell):
    """
    Classify a 10x10x3 cell by colour and shape using pixel analysis.
    Returns (shape, colour) or (None, None) for empty cells.
    """
    mask = cell.sum(axis=2) > 10
    n_pixels = int(mask.sum())
    if n_pixels < 5:
        return None, None

    # Colour: dominant RGB channel
    mean_rgb = cell[mask].astype(np.float32).mean(axis=0)
    color = COLORS[int(np.argmax(mean_rgb))]

    # Shape: row-width profile analysis
    row_widths = [int(mask[r].sum()) for r in range(10) if mask[r].sum() > 0]
    if len(row_widths) < 2:
        return None, None

    widths = np.array(row_widths)

    # Triangle: monotonically non-decreasing with non-trivial range
    if np.all(np.diff(widths) >= 0) and widths.max() - widths.min() > 1:
        return 'triangle', color

    # Check profile against calibrated square overrides
    nz_profile = tuple(row_widths)
    if nz_profile in _SQUARE_OVERRIDE_PROFILES:
        return 'square', color

    # Fill-ratio boundary: >= 0.90 => square, else circle
    fill_ratio = n_pixels / (len(widths) * int(widths.max()))
    if fill_ratio >= 0.90:
        return 'square', color

    return 'circle', color


def auto_label_cells(images):
    """
    Label all non-empty cells in *images* using the heuristic.
    Returns (cells_array, shape_labels, color_labels).
    """
    all_cells, all_shapes, all_colors = [], [], []
    for img in images:
        for cell in extract_cells(img):
            shape, color = heuristic_label_cell(cell)
            if shape is not None:
                all_cells.append(cell)
                all_shapes.append(SHAPE_TO_IDX[shape])
                all_colors.append(COLOR_TO_IDX[color])
    return np.array(all_cells), np.array(all_shapes), np.array(all_colors)


# ---- CNN model ----

class CellClassifier(nn.Module):
    """Shallow CNN with two heads for shape + colour."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.shape_head = nn.Linear(64, len(SHAPES))
        self.color_head = nn.Linear(64, len(COLORS))

    def forward(self, x):
        feat = self.features(x).view(x.size(0), -1)
        return self.shape_head(feat), self.color_head(feat)


class CellDataset(Dataset):
    def __init__(self, cells, shapes, colors):
        self.cells = torch.from_numpy(cells).permute(0, 3, 1, 2).float() / 255.0
        self.shapes = torch.from_numpy(shapes).long()
        self.colors = torch.from_numpy(colors).long()

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx], self.shapes[idx], self.colors[idx]


def train_cnn(cells, shapes, colors, epochs=30, batch_size=256, lr=1e-3, verbose=True):
    """Train the cell classifier and return (model, device)."""
    dataset = CellDataset(cells, shapes, colors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CellClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct_s = correct_c = total = 0
        for bx, bs, bc in loader:
            bx, bs, bc = bx.to(device), bs.to(device), bc.to(device)
            sl, cl = model(bx)
            loss = criterion(sl, bs) + criterion(cl, bc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            correct_s += (sl.argmax(1) == bs).sum().item()
            correct_c += (cl.argmax(1) == bc).sum().item()
            total += bx.size(0)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>2}/{epochs}  Loss {total_loss/total:.4f}  "
                  f"Shape {correct_s/total:.4f}  Color {correct_c/total:.4f}")
    model.eval()
    return model, device


# ---- CNN inference ----

def perceive_image(model, device, image):
    """Run CNN on a 30x30 image -> list of {shape, color, x, y} dicts."""
    cells = extract_cells(image)
    non_empty, positions = [], []
    for idx, cell in enumerate(cells):
        if cell.sum(axis=2).max() > 10:
            non_empty.append(cell)
            row, col = divmod(idx, 3)
            positions.append((col, row))
    if not non_empty:
        return []
    batch = (torch.from_numpy(np.array(non_empty))
             .permute(0, 3, 1, 2).float() / 255.0).to(device)
    with torch.no_grad():
        sl, cl = model(batch)
        sp = sl.argmax(1).cpu().numpy()
        cp = cl.argmax(1).cpu().numpy()
    return [{'shape': SHAPES[sp[i]], 'color': COLORS[cp[i]],
             'x': positions[i][0], 'y': positions[i][1]}
            for i in range(len(non_empty))]


# ============================================================
# Stage 2: Query Parsing
# ============================================================

def tokenize(query):
    tokens, i = [], 0
    while i < len(query):
        ch = query[i]
        if ch in '()':
            tokens.append(ch); i += 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < len(query) and query[j] not in '() \t\n':
                j += 1
            tokens.append(query[i:j]); i = j
    return tokens


def _parse(tokens, pos):
    if tokens[pos] == '(':
        pos += 1
        op = tokens[pos]; pos += 1
        args = []
        while tokens[pos] != ')':
            arg, pos = _parse(tokens, pos)
            args.append(arg)
        return (op, *args), pos + 1
    return tokens[pos], pos + 1


def parse_query(query_str):
    """Parse a parenthesised prefix query into a nested-tuple AST."""
    tree, _ = _parse(tokenize(query_str.strip()), 0)
    return tree


# ============================================================
# Stage 3: Compositional Reasoning
# ============================================================

def execute(tree, objects):
    """
    Evaluate the query AST over the list of detected objects.

    Leaves return the set of matching object indices.
    Relational operators (left_of, right_of, above, below) return sets.
    'is' returns a boolean.
    """
    if isinstance(tree, str):
        return {i for i, o in enumerate(objects)
                if o['color'] == tree or o['shape'] == tree}

    op = tree[0]

    if op == 'is':
        attr = tree[1]
        result_set = execute(tree[2], objects)
        return any(objects[i]['color'] == attr or objects[i]['shape'] == attr
                   for i in result_set)

    inner = execute(tree[1], objects)
    if not inner:
        return set()

    if op == 'left_of':
        return {i for i in range(len(objects))
                if any(objects[i]['x'] < objects[j]['x'] for j in inner)}
    if op == 'right_of':
        return {i for i in range(len(objects))
                if any(objects[i]['x'] > objects[j]['x'] for j in inner)}
    if op == 'above':
        return {i for i in range(len(objects))
                if any(objects[i]['y'] < objects[j]['y'] for j in inner)}
    if op == 'below':
        return {i for i in range(len(objects))
                if any(objects[i]['y'] > objects[j]['y'] for j in inner)}

    raise ValueError(f"Unknown operator: {op}")


# ============================================================
# Self-training label refinement
# ============================================================

def _build_cell_data(images):
    """Pre-compute per-image cell info with heuristic labels."""
    all_cells, all_shapes, all_colors = [], [], []
    image_cell_info = []
    for img in images:
        cells = extract_cells(img)
        img_cells = []
        for idx, cell in enumerate(cells):
            shape, color = heuristic_label_cell(cell)
            if shape is not None:
                gi = len(all_cells)
                row, col = divmod(idx, 3)
                all_cells.append(cell)
                all_shapes.append(SHAPE_TO_IDX[shape])
                all_colors.append(COLOR_TO_IDX[color])
                img_cells.append({'color': color, 'x': col, 'y': row, 'gi': gi})
        image_cell_info.append(img_cells)
    return (np.array(all_cells), np.array(all_shapes), np.array(all_colors),
            image_cell_info)


def _eval_labels(shapes_arr, image_cell_info, parsed_queries, labels):
    correct = 0
    for i, ici in enumerate(image_cell_info):
        objs = [{'shape': SHAPES[shapes_arr[c['gi']]],
                 'color': c['color'], 'x': c['x'], 'y': c['y']} for c in ici]
        if bool(execute(parsed_queries[i], objs)) == labels[i]:
            correct += 1
    return correct


def _single_cell_corrections(shapes_arr, image_cell_info, parsed_queries, labels):
    n_fixed = 0
    for i, ici in enumerate(image_cell_info):
        objs = [{'shape': SHAPES[shapes_arr[c['gi']]],
                 'color': c['color'], 'x': c['x'], 'y': c['y']} for c in ici]
        if bool(execute(parsed_queries[i], objs)) == labels[i]:
            continue
        fixes = []
        for j, c in enumerate(ici):
            orig = objs[j]['shape']
            for ns in SHAPES:
                if ns == orig:
                    continue
                objs[j]['shape'] = ns
                if bool(execute(parsed_queries[i], objs)) == labels[i]:
                    fixes.append((c['gi'], SHAPE_TO_IDX[ns]))
                objs[j]['shape'] = orig
        if len(fixes) == 1:
            shapes_arr[fixes[0][0]] = fixes[0][1]
            n_fixed += 1
    return n_fixed


def refine_labels(cells, shapes, colors, image_cell_info,
                  parsed_queries, labels, rounds=3, cnn_epochs=30, verbose=True):
    """
    Self-training loop:
      1. Apply unique single-cell corrections from query feedback
      2. Train CNN on current labels
      3. Adopt CNN predictions where they fix remaining wrong answers
    Returns the final (model, device).
    """
    best_model = best_device = None

    for r in range(1, rounds + 1):
        n_fix = _single_cell_corrections(shapes, image_cell_info,
                                         parsed_queries, labels)
        acc = _eval_labels(shapes, image_cell_info, parsed_queries, labels)
        if verbose:
            print(f"    Round {r}: {n_fix} corrections -> "
                  f"accuracy {acc}/{len(labels)} = {acc/len(labels):.4f}")

        model, dev = train_cnn(cells, shapes, colors,
                               epochs=cnn_epochs, verbose=False)
        best_model, best_device = model, dev

        # CNN batch-predict
        preds = []
        for s in range(0, len(cells), 512):
            b = (torch.from_numpy(cells[s:s+512])
                 .permute(0, 3, 1, 2).float() / 255.0).to(dev)
            with torch.no_grad():
                preds.append(model(b)[0].argmax(1).cpu().numpy())
        cnn_shapes = np.concatenate(preds)

        improved = shapes.copy()
        any_change = False
        for i, ici in enumerate(image_cell_info):
            objs_h = [{'shape': SHAPES[shapes[c['gi']]],
                       'color': c['color'], 'x': c['x'], 'y': c['y']}
                      for c in ici]
            if bool(execute(parsed_queries[i], objs_h)) == labels[i]:
                continue
            objs_c = [{'shape': SHAPES[cnn_shapes[c['gi']]],
                       'color': c['color'], 'x': c['x'], 'y': c['y']}
                      for c in ici]
            if bool(execute(parsed_queries[i], objs_c)) == labels[i]:
                for c in ici:
                    improved[c['gi']] = cnn_shapes[c['gi']]
                any_change = True

        imp_acc = _eval_labels(improved, image_cell_info, parsed_queries, labels)
        if imp_acc > acc:
            shapes[:] = improved
            if verbose:
                print(f"             CNN fixes -> {imp_acc}/{len(labels)}")
        elif n_fix == 0:
            break

    return best_model, best_device


# ============================================================
# Full Pipeline
# ============================================================

def evaluate(model, device, images, queries, labels=None):
    predictions, correct = [], 0
    for i in range(len(images)):
        objs = perceive_image(model, device, images[i])
        pred = bool(execute(parse_query(queries[i]), objs))
        predictions.append(pred)
        if labels is not None and pred == labels[i]:
            correct += 1
    acc = correct / len(images) if labels is not None else None
    return predictions, acc


def main():
    print("=" * 64)
    print("  VQA Pipeline: CNN Perception + Symbolic Reasoning")
    print("=" * 64)

    # ---- Load data ----
    print("\n[1/6] Loading data ...")
    train_images = np.load('train.large.input.npy')
    test_images  = np.load('test.input.npy')
    with open('train.large.query') as f:
        train_queries = [l.strip() for l in f]
    with open('test.query') as f:
        test_queries = [l.strip() for l in f]
    with open('train_large.output') as f:
        train_labels = [l.strip() == 'true' for l in f]
    with open('test.output') as f:
        test_labels = [l.strip() == 'true' for l in f]
    print(f"  Train: {len(train_images)} images   Test: {len(test_images)} images")

    # ---- Heuristic labelling ----
    print("\n[2/6] Heuristic cell labelling ...")
    cells, shapes, colors, ici = _build_cell_data(train_images)
    print(f"  {len(cells)} non-empty cells")
    print(f"  Shapes: { {s: int((shapes==i).sum()) for i,s in enumerate(SHAPES)} }")
    print(f"  Colors: { {c: int((colors==i).sum()) for i,c in enumerate(COLORS)} }")

    # ---- Self-training refinement ----
    print("\n[3/6] Self-training label refinement ...")
    parsed_train = [parse_query(q) for q in train_queries]
    model, device = refine_labels(
        cells, shapes, colors, ici,
        parsed_train, train_labels, rounds=3, cnn_epochs=25
    )

    # ---- Final CNN training on refined labels ----
    print("\n[4/6] Final CNN training (30 epochs) ...")
    model, device = train_cnn(cells, shapes, colors, epochs=30, batch_size=256)

    # ---- Evaluate ----
    print("\n[5/6] Evaluating on training set ...")
    _, train_acc = evaluate(model, device, train_images, train_queries, train_labels)
    print(f"  Train accuracy: {train_acc:.4f}  "
          f"({int(train_acc*len(train_images))}/{len(train_images)})")

    print("\n[6/6] Evaluating on test set ...")
    test_preds, test_acc = evaluate(model, device, test_images, test_queries, test_labels)
    print(f"  Test accuracy:  {test_acc:.4f}  "
          f"({int(test_acc*len(test_images))}/{len(test_images)})")

    # ---- Save outputs ----
    with open('predictions.txt', 'w') as f:
        for p in test_preds:
            f.write('true\n' if p else 'false\n')
    torch.save(model.state_dict(), 'cell_classifier.pth')
    print("\n  predictions.txt + cell_classifier.pth saved")
    print("=" * 64)


if __name__ == '__main__':
    main()
