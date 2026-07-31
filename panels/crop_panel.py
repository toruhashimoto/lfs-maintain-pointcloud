# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual input-crop panel: RealityScan-style region editing in the viewer.

The workflow this implements
----------------------------
Cropping the input cloud is the one measure that moved out-of-box splats
(-56/-59% on two captures), and until now it was CLI-only. This panel makes
it operable from the viewer the way RealityScan's reconstruction region is:

  1. the operator seeds a box (default ROI, or fitted to the cloud's
     percentiles), which appears in the viewport as the engine's own crop
     box node;
  2. adjusts it with the engine's native crop-box gizmo (the same tool the
     rendering panel uses -- this panel adds no gizmo of its own);
  3. watches a live sampled count of what the box would drop;
  4. exports a cropped COLMAP dataset (crop_input.crop_dataset: exact full
     pass, hardlinked cameras/images.txt, junctioned images/, provenance
     JSON) ready for --data-path.

Division of labour: everything testable lives in cropbox.py / crop_input.py
(fit_box, box_to_text, scan_points3d(sample_every), crop_dataset). This
file is only glue: widget state, the scene crop-box node, and two worker
threads. Long operations must not run in draw() -- it executes on the UI
thread every frame, and the export takes ~14 s on a 4M-point cloud.

Known limits (deliberate, v1):
  * The exported crop is the box's axis-aligned min/max. Rotating the crop
    box NODE with the transform gizmo is not reflected in the export; the
    panel says so rather than silently exporting a different region.
  * "inverse" crop boxes (keep-outside) are refused for export -- an
    inverted region would delete the subject and keep the background.
"""

import os
import threading

import lichtfeld as lf

try:
    from ..cropbox import box_to_text, fit_box, pad_box, parse_box
    from ..crop_input import (DEFAULT_PAD, crop_dataset, read_xyz_sample,
                              scan_points3d)
except ImportError:  # flat import when loaded outside the package
    from cropbox import box_to_text, fit_box, pad_box, parse_box
    from crop_input import (DEFAULT_PAD, crop_dataset, read_xyz_sample,
                            scan_points3d)

# Starting region offered by the "seed" button. Operations keep most
# subjects inside this frame; the operator adjusts from here with the gizmo.
DEFAULT_ROI = "-1,-1.5,-1:3,0.5,1"

# Sampled counting looks at every Nth point: ~1 s instead of ~7 s on a 4M
# cloud, and the kept/dropped FRACTION is what the operator is reading.
COUNT_SAMPLE_EVERY = 8


def _points3d_path(data_root):
    return os.path.join(data_root, "sparse", "0", "points3D.txt")


class InputCropPanel(lf.ui.Panel):
    """Crop the training input cloud to a region edited in the viewport."""

    id = "maintain_pointcloud.input_crop_panel"
    label = "Input Crop"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 221

    def __init__(self):
        self.data_root = ""
        self.pad = DEFAULT_PAD
        self.status = ""
        # Worker-thread results land in these; draw() only reads them.
        # Python attribute stores are GIL-atomic, same contract as the
        # anchor panel's request flags.
        self._busy = False
        self._result = None

    # ------------------------------------------------------------- scene box
    def _splat_node_id(self, scene):
        for node in scene.get_nodes():
            try:
                if node.splat_data() is not None:
                    return node.id
            except Exception:
                continue
        return None

    def _scene_box(self, create=False, seed=None):
        """(cropbox_data, id) of the training model's crop box, or None."""
        scene = lf.get_scene()
        splat_id = self._splat_node_id(scene)
        if splat_id is None:
            return None
        if not create:
            # get_or_create would add a node as a side effect of drawing.
            cb_id = None
            try:
                cb_id = scene.get_cropbox_for_splat(splat_id)
            except AttributeError:
                # Older binding: only get_or_create exists. Creating on
                # every frame is wrong, so treat "unknown" as absent until
                # the operator presses a seed button.
                return None
            if cb_id is None or int(cb_id) < 0:
                return None
            return scene.get_cropbox_data(cb_id), int(cb_id)
        cb_id = scene.get_or_create_cropbox_for_splat(splat_id)
        cb = scene.get_cropbox_data(cb_id)
        if seed is not None:
            lo, hi = seed
            # Tuples, not lists: the vec3 properties raise
            # RuntimeError('bad cast') on a list.
            cb.set("min", tuple(float(v) for v in lo))
            cb.set("max", tuple(float(v) for v in hi))
            cb.set("inverse", False)
            cb.set("enabled", True)
            scene.set_cropbox_data(cb_id, cb)
            cb = scene.get_cropbox_data(cb_id)
        return cb, int(cb_id)

    def _read_box(self, cb):
        lo = tuple(float(v) for v in cb.get("min"))
        hi = tuple(float(v) for v in cb.get("max"))
        # The gizmo may hand corners back in either order per axis.
        return (tuple(min(a, b) for a, b in zip(lo, hi)),
                tuple(max(a, b) for a, b in zip(lo, hi)))

    def _write_box(self, cb_id, box):
        scene = lf.get_scene()
        cb = scene.get_cropbox_data(cb_id)
        lo, hi = box
        cb.set("min", tuple(float(v) for v in lo))
        cb.set("max", tuple(float(v) for v in hi))
        cb.set("enabled", True)
        scene.set_cropbox_data(cb_id, cb)

    def _set_crop_gizmo(self, gizmo_type):
        """Switch the active crop tool's gizmo (translate/scale).

        Same call the main toolbar makes; exposed here so resizing is one
        click from the panel instead of a toolbar hunt.
        """
        try:
            lf.ui.set_active_operator("builtin.cropbox", gizmo_type)
            self.status = "crop gizmo: %s" % gizmo_type
        except Exception as exc:
            self.status = ("gizmo switch failed (%s); use the main "
                           "toolbar's move/scale instead" % exc)

    def _select_box_node(self, cb_id):
        """Select the crop box node, which activates the engine's crop tool.

        The gizmo is selection-driven: the viewer's NodeSelected handler is
        what calls setActiveOperator("builtin.cropbox"). A box that is merely
        VISIBLE has no gizmo, which is exactly the "I can see it but cannot
        adjust it" report this fixes.
        """
        try:
            scene = lf.get_scene()
            for node in scene.get_nodes():
                if int(node.id) == int(cb_id):
                    lf.select_node(node.name)
                    return True
        except Exception as exc:
            self.status = "select failed: %s" % exc
        return False

    # --------------------------------------------------------------- workers
    def _start(self, label, fn):
        if self._busy:
            return
        self._busy = True
        self._result = None
        self.status = label + "..."

        def run():
            try:
                self._result = fn()
            except Exception as exc:  # surfaced in the panel, not a traceback
                self._result = ("error", "%s" % exc)
            finally:
                self._busy = False

        threading.Thread(target=run, name="input_crop_worker",
                         daemon=True).start()

    def _count_job(self, points3d, box):
        kept, total = scan_points3d(points3d, box,
                                    sample_every=COUNT_SAMPLE_EVERY)
        return ("count", kept, total)

    def _export_job(self, data_root, box_text, pad):
        out = data_root.rstrip("\\/") + "_cropped"
        report = crop_dataset(data_root, out, box_text, pad=pad)
        return ("export", report)

    def _fit_job(self, points3d):
        xyz = read_xyz_sample(points3d, sample_every=COUNT_SAMPLE_EVERY)
        return ("fit", fit_box(xyz, 0.5, 99.5))

    # ------------------------------------------------------------------ draw
    def draw(self, ui):
        ui.heading("Input Crop")
        ui.text_wrapped(
            "Crop the training input cloud to a region, before training. "
            "Edit the region in the viewport with the crop-box gizmo; "
            "export writes <data>_cropped next to the dataset.")

        # -- dataset ------------------------------------------------------
        if not self.data_root:
            try:
                self.data_root = lf.dataset_params().data_path or ""
            except Exception:
                pass
        _, self.data_root = ui.input_text_with_hint(
            "COLMAP dataset", "(path holding sparse/0 and images)",
            self.data_root)
        points3d = _points3d_path(self.data_root) if self.data_root else ""
        have_cloud = bool(points3d) and os.path.isfile(points3d)
        if self.data_root and not have_cloud:
            ui.text_wrapped("points3D.txt not found under this path.")

        _, self.pad = ui.input_float(
            "Pad (fraction of each axis)", self.pad, 0.01, 0.05, "%.2f")

        # -- the box ------------------------------------------------------
        ui.separator()
        found = None
        try:
            found = self._scene_box(create=False)
        except Exception:
            found = None
        cb_id = None
        if found is not None:
            cb, cb_id = found
            box = self._read_box(cb)
            enabled = bool(cb.get("enabled"))
            inverse = bool(cb.get("inverse"))
            ui.text_wrapped("Region: %s%s" % (
                box_to_text(box),
                "" if enabled else "   [crop box disabled in scene]"))
            if inverse:
                ui.text_wrapped(
                    "This crop box is INVERTED (keeps the outside). "
                    "Export is disabled: it would delete the subject.")

            # Numeric editing: exact, always available, no tool required.
            # Two equivalent representations, SuperSplat-style: corner
            # min/max, and center+size (the natural one for "make it
            # smaller"). Both write straight into the crop box data, so
            # they are always what the export will use.
            lo, hi = list(box[0]), list(box[1])
            changed = False
            for axis in range(3):
                c, lo[axis] = ui.input_float(
                    "min %s" % "xyz"[axis], lo[axis], 0.01, 0.1, "%.4f")
                changed = changed or c
                ui.same_line()
                c, hi[axis] = ui.input_float(
                    "max %s" % "xyz"[axis], hi[axis], 0.01, 0.1, "%.4f")
                changed = changed or c

            center = [(l + h) / 2.0 for l, h in zip(lo, hi)]
            size = [max(h - l, 1e-3) for l, h in zip(lo, hi)]
            resized = False
            for axis in range(3):
                c, center[axis] = ui.input_float(
                    "center %s" % "xyz"[axis], center[axis], 0.01, 0.1,
                    "%.4f")
                resized = resized or c
                ui.same_line()
                c, size[axis] = ui.input_float(
                    "size %s" % "xyz"[axis], size[axis], 0.01, 0.1, "%.4f")
                resized = resized or c
            if ui.button("Shrink 5%"):
                size = [s * 0.95 for s in size]
                resized = True
            ui.same_line()
            if ui.button("Expand 5%"):
                size = [s * 1.05 for s in size]
                resized = True
            if resized:
                size = [max(s, 1e-3) for s in size]
                lo = [c - s / 2.0 for c, s in zip(center, size)]
                hi = [c + s / 2.0 for c, s in zip(center, size)]
                changed = True

            if changed:
                try:
                    self._write_box(cb_id, (tuple(lo), tuple(hi)))
                    box = (tuple(lo), tuple(hi))
                except Exception as exc:
                    self.status = "box update failed: %s" % exc

            if ui.button("Move (gizmo)"):
                if self._select_box_node(cb_id):
                    self._set_crop_gizmo("translate")
            ui.same_line()
            if ui.button("Scale (gizmo)"):
                if self._select_box_node(cb_id):
                    self._set_crop_gizmo("scale")
            ui.text_disabled(
                "Selecting the box (here or in the scene panel) activates "
                "the crop tool; the main toolbar's move/scale also switches "
                "the gizmo. After a gizmo drag, check the numbers above "
                "followed -- the export uses exactly these min/max values. "
                "Box-node rotation is NOT exported (axis-aligned only).")
        else:
            box = None
            ui.text_wrapped("No crop box on the training model yet. "
                            "Seed one:")

        if ui.button("Seed default region"):
            try:
                seeded = self._scene_box(create=True,
                                         seed=parse_box(DEFAULT_ROI))
                self.status = "seeded %s" % DEFAULT_ROI
                if seeded is not None:
                    self._select_box_node(seeded[1])
            except Exception as exc:
                self.status = "seed failed: %s" % exc
        ui.same_line()
        if have_cloud and not self._busy:
            if ui.button("Fit to cloud (p0.5-p99.5)"):
                self._start("fitting", lambda: self._fit_job(points3d))

        # -- actions ------------------------------------------------------
        ui.separator()
        can_count = have_cloud and box is not None and not self._busy
        if can_count and ui.button("Count (sampled)"):
            padded = pad_box(box, self.pad)
            self._start("counting",
                        lambda: self._count_job(points3d, padded))
        if can_count:
            ui.same_line()
            inverse_box = found is not None and bool(found[0].get("inverse"))
            if not inverse_box and ui.button_styled("Export cropped dataset",
                                                    "primary"):
                text = box_to_text(box)
                pad = self.pad
                self._start("exporting (exact pass)",
                            lambda: self._export_job(self.data_root, text,
                                                     pad))

        # -- status / results --------------------------------------------
        if self._busy:
            ui.text_wrapped(self.status)
        res = self._result
        if res is None:
            return
        kind = res[0]
        if kind == "error":
            ui.text_wrapped("Failed: %s" % res[1])
        elif kind == "count":
            _, kept, total = res
            dropped = total - kept
            ui.text_wrapped(
                "Sampled 1/%d: keeps %.2f%%, drops %.2f%% "
                "(~%d of ~%d points, pad %.2f included)"
                % (COUNT_SAMPLE_EVERY,
                   100.0 * kept / max(total, 1),
                   100.0 * dropped / max(total, 1),
                   dropped * COUNT_SAMPLE_EVERY,
                   total * COUNT_SAMPLE_EVERY, self.pad))
        elif kind == "fit":
            try:
                seeded = self._scene_box(create=True, seed=res[1])
                self.status = "fitted box to cloud percentiles"
                if seeded is not None:
                    self._select_box_node(seeded[1])
            except Exception as exc:
                self.status = "fit apply failed: %s" % exc
            self._result = None
        elif kind == "export":
            report = res[1]
            ui.text_wrapped(
                "Exported %s\nkept %d / %d (dropped %d, %.3f%%), pad %.2f\n"
                "Train with --data-path pointed at it."
                % (report["dst"], report["kept"], report["total"],
                   report["dropped"],
                   100.0 * report["dropped"] / max(report["total"], 1),
                   report["pad"]))
