# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit tests for cropping the surveyed cloud before training.

This is the one measure that moved out-of-box splats on the C capture:
-56.0% (effect/noise 15.5) against -8.6% for the best anchoring arm, at a
cost of 0.24 dB inside the delivered box. It works by deleting the 0.68% of
surveyed points that already lie outside the box, so training never grows
splats there.

The failure mode to guard against is a crop that looks like it worked. A
dropped track field, a stale `# Number of points:` header, or a box that
silently parsed to nothing all produce a dataset that trains without
complaint and invalidates the comparison.

Run:  .venv\\Scripts\\python.exe -m pytest mpc_tests/
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crop_input import crop_dataset, crop_points3d  # noqa: E402
from cropbox import inside_box  # noqa: E402

BOX = ((-1.0, -1.0, -1.0), (1.0, 2.0, 3.0))

# Point 2 is outside on x. Point 3 carries a multi-entry track, which is the
# field most likely to be mangled by a crop that reformats instead of copying.
POINTS3D = (
    b"# 3D point list with one line of data per point:\n"
    b"#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
    b"# Number of points: 3\n"
    b"1 0.0 0.0 0.0 1 2 3 0.5 10 0 11 1\n"
    b"2 9.0 0.0 0.0 4 5 6 0.5 12 0\n"
    b"3 -0.23906730947489219 0.5 1.0 7 8 9 0.5 13 0 14 2 15 3\n"
)


def _write_src(tmp_path, body=POINTS3D):
    src = tmp_path / "points3D.txt"
    src.write_bytes(body)
    return src


def test_counts_kept_and_total(tmp_path):
    kept, total = crop_points3d(_write_src(tmp_path),
                               tmp_path / "out.txt", BOX)
    assert (kept, total) == (2, 3)


def test_header_records_the_kept_count_not_the_original(tmp_path):
    # A stale count is the quiet failure: the file trains fine and every
    # downstream number derived from it (--max-cap, points-per-splat) is wrong.
    out = tmp_path / "out.txt"
    crop_points3d(_write_src(tmp_path), out, BOX)
    assert b"# Number of points: 2\n" in out.read_bytes()
    assert b"# Number of points: 3" not in out.read_bytes()


def test_kept_lines_are_copied_verbatim(tmp_path):
    # Re-serialising would round the coordinates and could reorder or drop the
    # track. The line must survive byte for byte.
    out = tmp_path / "out.txt"
    crop_points3d(_write_src(tmp_path), out, BOX)
    body = out.read_bytes()
    assert b"3 -0.23906730947489219 0.5 1.0 7 8 9 0.5 13 0 14 2 15 3\n" in body
    assert b"1 0.0 0.0 0.0 1 2 3 0.5 10 0 11 1\n" in body


def test_the_outside_point_is_gone(tmp_path):
    out = tmp_path / "out.txt"
    crop_points3d(_write_src(tmp_path), out, BOX)
    assert b"2 9.0 0.0 0.0" not in out.read_bytes()


def test_descriptive_comments_survive(tmp_path):
    out = tmp_path / "out.txt"
    crop_points3d(_write_src(tmp_path), out, BOX)
    assert out.read_bytes().startswith(
        b"# 3D point list with one line of data per point:\n#   POINT3D_ID,")


def test_line_endings_are_not_translated(tmp_path):
    # COLMAP writes LF even on Windows. Opening these files in text mode turns
    # every line into CRLF, changing the size of a 437 MB file and depending on
    # a reader's tolerance for it. Nothing here may introduce a CR.
    out = tmp_path / "out.txt"
    crop_points3d(_write_src(tmp_path), out, BOX)
    assert b"\r" not in out.read_bytes()


def test_dry_run_counts_without_writing(tmp_path):
    # Choosing a box means trying several. Each trial costs a 437 MB write if
    # counting requires producing the output.
    out = tmp_path / "out.txt"
    kept, total = crop_points3d(_write_src(tmp_path), None, BOX)
    assert (kept, total) == (2, 3)
    assert not out.exists()


def test_interior_point_is_inside():
    assert inside_box((0.0, 0.5, 1.0), BOX)


def test_point_beyond_any_single_axis_is_outside():
    # One axis is enough to reject; a bug that requires all three to be out
    # would keep most of the floaters the crop exists to remove.
    assert not inside_box((2.0, 0.5, 1.0), BOX)
    assert not inside_box((0.0, 9.0, 1.0), BOX)
    assert not inside_box((0.0, 0.5, -4.0), BOX)


def test_corners_are_kept():
    # Half-open intervals would drop the exact boundary. Harmless for random
    # points, but the padded box is derived from measured extremes, so the
    # extreme points are exactly the ones sitting on it.
    assert inside_box((-1.0, -1.0, -1.0), BOX)
    assert inside_box((1.0, 2.0, 3.0), BOX)


# --------------------------------------------------------------- whole dataset

BOX_TEXT = "-1,-1,-1:1,2,3"


def _make_capture(tmp_path):
    """A miniature COLMAP capture: sparse/0/*.txt plus images/."""
    root = tmp_path / "colmap"
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    (sparse / "points3D.txt").write_bytes(POINTS3D)
    (sparse / "cameras.txt").write_bytes(b"# cameras\n1 PINHOLE 8 8 4 4 4 4\n")
    (sparse / "images.txt").write_bytes(b"# images\n1 1 0 0 0 0 0 0 1 a.jpg\n\n")
    (root / "images").mkdir()
    (root / "images" / "a.jpg").write_bytes(b"jpeg-bytes")
    return root


def test_writes_a_cropped_cloud_beside_the_original(tmp_path):
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    report = crop_dataset(src, dst, BOX_TEXT, pad=0.0)
    assert report["kept"] == 2 and report["total"] == 3
    assert b"# Number of points: 2\n" in (dst / "sparse" / "0" / "points3D.txt").read_bytes()


def test_cameras_and_images_txt_are_hardlinked_not_copied(tmp_path):
    # images.txt is 479 MB on the real capture. Copying it per arm of a
    # campaign is the difference between 0.91 GB and tens of GB.
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    crop_dataset(src, dst, BOX_TEXT, pad=0.0)
    for name in ("cameras.txt", "images.txt"):
        a = os.stat(src / "sparse" / "0" / name)
        b = os.stat(dst / "sparse" / "0" / name)
        assert (a.st_ino, a.st_dev) == (b.st_ino, b.st_dev), name


def test_the_image_directory_is_shared_not_duplicated(tmp_path):
    # Proven behaviourally rather than by inspecting the link flavour: a file
    # added to the source afterwards must be visible through the copy.
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    crop_dataset(src, dst, BOX_TEXT, pad=0.0)
    assert (dst / "images" / "a.jpg").read_bytes() == b"jpeg-bytes"
    (src / "images" / "later.jpg").write_bytes(b"added-after")
    assert (dst / "images" / "later.jpg").read_bytes() == b"added-after"


def test_padding_defaults_to_five_percent_and_widens_the_box(tmp_path):
    # The delivered box is only knowable after delivery, so a box configured
    # in advance is nominal and needs margin. On C a tight crop dropped
    # 29,175 surveyed points and a 5% pad dropped 17,857.
    src = _make_capture(tmp_path)
    body = (b"# Number of points: 1\n"
            b"1 1.05 2.1 3.15 1 2 3 0.5 10 0\n")
    (src / "sparse" / "0" / "points3D.txt").write_bytes(body)
    tight = crop_dataset(src, tmp_path / "a", BOX_TEXT, pad=0.0)
    padded = crop_dataset(src, tmp_path / "b", BOX_TEXT)
    assert tight["kept"] == 0
    assert padded["kept"] == 1
    assert padded["pad"] == 0.05


def test_an_unset_box_is_an_error_not_a_passthrough(tmp_path):
    # Copying the cloud through unchanged would produce a dataset that looks
    # cropped, trains fine, and silently makes the comparison meaningless.
    src = _make_capture(tmp_path)
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            crop_dataset(src, tmp_path / "out", bad)


def test_refuses_to_overwrite_an_existing_cropped_dataset(tmp_path):
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    crop_dataset(src, dst, BOX_TEXT, pad=0.0)
    with pytest.raises(FileExistsError):
        crop_dataset(src, dst, BOX_TEXT, pad=0.0)


def test_dry_run_reports_counts_and_creates_nothing(tmp_path):
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    report = crop_dataset(src, dst, BOX_TEXT, pad=0.0, dry_run=True)
    assert (report["kept"], report["total"]) == (2, 3)
    assert not dst.exists()


def test_report_carries_the_padded_box_that_was_applied(tmp_path):
    # The stats have to say which box produced the dataset, or a campaign of
    # several boxes cannot be told apart afterwards.
    src = _make_capture(tmp_path)
    report = crop_dataset(src, tmp_path / "out", "0,0,0:10,2,4", pad=0.05)
    assert report["box"] == ((-0.5, -0.1, -0.2), (10.5, 2.1, 4.2))


def test_provenance_is_recorded_beside_the_dataset(tmp_path):
    # A campaign runs several boxes. Without the box written next to the
    # dataset, a cropped colmap directory is indistinguishable from any other
    # and the run that used it cannot be attributed.
    import json
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    crop_dataset(src, dst, BOX_TEXT, pad=0.0)
    saved = json.loads((dst / "crop_input.json").read_text())
    assert saved["box_requested"] == BOX_TEXT
    assert saved["box"] == [[-1.0, -1.0, -1.0], [1.0, 2.0, 3.0]]
    assert (saved["kept"], saved["total"], saved["dropped"]) == (2, 3, 1)


# ----------------------------------------------------------------------- CLI


def test_cli_crops_the_dataset(tmp_path):
    from crop_input import main
    src = _make_capture(tmp_path)
    dst = tmp_path / "out"
    assert main(["--data", str(src), "--out", str(dst),
                 "--box", BOX_TEXT, "--pad", "0"]) == 0
    assert b"# Number of points: 2\n" in (dst / "sparse" / "0" / "points3D.txt").read_bytes()


def test_cli_defaults_the_output_to_colmap_cropped(tmp_path):
    # The campaign runner is invoked with -Data ...\colmap_cropped, so the
    # default has to land exactly there.
    from crop_input import main
    src = _make_capture(tmp_path)
    assert main(["--data", str(src), "--box", BOX_TEXT, "--pad", "0"]) == 0
    assert (tmp_path / "colmap_cropped" / "sparse" / "0" / "points3D.txt").exists()


def test_cli_dry_run_reports_without_writing(tmp_path, capsys):
    from crop_input import main
    src = _make_capture(tmp_path)
    assert main(["--data", str(src), "--box", BOX_TEXT, "--dry-run"]) == 0
    assert not (tmp_path / "colmap_cropped").exists()
    assert "3" in capsys.readouterr().out


def test_cli_reports_a_bad_box_as_a_nonzero_exit(tmp_path):
    # A campaign script must not carry on to the training run believing the
    # dataset was prepared.
    from crop_input import main
    src = _make_capture(tmp_path)
    assert main(["--data", str(src), "--box", "1,2,3"]) != 0


def test_cli_accepts_a_box_whose_first_coordinate_is_negative(tmp_path):
    # Every real box starts with a negative number -- the measured C box is
    # -0.8777,-1.2579,-1.0:3.1204,0.3682,1.0. argparse treats a value starting
    # with '-' as another option and dies with "expected one argument", so the
    # separated form has to be glued before parsing.
    from crop_input import main
    src = _make_capture(tmp_path)
    assert main(["--data", str(src), "--box", "-1,-1,-1:1,2,3", "--pad", "0",
                 "--dry-run"]) == 0


def test_cli_still_accepts_the_equals_form(tmp_path):
    from crop_input import main
    src = _make_capture(tmp_path)
    assert main(["--data", str(src), "--box=-1,-1,-1:1,2,3", "--pad", "0",
                 "--dry-run"]) == 0


def test_shares_the_image_directory_given_forward_slash_paths(tmp_path):
    # mklink is a cmd builtin and rejects '/'-separated paths outright. Paths
    # arrive that way whenever the tool is driven from a POSIX-style shell, and
    # every pytest tmp_path is natively backslashed, so nothing else here
    # catches it -- it failed on the first real invocation.
    src = _make_capture(tmp_path)
    dst = tmp_path / "colmap_cropped"
    report = crop_dataset(str(src).replace("\\", "/"),
                          str(dst).replace("\\", "/"), BOX_TEXT, pad=0.0)
    assert report.get("images") in ("junction", "symlink")
    assert (dst / "images" / "a.jpg").read_bytes() == b"jpeg-bytes"


def test_negative_pad_is_an_error_not_a_shrink(tmp_path):
    # pad exists to protect geometry near the boundary; a negative value
    # silently inverts that intent and can empty the box entirely.
    src = _make_capture(tmp_path)
    with pytest.raises(ValueError):
        crop_dataset(src, tmp_path / "out", BOX_TEXT, pad=-0.1)


def test_scan_can_subsample_for_a_quick_estimate(tmp_path):
    # The GUI's live count runs on every Nth point so a 4M-point cloud
    # answers in ~1 s; the export still does the exact full pass.
    from crop_input import scan_points3d
    body = b"# Number of points: 4\n" + b"".join(
        b"%d %d.5 0.0 0.0 1 2 3 0.5 10 0\n" % (i, i) for i in range(4))
    src = tmp_path / "points3D.txt"
    src.write_bytes(body)
    # sample_every=2 sees data lines 0 and 2 only (x = 0.5, 2.5)
    kept, total = scan_points3d(src, ((0.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
                                sample_every=2)
    assert (kept, total) == (1, 2)


def test_read_xyz_sample_returns_every_nth_point(tmp_path):
    from crop_input import read_xyz_sample
    body = b"# c\n" + b"".join(
        b"%d %d.0 %d.0 %d.0 1 2 3 0.5 10 0\n" % (i, i, i * 2, i * 3)
        for i in range(5))
    src = tmp_path / "points3D.txt"
    src.write_bytes(body)
    xyz = read_xyz_sample(src, sample_every=2)
    assert xyz.shape == (3, 3)                     # lines 0, 2, 4
    assert list(xyz[1]) == [2.0, 4.0, 6.0]
