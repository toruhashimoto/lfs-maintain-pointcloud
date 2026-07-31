# SPDX-FileCopyrightText: 2026 Toru Hashimoto
# SPDX-License-Identifier: GPL-3.0-or-later
"""The PLY header/payload boundary must be found byte-exactly.

Found in review: both readers skipped newline bytes after ``end_header``
with a while-loop, which also consumes the FIRST PAYLOAD BYTE whenever it
happens to be 0x0A or 0x0D -- the LSB of vertex 0's float32 x, so roughly
0.8% of real files. Everything after that reads shifted by one byte:
cleanup_cost.read_ply then dies on its size check, and inbox_rule's reader
returns garbage columns. The header ends with exactly one newline, so the
skip must consume exactly one (with an optional preceding CR).

Run:  .venv\\Scripts\\python.exe -m pytest mpc_tests/
"""

import os
import struct
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from cleanup_cost import read_ply  # noqa: E402
from inbox_rule import read_ply_full  # noqa: E402

# The properties both readers expect from a splat PLY.
_PROPS = (["x", "y", "z", "nx", "ny", "nz"]
          + ["f_dc_%d" % i for i in range(3)]
          + ["opacity"]
          + ["scale_%d" % i for i in range(3)]
          + ["rot_%d" % i for i in range(4)])


def _write_ply(path, xyz):
    """One-vertex binary PLY whose payload starts at an exact offset."""
    head = ("ply\nformat binary_little_endian 1.0\n"
            "element vertex 1\n"
            + "".join("property float %s\n" % p for p in _PROPS)
            + "end_header\n").encode("ascii")
    row = list(xyz) + [0.0] * (len(_PROPS) - 3)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(struct.pack("<%df" % len(row), *row))


def test_first_payload_byte_0x0a_is_not_eaten(tmp_path):
    # float32 x whose little-endian LSB is 0x0A: 0x3F80000A.
    x = struct.unpack("<f", bytes([0x0A, 0x00, 0x80, 0x3F]))[0]
    p = tmp_path / "lsb_0a.ply"
    _write_ply(p, (x, 2.0, 3.0))
    xyz, op, ax = read_ply(str(p))
    assert xyz[0][0] == x and xyz[0][1] == 2.0 and xyz[0][2] == 3.0
    full = read_ply_full(str(p))
    assert full["xyz"][0][0] == x


def test_first_payload_byte_0x0d_is_not_eaten(tmp_path):
    x = struct.unpack("<f", bytes([0x0D, 0x00, 0x80, 0x3F]))[0]
    p = tmp_path / "lsb_0d.ply"
    _write_ply(p, (x, 2.0, 3.0))
    xyz, _, _ = read_ply(str(p))
    assert xyz[0][0] == x
    assert read_ply_full(str(p))["xyz"][0][0] == x


def test_ordinary_first_byte_still_parses(tmp_path):
    p = tmp_path / "plain.ply"
    _write_ply(p, (1.0, 2.0, 3.0))
    xyz, _, _ = read_ply(str(p))
    assert tuple(xyz[0]) == (1.0, 2.0, 3.0)
    assert tuple(read_ply_full(str(p))["xyz"][0]) == (1.0, 2.0, 3.0)
