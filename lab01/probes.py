"""Probes — read what the machine says about itself.

INSTRUCTOR SOLUTION. Do not distribute. The student copy of this file has the
body of every function below replaced by `raise NotImplementedError`.

Every probe takes a `root` argument and reads nothing outside it. That is not
decoration: it is what makes this lab gradeable without twenty boards on a
desk, and it is the reason the test suite can present a fake SD-booted machine
and check that the student's code notices. Code that hardcodes "/" cannot be
tested, and a measurement you cannot test is a measurement you cannot trust —
which is the whole argument of Lecture 01, applied to the student's own code.

Each probe returns a dict with, at minimum, a `value` and a `source` key. The
`source` is the path or command the value came from. A number without its
provenance is not evidence, so the report format refuses to carry one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
import pdb
import json
import os

# ---------------------------------------------------------------------------
# Small helpers. These are given to students; the exercise is the probes.
# ---------------------------------------------------------------------------


def read_text(root: Path, rel: str) -> str | None:
    """Read `root/rel`, returning None if it is missing or unreadable.

    Missing is a normal outcome here, not an error: a devkit with no NVMe
    genuinely has no /sys/block/nvme0n1, and the report needs to say so rather
    than crash.
    """
    p = Path(root) / rel.lstrip("/")
    try:
        return p.read_text(errors="replace").strip("\x00").strip()
    except (OSError, UnicodeDecodeError):
        return None


def run(cmd: list[str]) -> str | None:
    """Run a command, returning stdout, or None if it is absent or fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def unknown(source: str, why: str) -> dict[str, Any]:
    """The value this lab returns when it cannot determine something.

    Note what this is not: it is not None threaded through the report, and it
    is not a plausible default. It is an explicit record that the probe ran and
    failed, carrying the reason. Assignment 1's rubric gives credit for these.
    """
    return {"value": None, "source": source, "status": "unknown", "detail": why}

# LnkSta/LnkCap lines look like:
#   LnkSta: Speed 8GT/s, Width x4, TrErr- Train- SlotClk+ DLActive- ...
#   LnkCap: Port #0, Speed 16GT/s, Width x4, ASPM L1, Exit Latency L1 <64us
_SPEED_RE = re.compile(r"Speed\s+([\d.]+)GT/s")
_WIDTH_RE = re.compile(r"Width\s+x(\d+)")

# PCIe generation by per-lane transfer rate. Gen3 is 8 GT/s; the Orin Nano
# devkit's M.2 Key-M slot is wired Gen3 x4, so a Gen4 drive reporting 16 GT/s
# capability and 8 GT/s status is behaving correctly, not underperforming.
#
# Keyed by float, not by the string lspci printed. Keying by string means
# deciding whether "8", "8.0" and "08" are the same rate, and the obvious
# normalisation — stripping trailing zeros and dots — silently turns 20 into 2.
_GEN_BY_GTS = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}


def _parse_link_line(line: str) -> dict[str, Any]:
    speed = _SPEED_RE.search(line)
    width = _WIDTH_RE.search(line)
    gts = float(speed.group(1)) if speed else None
    return {
        "raw": line.strip(),
        "gts": gts,
        "width": int(width.group(1)) if width else None,
        "gen": _GEN_BY_GTS.get(gts) if gts is not None else None,
    }

def generate_interpretation_string(neg_speed, cap_speed, negotiated, capability):
    if cap_speed > neg_speed:
        interpretation = (
            f"drive capable of Gen{capability['gen']}, link running at "
            f"Gen{negotiated['gen']} — expected on this carrier board, "
            "whose M.2 Key-M slot is wired Gen3 x4"
        )
    else:
        interpretation = (
            f"link running at its full capability, Gen{negotiated['gen']} "
            f"x{negotiated['width']}"
        )
    return interpretation
    


# ---------------------------------------------------------------------------
# The probes.
# ---------------------------------------------------------------------------

## An example code.
def probe_module_model(root: Path = Path("/")) -> dict[str, Any]:
    """Which board is this?

    The device tree model string is the most trustworthy identity on a Jetson —
    it comes from the hardware description the bootloader handed the kernel,
    not from anything installed afterwards.
    """

    # step 1: Read the raw null-terminated text from /proc/device-tree/model
    src = "/proc/device-tree/model"
    raw = read_text(root, src)

    # if unable to read, return an empty dictionary by calling unknown().
    if not raw:
        return unknown(src, "device tree model node absent — not a Jetson, or /proc not mounted")
    
    # step 2: strip null bytes and whitespace from raw string
    raw = raw.rstrip("\x00").strip()
    return {"value": raw, "source": src, "status": "ok"}

# todo by students
def probe_memory_total_kb(root: Path = Path("/")) -> dict[str, Any]:
    """How much memory is there, in kB, as the kernel counts it?

    This will read a little under 8 GB on an 8 GB board. That gap is not a
    fault: the carveout for the GPU and other hardware is taken before Linux
    ever sees the pool. Students are expected to notice and to explain it in
    their report rather than round it up.
    """

    src = "/proc/meminfo"
    raw = read_text(root, src)

    # if not able to read, return an empty dictionary by calling unknown().
    if not raw:
        return unknown(src, "Device memory distribution absent.")

    match = re.search("^MemTotal:\s+(\d+)\s*kB", raw)
    
    return {"value": int(match.group(1)), "source": src, "status": "ok"}


def probe_root_source(root: Path = Path("/")) -> dict[str, Any]:
    """What device is the root filesystem actually mounted from?

    This is the probe the lab is built around. A unit that boots from the SD
    card works, boots, and passes every casual inspection — and then runs the
    semester's benchmarks against a card an order of magnitude slower than the
    NVMe sitting unused in the slot. The failure is silent, which is exactly
    why it has to be a command rather than an assumption.

    /proc/mounts is preferred over `findmnt` because it needs no external
    binary and no elevation, and because it is what findmnt reads anyway.
    """

    src = "/proc/mounts"
    raw = read_text(root, src)

    if not raw:
        return unknown(src, "no root mount entry found in mount table")

    type = "other"
    name = "other"
    
    for line in raw.splitlines():
        for word in line.split():

            # making sure that the first word is long enough to evaluate
            if len(word) < 7:
                break # only need to evaluate the first word of each line
            
            if word[0:7] == "/dev/nv":
                type = "nvme"
                name = word[5:]
                break
            elif word[0:7] == "/dev/sd" or word[0:7] == "/dev/mm":
                type = "ssd"
                name = word[5:]
                break
            else:
                break # if first word length > 7 but neither nvme or ssd, just skip to next line
    
    return {"value": name, "kind": type, "source": src, "status": "ok"}


def probe_nvme_present(root: Path = Path("/")) -> dict[str, Any]:
    """Is there an NVMe device visible as a block device at all?

    Deliberately separate from probe_root_source. A machine can have an NVMe
    fitted and still boot from the SD card, and telling those two states apart
    is what lets the troubleshooting tree in the lab guide send a student to
    the right branch.
    """

    src = "/sys/block/nvme0n1/"
    if os.path.exists(src):
        raw = read_text(os.path.join(src, "device/model"), "")
        match = re.search("\s+(.+)", raw)
        model = match.group(1).strip("\x00").strip()
    else:
        return unknown("src", "/sys/block/nvme0n1 does not exist")
    
    return {
        "value": os.path.exists(src),
        "model": model,
        "source": "/proc/device-tree/model",
        "status": "ok",
    }


def probe_pcie_link(root: Path = Path("/"), lspci_output: str | None = None) -> dict[str, Any]:
    """What did the PCIe link negotiate, and what was it capable of?

    Two numbers, not one. The gap between them is the lab's worked example of
    spec sheet against measured reality: a Gen4 drive in a Gen3 slot advertises
    16 GT/s and settles at 8 GT/s, and a student who reports only the second
    number has recorded a fact without recording what it means.

    `lspci_output` exists so the tests can drive this without root or hardware.
    In normal use it is None and the probe shells out.
    """

    output = run(["lspci", "-vv"])
    lnkcap = ""
    lnksta = ""
    
    for line in output.split("\n"):
        if "LnkCap:" in line:
            lnkcap = line
        
        if "LnkSta:" in line:
            lnksta = line

    parsed_linkcap = _parse_link_line(lnkcap)
    parsed_linksta = _parse_link_line(lnksta)
        
    return {
        "value": parsed_linksta["gts"],
        "negotiated": parsed_linksta,
        "capability": parsed_linkcap,
        "interpretation": generate_interpretation_string(parsed_linksta["gts"], parsed_linkcap["gts"], parsed_linksta, parsed_linkcap),
        "source": "/proc/device-tree/model",
        "status": "ok",
    }


def probe_thermal_zones(root: Path = Path("/")) -> dict[str, Any]:
    """Every thermal zone the kernel exposes, in degrees C.

    Sysfs reports millidegrees. The division by 1000 is the entire trap: a
    report claiming the board idles at 43,000 degrees has been submitted more
    than once, and it is a good, cheap lesson in reading units before reading
    numbers.
    """

    # Need to get a list of all subdirectories under /sys/class/thermal/thermal_zone*/
    # After getting that list (0-8), read temp file and a type file

    temps = []
    zone_type = []
    subdirectories = []

    base = Path("/sys/class/thermal")
    for sub in base.glob("thermal_zone*"):
        subdirectories.append(str(sub))
    subdirectories.sort()

    for sub in subdirectories:
        temps.append(int(float(read_text(root, os.path.join(sub, "temp"))) / 1000))
        zone_type.append(read_text(root, os.path.join(sub, "type")))

    
    return {
        "value": max(temps),
        "zones": zone_type,
        "source": "/sys/class/thermal/thermal_zone*/temp",
        "status": "ok",
    }


def probe_power_mode(root: Path = Path("/"), nvpmodel_output: str | None = None) -> dict[str, Any]:
    """Which nvpmodel power mode is active?

    Recorded on every artifact this course produces. Lecture 01 slide 24 is
    the argument for why: two students reporting different throughput for the
    same model are usually reporting different power modes, and without this
    field there is no way to find that out after the fact.
    """

    output = run(["nvpmodel", "-q"])
    if not output:
        return unknown("nvpmodel -q", "power mode not found")

    match = re.match("NV Power Mode:\s*(.+).", output)
    mode_name = match.group(1)

    id_match = re.match("^\s*(\d+)\s*$", mode_name)
    mode_id = id_match.group(1)

    if mode_id == None:
        unknown("nvpmodel -q", "integer mode_id not found")

    return {
        "mode_id": mode_id,
        "source": "nvpmodel -q",
        "status": "ok",
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
#     out = probe_power_mode()
#     print(out)

# for generating system_report.json
if __name__ == "__main__":
    report = {
        "module_model": probe_module_model(),
        "memory_total_kb": probe_memory_total_kb(),
        "root_source": probe_root_source(),
        "nvme_present": probe_nvme_present(),
        "pcie_link": probe_pcie_link(),
        "thermal_zones": probe_thermal_zones(),
        "power_mode": probe_power_mode()
    }
    
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

