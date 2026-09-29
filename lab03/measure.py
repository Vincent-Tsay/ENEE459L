from __future__ import annotations

import statistics
import math
import re
import os

from typing import Any
from pathlib import Path

from bench import Bench, measured, read_first, read_text, unknown

import json

# A sample is still warm-up while it exceeds the settled rate by this fraction.
WARMUP_TOL = 0.5

# How many samples must sit strictly above a quantile before that quantile is an
# estimate rather than "the biggest number we saw, wearing a hat".
MIN_SAMPLES_ABOVE = 5

# Percentiles the record carries, in the order the schema lists them.
PERCENTILES = (50, 95, 99)

# The widest gap between neighbouring measurements, as a multiple of the typical
# gap, beyond which the sample is treated as coming from two populations.
MULTIMODAL_GAP_RATIO = 20.0

# Neither side of that gap is a mode unless it holds at least this fraction.
MIN_MODE_FRACTION = 0.10

# Below this many retained samples, modality is not a question worth answering.
MIN_SAMPLES_FOR_MODALITY = 20

# How far the last third of a run may drift from the first third, relative to
# the run's own median, before the run is not one population either.
STATIONARITY_TOL = 0.10
MIN_SAMPLES_FOR_STATIONARITY = 12

THERMAL_ZONES = "sys/devices/virtual/thermal"

POWER_RAIL_CANDIDATES = (
    "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
    "sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input",
    "sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
)

GPU_LOAD_CANDIDATES = (
    "sys/devices/platform/gpu.0/load",
    "sys/devices/gpu.0/load",
)

CPUFREQ_MIN = "sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq"
CPUFREQ_MAX = "sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"


# ===========================================================================
# 1. The loop
# ===========================================================================
def run_timed_iterations(bench: Bench, repeats: int = 100) -> list[float]:

    bench.workload.synchronize()
    iterations = []

    for i in range(repeats):
        start = bench.clock()
        bench.workload.run()
        bench.workload.synchronize()
        end = bench.clock()
        iterations.append((end-start)/1000000.0)

    # List of elapsed times in milliseconds
    return iterations


def find_warmup_boundary(samples: list[float]) -> dict[str, Any]:
    
    if len(samples) < 4:
        return unknown("Finding warm up boundary", "There are too few samples")

    settled_ss = statistics.median(samples[int((len(samples)/2)):])

    if settled_ss <= 0:
        return unknown("Finding settled steady state median", "Settled steady state median is <= 0")

    threshold = settled_ss * (1 + WARMUP_TOL)
    discard = 0

    for run in samples:
        if run > threshold:
            discard = discard + 1
        else:
            break
    
    return {
        "discard": discard,
        "source": "leading prefix above (1 + 0.5) x median of the run's second half",
        "status": "ok",
        "settled_rate_ms": round(settled_ss, 4),
        "threshold": round(threshold, 4),
        "tolerance": WARMUP_TOL,
        "retained": len(samples) - discard
    }


def summarize(samples: list[float]) -> dict[str, Any]:
    
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "p99": None
        }

    n = len(samples)    
    samples.sort()
    mean = statistics.fmean(samples)
    minimum = samples[0]
    maximum = samples[n-1]

    if n > 2:
        std = statistics.stdev(samples)
    else:
        std = 0.00

    # getting 50th percentile
    h = (n-1)*(50.0/100)
    i = math.floor(h)
    p50 = samples[i] + (h-i)*(samples[i+1] - samples[i])

    # getting 95th percentile
    h = (n-1)*(95.0/100)
    i = math.floor(h)
    p95 = samples[i] + (h-i)*(samples[i+1] - samples[i])

    # getting 99th percentile
    h = (n-1)*(99.0/100)
    i = math.floor(h)
    p99 = samples[i] + (h-i)*(samples[i+1] - samples[i])

    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(minimum, 4),
        "max": round(maximum, 4),
        "p50": round(p50, 4),
        "p95": round(p95, 4),
        "p99": round(p99, 4)
    }


def is_multimodal(samples: list[float]) -> dict[str, Any]:
    n = len(samples)
    
    if n < MIN_SAMPLES_FOR_MODALITY:
        return unknown("checking samples for multimodal", "not enough samples")

    samples.sort()

    trimmed = samples[math.floor(0.05*n):math.floor(0.95*n)]
    removed_left_samples = len(samples[0:math.floor(0.05*n)])
    removed_right_samples = n - len(trimmed) - removed_left_samples
    gaps = []
    
    # starts counting from index 1 and looks one element behind to find diff
    for i in range(1, len(trimmed)-1):
        gaps.append(trimmed[i] - trimmed[i-1])
    
    median_gap = statistics.median(gaps)

    if median_gap <= 0:
        return unknown("finding multimodality from median_gap calculation", "median gap <= 0, timer resolution is too coarse")

    widest_gap = max(gaps)
    ratio = widest_gap/median_gap

    # gap[i] stores the difference between trimmed[i+1] - trimmed[i]
    # so all samples to left of split_point would be trimmed[0:i+1]
    # all samples to right of split-point would be trimmed[i+1:]
    split_point = gaps.index(widest_gap)
    left_samples = split_point+1
    right_samples = len(trimmed) - left_samples

    if ratio >= MULTIMODAL_GAP_RATIO and float(left_samples + removed_left_samples)/n >= MIN_MODE_FRACTION and float(right_samples + removed_right_samples)/n >= MIN_MODE_FRACTION:
        value = True
    else:
        value = False
    
    left_subgroup = {
        "n": left_samples + removed_left_samples,
        "share": round(float(left_samples + removed_left_samples)/n, 4),
        "median_ms": round(statistics.median(samples[0:left_samples+removed_left_samples]), 4)
    }

    right_subgroup = {
        "n": right_samples,
        "share": round(float(right_samples + removed_right_samples)/n, 4),
        "median_ms": round(statistics.median(samples[left_samples+removed_left_samples:]), 4)
    }

    modes = [left_subgroup, right_subgroup]

    return measured(value, "widest trimmed gap >= 20.0x the median gap, with >= 10% of samples on each side", gap_ratio = round(ratio, 4), widest_gap_ms = round(widest_gap, 4), typical_gap_ms = round(median_gap, 4), modes = modes)
    

# ===========================================================================
# 7. The clock ceiling the run happened under
# ===========================================================================


def probe_power_state(bench: Bench) -> dict[str, Any]:
    output = bench.runner(["nvpmodel", "-q"])

    if output.ok == False or output.returncode != 0:
        return("run nvpmodel -q to get power state", output.error)

    match = re.match("NV Power Mode:\s*(.+)\n*(\d*)", output.stdout)
    mode_name = match.group(1)
    mode_index = int(match.group(2))

    min_freq = read_text("", CPUFREQ_MIN)
    max_freq = read_text("", CPUFREQ_MAX)

    if min_freq == None or max_freq == None:
        jetson_clocks = None
    elif int(min_freq) == int(max_freq):
        jetson_clocks = True
    else:
        jetson_clocks = False

    jetson_clocks_source = {
        "value": f"scaling_min_freq={min_freq}, scaling_max_freq={max_freq}",
        "source": f"{CPUFREQ_MIN} vs. {CPUFREQ_MAX}",
        "status": "ok"
    }

    return {
        "value": mode_name,
        "source": "nvpmodel -q",
        "status": "ok",
        "mode_index": mode_index,
        "jetson_clocks": jetson_clocks,
        "jetson_clocks_source": jetson_clocks_source
    }



def probe_telemetry(bench: Bench) -> dict[str, Any]:

    root = "/"
    temps = []
    zone_type = []
    subdirectories = []

    base = Path("/sys/class/thermal")
    for sub in base.glob("thermal_zone*"):
        subdirectories.append(str(sub))
    subdirectories.sort()

    for sub in subdirectories:
        try:
            temps.append(float(read_text(root, os.path.join(sub, "temp"))) / 1000)
            zone_type.append(read_text(root, os.path.join(sub, "type")))
        except TypeError:
            continue

    temperature_c = {
        "value": round(max(temps), 4),
        "source": "sys/devices/virtual/thermal/*/temp",
        "status": "ok",
        "zone": zone_type[temps.index(max(temps))],
        "zones_read": len(zone_type)
    }

    output =  read_first("", POWER_RAIL_CANDIDATES)

    if not output:
        power_mw = unknown("sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input | sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input | sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input", "none of the documented INA3221 rail paths could be read")
    else:
        power_mw = {
            "value": int(output[1]),
            "source": str(output[0]),
            "status": "ok",
        }

    output = read_first("", GPU_LOAD_CANDIDATES)

    if not output:
        gpu_utilization_percent = unknown("sys/devices/platform/gpu.0/load | sys/devices/gpu.0/load", "none of the documented GPU paths could be read")
    else:
        gpu_utilization_percent = {
            "value": int(output[1])/10.0,
            "source": str(output[0]),
            "status": "ok",
            "units": "per-mille / 10.0"
        }

    return {
        "temperature_c": temperature_c,
        "power_mw": power_mw,
        "gpu_utilization_percent": gpu_utilization_percent
    }

## for debugging - uncomment the following lines for debugging.
# if __name__ == "__main__":
    # env = Bench.real()
    # out = find_warmup_boundary(samples)
    # print(out)

# for generating system_report.json
if __name__ == "__main__":
    # calling base environment
    env = Bench.real()

    # get your samples
    samples = run_timed_iterations(env, repeats=100)

    # testing measurments and probes
    report = {
        "warmup_boundary": find_warmup_boundary(samples),
        "summarize_setup": summarize(samples),
        "is_multimodal": is_multimodal(samples),
        "probe_power_state": probe_power_state(env),
        "probe_telemetry": probe_telemetry(env),
    }

    # save samples
    path = "samples_analysis.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)

    # save report
    path = "system_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)