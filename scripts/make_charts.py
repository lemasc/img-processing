"""Generate the report charts and diagrams from results/data/*.csv into results/charts/.

Usage: uv run scripts/make_charts.py
"""
import csv
from pathlib import Path

import matplotlib
import matplotlib.font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
OUT = ROOT / "results" / "charts"

# Same Thai font as the report; matplotlib does not always scan ~/.local/share/fonts.
for f in Path.home().glob(".local/share/fonts/**/THSarabun*.ttf"):
    matplotlib.font_manager.fontManager.addfont(str(f))

plt.rcParams.update({
    "font.family": ["TH SarabunPSK", "DejaVu Sans"],
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Okabe-Ito, colour-blind safe
BLUE, ORANGE, GREEN, VERMILION, SKY, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#666666"

GPU_BW = 331.0     # GB/s, measured copy kernel
GPU_BW_TH = 360.0  # GB/s, theoretical
GPU_FP32 = 12.7e3  # GFLOP/s, theoretical
CPU_BW = 51.4      # GB/s, STREAM triad on 12 threads
CPU_BW_TH = 76.8   # GB/s, DDR5-4800 dual channel
CPU_FP32 = 6 * 4.6 * 32  # GFLOP/s: 6 P-cores x 4.6 GHz x 2 FMA ports x 8 lanes x 2 FLOP
COLOR_FLOP = 35    # FLOP per pixel of the colour adjustment (see chapter 3)
P_CORES, TWELVE, SIXTEEN = "0,2,4,6,8,10", "0-11", "0-15"


def load(name):
    """Rows of a results CSV as dicts. Fields 0-2 and the last 7 are fixed; the middle is params."""
    rows = []
    for r in list(csv.reader(open(DATA / f"{name}.csv")))[1:]:
        n = len(r)
        kv = dict(p.split("=", 1) for p in ",".join(r[3:n - 7]).split(";") if "=" in p)
        rows.append({"kernel": r[0], "platform": r[1], "mp": float(r[2]), "p": kv, "ms": float(r[n - 6])})
    return rows


def pick(rows, kernel, platform, **want):
    """Median ms of the last matching row (later runs override earlier ones, as in the report)."""
    m = [r for r in rows if r["kernel"] == kernel and r["platform"] == platform
         and all(r["p"].get(k) == v for k, v in want.items())]
    return m[-1]["ms"] if m else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- colour adjustment

SIZES = (1, 6, 24, 48)


def color_rows(mp):
    return [r for r in load("color") if r["p"].get("img") == f"photo_{mp}mp.png"]


def chart_color_breakdown():
    parts = (("H2D (คัดลอกไป GPU)", "cuda_h2d", ORANGE), ("Kernel", "cuda_kernel", BLUE), ("D2H (คัดลอกกลับ)", "cuda_d2h", VERMILION))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bottom = [0.0] * len(SIZES)
    totals = []
    for label, plat, col in parts:
        vals = [pick(color_rows(s), "color", plat) for s in SIZES]
        ax.bar([f"{s} MP" for s in SIZES], vals, bottom=bottom, label=label, color=col, width=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, s in enumerate(SIZES):
        rows = color_rows(s)
        total = pick(rows, "color", "cuda_e2e")
        kern = pick(rows, "color", "cuda_kernel")
        ax.text(i, total + 0.4, f"{total:.1f} ms\nKernel {100 * kern / total:.0f}%", ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("เวลา (ms)")
    ax.set_xlabel("ขนาดภาพ")
    ax.set_ylim(0, max(bottom) * 1.22)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    save(fig, "color_breakdown.png")


def chart_color_speedup():
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    k, t = [], []
    for s in SIZES:
        rows = color_rows(s)
        base = pick(rows, "color", "cpu_scalar")
        k.append(base / pick(rows, "color", "cuda_kernel"))
        t.append(base / pick(rows, "color", "cuda_e2e"))
    x = list(range(len(SIZES)))
    ax.plot(x, k, "o-", color=BLUE, lw=2, label="เฉพาะ Kernel")
    ax.plot(x, t, "s-", color=VERMILION, lw=2, label="เวลาทั้งหมด (รวมคัดลอก)")
    ax.axhline(1, color=GREY, ls="--", lw=1)
    ax.text(len(SIZES) - 1, 1.1, "เท่ากับ CPU", ha="right", va="bottom", color=GREY, fontsize=13)
    for xi, a, b in zip(x, k, t):
        ax.annotate(f"{a:.0f}×", (xi, a), textcoords="offset points", xytext=(0, 8), ha="center", color=BLUE, fontsize=13)
        ax.annotate(f"{b:.1f}×", (xi, b), textcoords="offset points", xytext=(0, -17), ha="center", color=VERMILION, fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(0.7, max(k) * 2.2)
    ax.set_xticks(x, [f"{s} MP" for s in SIZES])
    ax.set_xlabel("ขนาดภาพ")
    ax.set_ylabel("อัตราเร่งเทียบ CPU แบบ Scalar (เท่า)")
    ax.legend(loc="center right")
    save(fig, "color_speedup.png")


# ---------------------------------------------------------------- blur

KS = (7, 15, 31, 63, 127)


def radius_label(k):
    return f"{(k - 1) // 2}\n(k={k})"


def blur_kernel_ms(rows, variant, k):
    # the small-k sweeps were pinned to core 0; the radius sweep (k >= 15 and separable) ran unpinned
    for cpus in ("0", "unpinned"):
        ms = pick(rows, "blur_gauss", "cuda_kernel", img="photo_24mp.png", k=str(k), v=variant, block="16x16", cpus=cpus)
        if ms is not None and (cpus == "0" or ms < 5000):
            return ms


def chart_radius():
    rows = load("blur_conv")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for v, label, col, mk in (("naive", "Naive", VERMILION, "o"), ("tiled", "Tiled", ORANGE, "s"), ("sep", "Separable", BLUE, "^")):
        pts = [(k, blur_kernel_ms(rows, v, k)) for k in KS]
        pts = [(k, ms) for k, ms in pts if ms is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", color=col, lw=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(KS, [radius_label(k) for k in KS])
    ax.minorticks_off()
    ax.set_xlabel("รัศมี r (ขนาดเคอร์เนล k)")
    ax.set_ylabel("เวลา Kernel (ms)")
    ax.legend(loc="upper left")
    save(fig, "blur_radius.png")


# ---------------------------------------------------------------- streams

def chart_streams():
    rows = load("streams")
    jobs = (("color", "การปรับสี", BLUE, "o"), ("blur_sep_k15", "เบลอ k=15", GREEN, "s"),
            ("blur_sep_k63", "เบลอ k=63", ORANGE, "^"), ("blur_sep_k127", "เบลอ k=127", VERMILION, "D"))
    chunks = (1, 2, 4, 8, 16)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for kern, label, col, mk in jobs:
        ys = [pick(rows, kern, f"gpu_pinned_n{n}", img="photo_24mp.png") for n in chunks]
        ax.plot(range(len(chunks)), ys, mk + "-", color=col, lw=2, label=label)
    floor = 144 / 15.3  # 144 MB over the 15.3 GB/s measured with both directions busy
    ax.axhline(floor, color=GREY, ls="--", lw=1.2)
    ax.text(len(chunks) - 1, floor - 0.8, f"ขั้นต่ำของการคัดลอก {floor:.1f} ms (15.3 GB/s)", ha="right", va="top", color=GREY, fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks(range(len(chunks)), [str(n) for n in chunks])
    ax.set_yticks([5, 10, 20, 40], ["5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_ylim(4, 80)
    ax.set_xlabel("จำนวน Chunk (Pinned Memory)")
    ax.set_ylabel("เวลาทั้งหมด (ms)")
    ax.legend(loc="upper left", ncol=2)
    save(fig, "streams_chunks.png")


# ---------------------------------------------------------------- ladders

def best(rows, kernel, platform, all_modes=False, **want):
    modes = ("0", P_CORES, TWELVE, SIXTEEN) if all_modes else (P_CORES, TWELVE, SIXTEEN)
    vals = [pick(rows, kernel, platform, cpus=c, **want) for c in modes]
    return min(v for v in vals if v is not None)


def ladder_levels(job, k=None):
    """(label, ms, kind) from scalar CPU up to CUDA with Pinned and Streams, for 24 MP."""
    lad = load("ladder")
    st = load("streams")
    img = {"img": "photo_24mp.png"}
    if k is not None:
        img["k"] = str(k)
    one = lambda plat: pick(lad, job, plat, cpus="0", **img)
    at = lambda plat, c: pick(lad, job, plat, cpus=c, **img)
    if job == "color":
        skern, chunks = "color", [pick(st, "color", f"gpu_pinned_n{n}", img="photo_24mp.png") for n in (4, 8, 16)]
    else:
        skern, chunks = f"blur_sep_k{k}", [pick(st, f"blur_sep_k{k}", f"gpu_pinned_n{n}", img="photo_24mp.png") for n in (4, 8, 16)]
    return [
        ("Scalar", one("r1_scalar"), "cpu"),
        ("Autovec AVX2, 1 Thread", one("r2_autovec_avx2"), "cpu"),
        ("AVX2 Intrinsics, 1 Thread", one("r6a_avx2_intrin"), "cpu"),
        ("AVX2 + OpenMP, 6 P-core", at("r6_avx2_intrin_omp", P_CORES), "cpu"),
        ("AVX2 + OpenMP, 12 Thread", at("r6_avx2_intrin_omp", TWELVE), "cpu"),
        ("Autovec + OpenMP, 12 Thread", at("r2b_autovec_avx2_omp", TWELVE), "cpu"),
        ("CUDA เฉพาะ Kernel", one("r7_cuda_kernel"), "kern"),
        ("CUDA ทั้งหมด", one("r7_cuda_e2e"), "gpu"),
        ("CUDA Pinned + Streams", min(chunks), "gpu"),
    ]


KIND_COLOR = {"cpu": BLUE, "kern": SKY, "gpu": GREEN}


def draw_ladder(ax, levels, title=None, labels=True):
    ys = list(range(len(levels)))[::-1]
    for y, (name, ms, kind) in zip(ys, levels):
        ax.barh(y, ms, color=KIND_COLOR[kind], height=0.68)
        ax.text(ms * 1.12, y, f"{ms:.3g}" if ms < 100 else f"{ms:.0f}", va="center", fontsize=13)
    ax.set_xscale("log")
    ax.set_yticks(ys, [l[0] for l in levels] if labels else [""] * len(levels))
    if not labels:
        ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(l[1] for l in levels) / 2, max(l[1] for l in levels) * 6)
    ax.set_xlabel("เวลา (ms, สเกล Log)")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title)


def legend_handles():
    from matplotlib.patches import Patch
    return [Patch(color=BLUE, label="CPU"), Patch(color=SKY, label="GPU เฉพาะ Kernel"), Patch(color=GREEN, label="GPU รวมการคัดลอก")]


def chart_ladder_color():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    draw_ladder(ax, ladder_levels("color"))
    ax.legend(handles=legend_handles(), loc="lower right")
    save(fig, "ladder_color.png")


def chart_ladder_both():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"wspace": 0.04})
    draw_ladder(a, ladder_levels("color"), "การปรับสี (24 MP)")
    draw_ladder(b, ladder_levels("blur_sep", k=15), "การเบลอ Separable k = 15 (24 MP)", labels=False)
    b.legend(handles=legend_handles(), loc="lower right")
    save(fig, "ladder_both.png")


# ---------------------------------------------------------------- roofline

def chart_roofline():
    import numpy as np
    conv, lad = load("blur_conv"), load("ladder")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.logspace(-1, 3, 200)
    for bw, bw_th, peak, col, name in ((GPU_BW, GPU_BW_TH, GPU_FP32, GREEN, "GPU"), (CPU_BW, CPU_BW_TH, CPU_FP32, BLUE, "CPU")):
        ax.plot(x, np.minimum(peak, bw * x), color=col, lw=2.2)
        ax.plot(x, np.minimum(peak, bw_th * x), color=col, lw=1, ls="--", alpha=0.7)
    ax.text(0.12, GPU_BW * 0.12 * 1.35, f"GPU {GPU_BW:.0f} GB/s", color=GREEN, rotation=34, fontsize=13)
    ax.text(0.12, CPU_BW * 0.12 * 0.5, f"CPU {CPU_BW:.1f} GB/s", color=BLUE, rotation=34, fontsize=13)
    ax.text(900, GPU_FP32 * 1.08, f"GPU FP32 {GPU_FP32 / 1e3:.1f} TFLOP/s", color=GREEN, ha="right", va="bottom", fontsize=13)
    ax.text(900, CPU_FP32 * 0.92, f"CPU FP32 {CPU_FP32:.0f} GFLOP/s", color=BLUE, ha="right", va="top", fontsize=13)

    mp = 24.0e6
    gflops = lambda flop_px, ms: flop_px * mp / (ms * 1e-3) / 1e9
    # GPU kernels (24 MP photo, 16x16 blocks)
    colour_ms = pick(color_rows(24), "color", "cuda_kernel")
    ax.plot(COLOR_FLOP / 6, gflops(COLOR_FLOP, colour_ms), "*", color=PURPLE, ms=15, mec="k", mew=0.5, label="การปรับสี (GPU)", zorder=5)
    for v, label, col, mk, ks in (("naive", "Naive (GPU)", VERMILION, "o", (3, 5, 7, 11, 15, 31, 63, 127)),
                                  ("tiled", "Tiled (GPU)", ORANGE, "s", (3, 5, 7, 11, 15, 31, 63))):
        pts = [(7 * k * k / 6, gflops(7 * k * k, blur_kernel_ms(conv, v, k))) for k in ks if blur_kernel_ms(conv, v, k)]
        ax.plot(*zip(*pts), mk + "-", color=col, ms=5, lw=1, label=label, zorder=4)
    pts = [(12 * k / 30, gflops(12 * k, blur_kernel_ms(conv, "sep", k))) for k in KS]
    ax.plot(*zip(*pts), "^-", color=BLUE if False else SKY, ms=7, lw=1.2, mec="k", mew=0.4, label="Separable (GPU)", zorder=4)
    # CPU AVX2 + OpenMP
    cpu_color = best(lad, "color", "r6_avx2_intrin_omp", img="photo_24mp.png")
    ax.plot(COLOR_FLOP / 6, gflops(COLOR_FLOP, cpu_color), "*", color="white", mec=PURPLE, mew=1.8, ms=15, label="การปรับสี (CPU)", zorder=5)
    pts = [(12 * k / 30, gflops(12 * k, best(lad, "blur_sep", "r6_avx2_intrin_omp", img="photo_24mp.png", k=str(k)))) for k in KS]
    ax.plot(*zip(*pts), "v-", color="white", mec=BLUE, mew=1.5, ms=7, lw=1, label="Separable (CPU AVX2 + OpenMP)", zorder=4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(5, 4e4)
    ax.set_xlabel("ความเข้มข้นของการคำนวณ (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.legend(loc="lower right", fontsize=12)
    save(fig, "roofline.png")


# ---------------------------------------------------------------- diagrams

def box(ax, x, y, w, h, text, fc="#eef3f8", ec=BLUE, fs=13):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, p, q, text=None, fs=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.3, color="#333"))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.05, (p[1] + q[1]) / 2, text, fontsize=fs, ha="left", va="center", color=GREY)


def diagram_dev_flow():
    steps = [
        ("1", "เตรียมเครื่อง\nและ Smoke Test"), ("2", "ปรับสีพื้นฐาน\nCPU + CUDA"), ("3", "ชุดวัดเวลา\n(CSV)"),
        ("4", "ตรวจความถูกต้อง\nCPU เทียบ GPU"), ("5", "เบลอ Naive"), ("6", "เบลอ Tiled\n+ Profiler"),
        ("7", "เบลอ Separable\nรัศมีใหญ่"), ("8", "Pinned Memory\n+ Streams"), ("9", "บันไดการประมวลผล\nแบบขนาน + Roofline"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.set_xlim(0, 3.3)
    ax.set_ylim(-0.1, 3.3)
    ax.axis("off")
    w, h = 0.98, 0.7
    pos = []
    for i, (n, t) in enumerate(steps):
        row, col = divmod(i, 3)
        if row % 2 == 1:
            col = 2 - col
        pos.append((0.55 + col * 1.1, 2.9 - row * 1.15))
    for i, ((n, t), (x, y)) in enumerate(zip(steps, pos)):
        fc = "#fdf1dc" if i in (4, 5, 6) else ("#e4f3ec" if i >= 7 else "#eef3f8")
        ec = ORANGE if i in (4, 5, 6) else (GREEN if i >= 7 else BLUE)
        box(ax, x, y, w, h, f"{n}. {t}", fc=fc, ec=ec, fs=12)
    for (x1, y1), (x2, y2) in zip(pos, pos[1:]):
        if abs(y1 - y2) < 1e-6:
            d = 1 if x2 > x1 else -1
            arrow(ax, (x1 + d * w / 2 + d * 0.01, y1), (x2 - d * w / 2 - d * 0.01, y2))
        else:
            arrow(ax, (x1, y1 - h / 2 - 0.01), (x2, y2 + h / 2 + 0.01))
    save(fig, "dev_flow.png")


def diagram_compile_flow():
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.set_xlim(0, 4.4)
    ax.set_ylim(-1.05, 4.4)
    ax.axis("off")
    box(ax, 0.75, 4.05, 1.4, 0.55, "CMakeLists.txt", fs=12)
    box(ax, 2.2, 4.05, 1.0, 0.55, "cmake", fc="#fdf1dc", ec=ORANGE, fs=12)
    box(ax, 3.6, 4.05, 1.4, 0.55, "build/\nbuild.ninja", fs=12)
    arrow(ax, (1.45, 4.05), (1.7, 4.05))
    arrow(ax, (2.7, 4.05), (2.9, 4.05))
    arrow(ax, (3.6, 3.75), (3.6, 3.3), "ninja")
    box(ax, 0.9, 3.0, 1.5, 0.55, "src/*.cu", fs=12)
    box(ax, 3.4, 3.0, 1.5, 0.55, "src/*.cpp", fs=12)
    box(ax, 0.9, 2.1, 1.5, 0.85, "nvcc\nDevice: PTX/SASS sm_86\nHost: g++", fc="#fdf1dc", ec=ORANGE, fs=11)
    box(ax, 3.4, 2.1, 1.5, 0.85, "g++\n-O2 / AVX2 / FMA\n/ OpenMP", fc="#fdf1dc", ec=ORANGE, fs=11)
    arrow(ax, (0.9, 2.72), (0.9, 2.53))
    arrow(ax, (3.4, 2.72), (3.4, 2.53))
    box(ax, 2.15, 1.1, 1.6, 0.55, "ไฟล์ Object (.o)", fs=12)
    arrow(ax, (1.2, 1.67), (1.9, 1.4))
    arrow(ax, (3.1, 1.67), (2.4, 1.4))
    box(ax, 0.95, 0.25, 1.7, 0.55, "ar → ไลบรารี .a", fc="#fdf1dc", ec=ORANGE, fs=12)
    arrow(ax, (1.9, 0.83), (1.4, 0.55))
    box(ax, 3.3, 0.25, 1.4, 0.55, "apps/*.cu", fs=12)
    box(ax, 2.1, -0.6, 1.8, 0.55, "nvcc (ลิงก์)", fc="#fdf1dc", ec=ORANGE, fs=12)
    arrow(ax, (1.0, -0.05), (1.7, -0.33))
    arrow(ax, (3.3, -0.05), (2.5, -0.33))
    box(ax, 3.85, -0.6, 0.95, 0.55, "โปรแกรม\nใน build/", fc="#e4f3ec", ec=GREEN, fs=11)
    arrow(ax, (3.0, -0.6), (3.37, -0.6))
    save(fig, "compile_flow.png")


def main():
    chart_color_breakdown()
    chart_color_speedup()
    chart_radius()
    chart_streams()
    chart_ladder_color()
    chart_ladder_both()
    chart_roofline()
    diagram_dev_flow()
    diagram_compile_flow()


if __name__ == "__main__":
    main()
