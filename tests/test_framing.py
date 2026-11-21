"""Tests for the framing subsystem: 11+ layout geometries, auto-reframe, face tracking, and retention zooms.
Usage: python tests/test_framing.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio import config, framing, render

fails = []


def check(name, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   {extra}" if extra else ""))
    if not ok:
        fails.append(name)


def run_tests():
    print("\n=== framing & layout geometries ===")

    # 1. Layout coverage
    expected_layouts = {
        "blur", "zoom", "crop", "stack", "bottom", "game",
        "pip_tl", "pip_tr", "pip_bl", "pip_br", "circle", "side", "trio"
    }
    check("all 13 layout specs are registered", expected_layouts.issubset(set(render.LAYOUTS.keys())),
          f"Found: {set(render.LAYOUTS.keys())}")

    # 2. Geometry calculations
    for lay in render.LAYOUTS:
        geom = render.geometry(1920, 1080, lay)
        check(f"geometry returns 4-tuple for {lay}", len(geom) == 4 and geom[0] in render.LAYOUTS)
        check(f"safe margins for {lay}", 100 <= geom[2] <= 550 and 900 <= geom[3] <= 1800,
              f"hook_y={geom[2]}, cap_y={geom[3]}")

    # 3. Coordinate normalization and even-parity
    b1 = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    px, py, pw, ph = framing.to_pixels(b1, 1920, 1080)
    check("to_pixels within frame", 0 <= px < 1920 and 0 <= py < 1080 and pw > 0 and ph > 0)
    check("to_pixels even parity (h264 requirement)", px % 2 == 0 and py % 2 == 0 and pw % 2 == 0 and ph % 2 == 0)

    # 4. Auto-reframe crop calculation
    face_box = {"x": 0.2, "y": 0.1, "w": 0.25, "h": 0.35}
    rx, ry, rw, rh = framing.calculate_auto_reframe_crop(1920, 1080, face_box)
    check("auto-reframe 9:16 aspect", abs(rw / rh - 9 / 16) < 0.05, f"w={rw}, h={rh}")
    check("auto-reframe tracks face to the left", rx < (1920 - rw) // 2, f"rx={rx}")
    check("auto-reframe even parity", rx % 2 == 0 and ry % 2 == 0 and rw % 2 == 0 and rh % 2 == 0)

    # 5. Dynamic retention zoom detection
    sample_words = [
        {"s": 1.0, "e": 1.4, "w": "Welcome"},
        {"s": 1.5, "e": 1.9, "w": "everyone"},
        {"s": 5.0, "e": 5.4, "w": "INSANE!"},
        {"s": 12.0, "e": 12.5, "w": "unbelievable..."}
    ]
    zooms = framing.detect_retention_zooms(sample_words, 0.0, 20.0)
    check("detects retention zooms from punctuation", len(zooms) >= 1, str(zooms))
    zf = framing.make_zoom_filter(zooms)
    check("generates valid zoom filter expression", "crop=w=iw/if(between(t" in zf and "scale=1080:1920" in zf)

    # 6. Filter compilation across all 13 layouts
    for lay in render.LAYOUTS:
        geom = render.geometry(1920, 1080, lay)
        flt = render._filter(lay, geom[1], 1920, 1080, clip={"smart_zoom": True}, words=sample_words, start=0.0, end=20.0)
        check(f"filter compiles with v0 output for {lay}", "[v0];" in flt)

    # 7. Real FFmpeg hardware/software filtergraph test on synthetic stream
    test_lays = ["stack", "bottom", "pip_tr", "circle", "side", "trio"]
    for lay in test_lays:
        geom = render.geometry(1920, 1080, lay)
        fc = render._filter(lay, geom[1], 1920, 1080, clip={"top": 0.35, "smart_zoom": True},
                            words=sample_words, start=0.0, end=2.0)
        cmd = [
            "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=1920x1080:rate=30",
            "-filter_complex", fc + "[v0]null[outv]", "-map", "[outv]", "-c:v", "libx264", "-preset", "ultrafast",
            "-f", "null", "-"
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        check(f"ffmpeg renders synthetic frame for {lay}", p.returncode == 0, p.stderr[-200:] if p.returncode != 0 else "")

    # 8. All 9 motion graphics caption styles build valid ASS events
    expected_styles = {"bounce", "hype", "clean", "hormozi", "mrbeast", "neon", "kinetic", "metallic", "cinematic"}
    check("all 9 caption styles registered", expected_styles.issubset(set(render.STYLES.keys())),
          f"Found: {set(render.STYLES.keys())}")
    for st_name in render.STYLES:
        ass_str = render.build_ass(sample_words, 0.0, 15.0, "MOTION GRAPHICS HOOK", st_name, ["insane"], 280, 1380)
        check(f"build_ass produces valid dialogue for {st_name}", "Dialogue: 2," in ass_str and "Style: Cap" in ass_str)

    if fails:
        print(f"\n{len(fails)} checks failed!")
        return 1
    print("\nALL FRAMING CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(run_tests())
