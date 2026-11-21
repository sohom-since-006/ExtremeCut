"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config, framing
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "shadow": 0, "pop": 112, "upper": True, "per": 3, "anim": "pop"},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "shadow": 0, "pop": 114, "upper": True, "per": 2, "anim": "pop"},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "shadow": 0, "pop": 106, "upper": False, "per": 4, "anim": "smooth"},
    "hormozi": {"label": "Hormozi", "font": "Poppins ExtraBold", "size": 138, "base": "FFFFFF", "active": "00FF44",
                "strong": "00D7FF", "outline": 8, "shadow": 4, "pop": 122, "upper": True, "per": 2, "anim": "tilt"},
    "mrbeast": {"label": "MrBeast", "font": "Archivo Black", "size": 146, "base": "FFFFFF", "active": "00E0FD",
                "strong": "0044FF", "outline": 9, "shadow": 5, "pop": 124, "upper": True, "per": 2, "anim": "bounce"},
    "neon": {"label": "Cyber Neon", "font": "Anton", "size": 146, "base": "F0FF00", "active": "FF33EE",
             "strong": "00FFFF", "outline": 7, "shadow": 6, "pop": 118, "upper": True, "per": 2, "anim": "glow"},
    "kinetic": {"label": "Kinetic Tilt", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
                "strong": "FFFFFF", "outline": 7, "shadow": 3, "pop": 120, "upper": True, "per": 3, "anim": "kinetic"},
    "metallic": {"label": "Extreme Cuts", "font": "Poppins ExtraBold", "size": 134, "base": "F4F5EA", "active": "DCE38B",
                 "strong": "8BE3DC", "outline": 7, "shadow": 3, "pop": 115, "upper": True, "per": 3, "anim": "pulse"},
    "cinematic": {"label": "Cinematic", "font": "Poppins ExtraBold", "size": 98, "base": "FFFFFF", "active": "8BE3DC",
                  "strong": "FFFFFF", "outline": 4, "shadow": 2, "pop": 106, "upper": False, "per": 4, "anim": "smooth"},
}
HOOK = {"font": "Poppins ExtraBold", "size": 96, "ink": "14100E", "box": "FFFFFF", "pad_x": 36, "pad_y": 22, "radius": 26,
        "max_w": 900}
FONT_FILES = {"Poppins ExtraBold": "Poppins-ExtraBold.ttf", "Anton": "Anton-Regular.ttf", "Archivo Black": "ArchivoBlack-Regular.ttf"}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


LAYOUTS = {
    "blur": "Full frame + blurred fill",
    "zoom": "Zoomed in + blurred fill",
    "crop": "Fill screen (auto-reframe / centre)",
    "stack": "Face top + Gameplay bottom",
    "bottom": "Gameplay top + Face bottom",
    "game": "Gameplay only",
    "pip_tl": "Top-left picture-in-picture",
    "pip_tr": "Top-right picture-in-picture",
    "pip_bl": "Bottom-left picture-in-picture",
    "pip_br": "Bottom-right picture-in-picture",
    "circle": "Circle camera overlay",
    "side": "Side-by-side split",
    "trio": "3-layer Gamer stack (Cam + Game + HUD)",
}


def _ts(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def geometry(src_w, src_h, layout):
    """Where the video sits in the 1080x1920 frame, and where hook and captions go."""
    if src_h >= src_w and layout not in ("crop", "fill"):
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout == "blur":
        fg_h = min(H, int(round(W * src_h / src_w / 2)) * 2)
        top = (H - fg_h) // 2
        return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)
    if layout == "zoom":
        fg_h = 1000
        top = (H - fg_h) // 2
        return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)
    if layout == "stack":
        return layout, H, 280, 1360
    if layout == "bottom":
        return layout, H, 260, 1080
    if layout in ("pip_tl", "pip_tr", "pip_bl", "pip_br", "circle"):
        return layout, H, 280, 1380
    if layout in ("side", "trio"):
        return layout, H, 260, 1380
    # default crop / fill / game
    return layout, H, 300, 1380


def _hook_events(hook, dur, y):
    """The heading: dark text in a white rounded box that fits it, on one line or two balanced ones."""
    f, size = _Font.get(HOOK["font"]), HOOK["size"]
    words = hook.split()
    if not words:
        return []
    lines = [" ".join(words)]
    if f.width(lines[0], size) > HOOK["max_w"] and len(words) > 1:      # two lines, split where they come out most even
        cut = min(range(1, len(words)), key=lambda i: max(f.width(" ".join(words[:i]), size), f.width(" ".join(words[i:]), size)))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    widest = max(f.width(l, size) for l in lines)
    if widest > HOOK["max_w"]:
        size = max(40, int(size * HOOK["max_w"] / widest))
        widest = max(f.width(l, size) for l in lines)
    line_h = f.em(size) * 1.16
    bw, bh = widest + 2 * HOOK["pad_x"], line_h * len(lines) + 2 * HOOK["pad_y"]
    x0, y0, r = (W - bw) / 2, y - bh / 2, min(HOOK["radius"], bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(dur)}"
    ev = [f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,0)\\c&H{HOOK['box']}&\\p1}}{box}{{\\p0}}"]
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2 + i * line_h - f.cap_shift(size)
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{cy:.0f})\\fad(160,0)\\fs{size}}}{line}")
    return ev


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    words = [dict(w, s=w["s"] - CAPTION_LEAD, e=w["e"] - CAPTION_LEAD) for w in words]
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{st['outline']},{st.get('shadow', 0)},5,0,0,0,1
Style: Hook,{HOOK['font']},{HOOK['size']},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y)
    f = _Font.get(st["font"])
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = [w for w in inside if w["w"]]
    groups = _groups(inside, st["per"])
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        size = st["size"]
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        if total > CAP_MAX_W:                                  # a long phrase: shrink it to fit, never run off the frame
            size = max(36, int(size * CAP_MAX_W / total))
            gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]   # a space, plus room for both outlines
            total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
        cy = cap_y - f.cap_shift(size)
        x = (W - total) / 2
        p = st["pop"]
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            anim = st.get("anim", "pop")
            if anim == "tilt":
                active_anim = (f"\\c&H{st['active']}&\\fscx92\\fscy92"
                               f"\\t(0,80,\\fscx{p}\\fscy{p}\\frz-3)\\t(80,180,\\fscx{p - 5}\\fscy{p - 5}\\frz0)")
            elif anim == "kinetic":
                active_anim = (f"\\c&H{st['active']}&\\fscx90\\fscy90"
                               f"\\t(0,70,\\fscx{p}\\fscy{p}\\frz-4)\\t(70,140,\\fscx{p - 3}\\fscy{p - 3}\\frz2)\\t(140,210,\\fscx{p - 5}\\fscy{p - 5}\\frz0)")
            elif anim == "bounce":
                active_anim = (f"\\c&H{st['active']}&\\fscx86\\fscy86"
                               f"\\t(0,70,\\fscx{p + 5}\\fscy{p + 5})\\t(70,150,\\fscx{p - 8}\\fscy{p - 8})\\t(150,220,\\fscx{p - 3}\\fscy{p - 3})")
            elif anim == "glow":
                active_anim = (f"\\c&H{st['active']}&\\3c&H00330033&"
                               f"\\t(0,80,\\3c&H00{st['active']}&\\fscx{p}\\fscy{p})\\t(80,200,\\3c&H00330033&\\fscx{p - 4}\\fscy{p - 4})")
            elif anim == "pulse":
                active_anim = (f"\\c&H{st['active']}&\\3c&H0010434A&\\fscx95\\fscy95"
                               f"\\t(0,90,\\fscx{p}\\fscy{p}\\3c&H0045B8B0&)\\t(90,200,\\fscx{p - 4}\\fscy{p - 4}\\3c&H0010434A&)")
            elif anim == "smooth":
                active_anim = f"\\c&H{st['active']}&\\alpha&H80&\\t(0,100,\\alpha&H00&\\fscx{p}\\fscy{p})"
            else:
                active_anim = (f"\\c&H{st['active']}&\\fscx94\\fscy94"
                               f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})")

            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and motion graphics animation
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}{active_anim}}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


def _filter(layout, fg_h=H, src_w=1920, src_h=1080, clip=None, src=None, words=None, start=0.0, end=0.0):
    clip = clip or {}
    words = words or []
    fps = config.OUT_FPS
    top_ratio = max(0.20, min(0.50, float(clip.get("top", 0.32))))
    top_h = int(H * top_ratio)
    top_h -= (top_h % 2)
    bot_h = H - top_h

    # Bounding boxes
    cam = clip.get("cam")
    game = clip.get("game")
    hud = clip.get("hud")

    if not cam and layout in ("stack", "bottom", "pip_tl", "pip_tr", "pip_bl", "pip_br", "circle", "side", "trio"):
        if src and Path(src).exists():
            try:
                cam = framing.find_facecam_box(Path(src), src_w, src_h, start, end)
            except Exception:
                cam = None
        if not cam:
            cam = {"x": 0.0, "y": 0.0, "w": 0.35, "h": 0.40}

    cx, cy, cw, ch = framing.to_pixels(cam, src_w, src_h)
    gx, gy, gw, gh = framing.to_pixels(game, src_w, src_h)
    hx, hy, hw, hh = framing.to_pixels(hud or {"x": 0.0, "y": 0.65, "w": 1.0, "h": 0.35}, src_w, src_h)

    v_in = f"[0:v]setpts=PTS-STARTPTS,fps={fps}[v_in];"
    if layout == "crop":
        crop_x, crop_y, crop_w, crop_h = framing.calculate_auto_reframe_crop(src_w, src_h, cam)
        base_filter = v_in + f"[v_in]crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={W}:{H},setsar=1[v_base];"
    elif layout == "game":
        base_filter = v_in + f"[v_in]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v_base];"
    elif layout == "stack":
        base_filter = v_in + (
            f"[v_in]split=2[s_cam][s_game];"
            f"[s_cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[l_top];"
            f"[s_game]crop={gw}:{gh}:{gx}:{gy},scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1[l_bot];"
            f"[l_top][l_bot]vstack=inputs=2,drawbox=x=0:y={top_h-1}:w={W}:h=3:color=#167D7F@0.9:t=fill,setsar=1[v_base];"
        )
    elif layout == "bottom":
        base_filter = v_in + (
            f"[v_in]split=2[s_game][s_cam];"
            f"[s_game]crop={gw}:{gh}:{gx}:{gy},scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1[l_top];"
            f"[s_cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[l_bot];"
            f"[l_top][l_bot]vstack=inputs=2,drawbox=x=0:y={bot_h-1}:w={W}:h=3:color=#167D7F@0.9:t=fill,setsar=1[v_base];"
        )
    elif layout.startswith("pip_"):
        pw = int(W * top_ratio * 1.15)
        ph = int(pw * 0.75)
        pw -= (pw % 2)
        ph -= (ph % 2)
        if layout == "pip_tl":
            px, py = 36, 80
        elif layout == "pip_tr":
            px, py = W - pw - 36, 80
        elif layout == "pip_bl":
            px, py = 36, H - ph - 260
        else:  # pip_br
            px, py = W - pw - 36, H - ph - 260
        base_filter = v_in + (
            f"[v_in]split=2[s_base][s_pip];"
            f"[s_base]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg];"
            f"[s_pip]crop={cw}:{ch}:{cx}:{cy},scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},"
            f"drawbox=x=0:y=0:w={pw}:h={ph}:color=#167D7F@0.95:t=4,setsar=1[pip];"
            f"[bg][pip]overlay={px}:{py},setsar=1[v_base];"
        )
    elif layout == "circle":
        dia = int(W * 0.38)
        dia -= (dia % 2)
        cx_pos, cy_pos = (W - dia) // 2, 80
        base_filter = v_in + (
            f"[v_in]split=2[s_base][s_circ];"
            f"[s_base]crop={gw}:{gh}:{gx}:{gy},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg];"
            f"[s_circ]crop={cw}:{ch}:{cx}:{cy},scale={dia}:{dia}:force_original_aspect_ratio=increase,crop={dia}:{dia},"
            f"format=rgba,geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':a='if(lte(pow(X-W/2,2)+pow(Y-H/2,2),pow(min(W,H)/2,2)),255,0)'[circ];"
            f"[bg][circ]overlay={cx_pos}:{cy_pos},setsar=1[v_base];"
        )
    elif layout == "side":
        lw = W // 2
        rw = W - lw
        base_filter = v_in + (
            f"[v_in]split=2[s_left][s_right];"
            f"[s_left]crop={cw}:{ch}:{cx}:{cy},scale={lw}:{H}:force_original_aspect_ratio=increase,crop={lw}:{H},setsar=1[l];"
            f"[s_right]crop={gw}:{gh}:{gx}:{gy},scale={rw}:{H}:force_original_aspect_ratio=increase,crop={rw}:{H},setsar=1[r];"
            f"[l][r]hstack=inputs=2,drawbox=x={lw-1}:y=0:w=3:h={H}:color=#167D7F@0.9:t=fill,setsar=1[v_base];"
        )
    elif layout == "trio":
        base_filter = v_in + (
            f"[v_in]split=3[s_cam][s_act][s_hud];"
            f"[s_cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:480:force_original_aspect_ratio=increase,crop={W}:480,setsar=1[l_cam];"
            f"[s_act]crop={gw}:{gh}:{gx}:{gy},scale={W}:960:force_original_aspect_ratio=increase,crop={W}:960,setsar=1[l_act];"
            f"[s_hud]crop={hw}:{hh}:{hx}:{hy},scale={W}:480:force_original_aspect_ratio=increase,crop={W}:480,setsar=1[l_hud];"
            f"[l_cam][l_act][l_hud]vstack=inputs=3,drawbox=x=0:y=479:w={W}:h=3:color=#167D7F@0.9:t=fill,"
            f"drawbox=x=0:y=1439:w={W}:h=3:color=#167D7F@0.9:t=fill,setsar=1[v_base];"
        )
    elif layout == "zoom":
        base_filter = v_in + (
            f"[v_in]split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:1000:force_original_aspect_ratio=increase,crop={W}:1000[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v_base];"
        )
    else:  # default blur
        base_filter = v_in + (
            f"[v_in]split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v_base];"
        )

    zooms = framing.detect_retention_zooms(words, start, end) if clip.get("smart_zoom") else []
    zoom_flt = framing.make_zoom_filter(zooms) if zooms else ""
    if zoom_flt:
        return base_filter + f"[v_base]{zoom_flt}[v0];"
    return base_filter.rstrip(";").replace("[v_base]", "[v0]") + ";"


_qsv_ok = None


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, clip.get("layout", "blur"))
    show_hook = clip.get("show_hook", True)
    hook_val = "" if (show_hook is False or clip.get("no_hook")) else clip.get("hook", "")
    ass = build_ass(words, start, end, hook_val, clip.get("style", "bounce"),
                    clip.get("highlight_words", []), hook_y, cap_y)
    # The clip is made in a short work folder and moved into place afterwards. The caption step of ffmpeg cannot
    # open files whose full path is longer than 260 characters on Windows, which a data folder deep inside other
    # folders easily reaches. Inside the work folder every name is short and relative (a drive colon would also
    # break the filter syntax).
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        return _render_in(work, src, out_dir, name, start, end, layout, fg_h, ass, src_w, src_h, clip, words)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, start, end, layout, fg_h, ass: str,
               src_w: int = 1920, src_h: int = 1080, clip: dict = None, words: list = None) -> dict:
    global _qsv_ok
    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)
    fc = _filter(layout, fg_h, src_w, src_h, clip=clip, src=src, words=words, start=start, end=end) + f"[v0]ass={name}.ass:fontsdir=f[v]"
    dur = end - start
    real_src = Path(src).resolve()
    if str(src).lower().endswith(".ts") and real_src.is_file():
        mp4_tmp = work / f"{name}_src.mp4"
        try:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(real_src),
                 "-c", "copy", "-movflags", "+faststart", mp4_tmp.name], cwd=work)
            if mp4_tmp.is_file() and mp4_tmp.stat().st_size > 0:
                real_src = Path(mp4_tmp.name)
        except Exception:
            pass
    av_offset = float((clip or {}).get("av_offset") or 0.0)
    if av_offset != 0.0:
        v_start = max(0.0, start + av_offset)
        a_start = max(0.0, start)
        seek_args = [
            "-ss", f"{v_start:.2f}", "-t", f"{dur:.2f}", "-i", str(real_src),
            "-ss", f"{a_start:.2f}", "-t", f"{dur:.2f}", "-i", str(real_src)
        ]
        a_map = "1:a:0"
    else:
        seek_args = ["-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", str(real_src)]
        a_map = "0:a:0"
    af = "loudnorm=I=-14:TP=-1.5:LRA=11"
    base = ["ffmpeg", "-y", "-v", "error"] + seek_args + [
        "-filter_complex", fc, "-map", "[v]", "-map", a_map, "-af", af,
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    encoders = []
    if _qsv_ok is not False:
        encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
    encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    err = None
    for label, enc in encoders:
        p = subprocess.run(base + enc + [f"{name}.mp4"], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=1800)
        if p.returncode == 0:
            if label == "qsv":
                _qsv_ok = True
            break
        err = p.stderr[-1200:]
        if label == "qsv":
            _qsv_ok = False
    else:
        raise RuntimeError(f"Render failed: {err}")
    cover_t = min(1.2, (end - start) / 2)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
         "-q:v", "3", f"{name}.jpg"], cwd=work)
    try:
        src_t = max(0.0, (start - (clip or {}).get("src_offset", 0.0)) + cover_t + av_offset)
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{src_t:.2f}", "-i", str(real_src), "-frames:v", "1",
             "-vf", "scale='min(1280,iw)':-2", "-q:v", "3", f"{name}_src.jpg"], cwd=work)
    except Exception:
        pass
    for ext in ("mp4", "jpg", "ass"):
        p = work / f"{name}.{ext}"
        if p.is_file():
            shutil.move(str(p), str(out_dir / f"{name}.{ext}"))
    p_src = work / f"{name}_src.jpg"
    if p_src.is_file():
        shutil.move(str(p_src), str(out_dir / f"{name}_src.jpg"))
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout}
