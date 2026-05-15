"""Generate the scida README banner.

Composes a 1600×500 banner from horizontal stripes of four sim projections
(SIMBA → THESAN → FLAMINGO → TNG100), overlays a flat cream "scida" title
with a soft halo for legibility, and writes it to docs/images/banner.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "docs" / "images"
OUT_PATH = IMG_DIR / "banner.png"

# Stripe order is brightness-graded top → bottom; SIMBA's warm fire anchors
# the eye at the top, TNG100's dark green grounds the bottom.
SIM_ORDER = ["simba", "thesan", "flamingo", "tng100"]
SIMS = {
    "flamingo": IMG_DIR / "projection_FLAMINGO_Density.jpg",
    "simba": IMG_DIR / "projection_SIMBA_Temperature.jpg",
    "thesan": IMG_DIR / "projection_THESAN_NeutralHydrogenAbundance.jpg",
    "tng100": IMG_DIR / "projection_TNG100_GFM_Metallicity.jpg",
}

BANNER = (1600, 500)
TITLE = "scida"
SUBTITLE = "scalable analysis of large scientific datasets"

CREAM = (245, 240, 216)
DARK = (16, 28, 56)

# The source projections carry a thin pale border on their right edge that
# reads as a bright stripe once composited; trim it off.
RIGHT_TRIM = 0.03

TITLE_SIZE = 280
SUB_SIZE = 34
Y_NUDGE = -65
SUB_GAP = 17
HALO_ALPHA = 200
HALO_BLUR = 28
TRACKING = 2
SATURATION = 0.85

FONT_BOLD = Path(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")))
FONT_REG = Path(fm.findfont(fm.FontProperties(family="DejaVu Sans")))


def _load(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img.crop((0, 0, int(round(w * (1 - RIGHT_TRIM))), h))


def _cover_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    iw, ih = img.size
    scale = max(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _stripes() -> Image.Image:
    n = len(SIM_ORDER)
    stripe_h = BANNER[1] // n
    canvas = Image.new("RGB", BANNER, DARK)
    y = 0
    for k in SIM_ORDER:
        canvas.paste(_cover_crop(_load(SIMS[k]), (BANNER[0], stripe_h)), (0, y))
        y += stripe_h
    if y < BANNER[1]:
        last = canvas.crop((0, y - stripe_h, BANNER[0], y))
        canvas.paste(last.resize((BANNER[0], BANNER[1] - y)), (0, y))

    d = ImageDraw.Draw(canvas)
    for i in range(1, n):
        sy = i * stripe_h
        d.line([(0, sy), (BANNER[0], sy)], fill=DARK, width=2)

    return ImageEnhance.Color(canvas).enhance(SATURATION)


def _draw_tracked(
    draw: ImageDraw.ImageDraw,
    cx: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: int,
) -> None:
    widths = [draw.textlength(c, font=font) for c in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x = cx - total / 2
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=font, fill=fill)
        x += w + tracking


def main() -> None:
    bg = _stripes().convert("RGBA")
    bg = Image.alpha_composite(bg, Image.new("RGBA", BANNER, (8, 14, 32, 80)))

    title_font = ImageFont.truetype(str(FONT_BOLD), TITLE_SIZE)
    sub_font = ImageFont.truetype(str(FONT_REG), SUB_SIZE)

    measure = ImageDraw.Draw(bg)
    tw, th = measure.textbbox((0, 0), TITLE, font=title_font)[2:]
    tx = (BANNER[0] - tw) // 2
    ty = (BANNER[1] - th) // 2 + Y_NUDGE
    sub_y = ty + th + SUB_GAP
    sw = measure.textlength(SUBTITLE, font=sub_font)
    sub_x = (BANNER[0] - sw) // 2

    halo = Image.new("RGBA", BANNER, (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(halo)
    hdraw.text((tx, ty), TITLE, font=title_font, fill=(0, 0, 0, HALO_ALPHA))
    hdraw.text((sub_x, sub_y), SUBTITLE, font=sub_font, fill=(0, 0, 0, HALO_ALPHA))
    bg = Image.alpha_composite(bg, halo.filter(ImageFilter.GaussianBlur(HALO_BLUR)))

    draw = ImageDraw.Draw(bg)
    draw.text((tx, ty), TITLE, font=title_font, fill=CREAM + (255,))
    _draw_tracked(
        draw,
        BANNER[0] // 2,
        sub_y,
        SUBTITLE,
        sub_font,
        fill=(230, 226, 208, 235),
        tracking=TRACKING,
    )

    bg.convert("RGB").save(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
