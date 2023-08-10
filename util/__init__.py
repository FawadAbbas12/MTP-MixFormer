from .annotations import Color, BaseAnnotator, TextAnnotator, Detection, Rect

white = Color.from_hex_string("#FFFFFF")
red = Color.from_hex_string("#850101")
green = Color.from_hex_string("#00D4BB")
yollow = Color.from_hex_string("#FFFF00")
COLORS = [
    white,
    red,
    green,
    yollow,
    white,
    red,
    green,
    yollow,
    white,
    red,
    green,
    yollow
]
THICKNESS = 4