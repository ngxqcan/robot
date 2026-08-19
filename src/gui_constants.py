import customtkinter as ctk

# Cyber Theme Palette
NEON = "#ff1744"          # Accent Red/Pink
NEON_GREEN = "#00e676"    # Active / Good / Connected
NEON_CYAN = "#00e5ff"     # Accent Blue / Highlights
NEON_ORANGE = "#ff6022"   # Target Bounding Box
BG_DARK = "#0c0d0e"       # Main background
BG_CARD = "#141519"       # Card panel background
BG_CARD_HOVER = "#1c1d24" # Card hover background
BORDER_COLOR = "#22252e"  # Subtle card border
TEXT_MUTED = "#8e929b"    # Secondary text
TEXT_LIGHT = "#e4e7eb"    # Primary text

# Backward compatibility
BG = BG_DARK

def neon_button(*args, **kwargs):
    fg = kwargs.pop("fg_color", NEON)
    hover = kwargs.pop("hover_color", "#d50000")
    text_color = kwargs.pop("text_color", "#ffffff")
    corner_radius = kwargs.pop("corner_radius", 6)
    font = kwargs.pop("font", ("Segoe UI", 12, "bold"))
    return ctk.CTkButton(*args, fg_color=fg, hover_color=hover, text_color=text_color, corner_radius=corner_radius, font=font, **kwargs)

def cyber_button(*args, **kwargs):
    fg = kwargs.pop("fg_color", "#1e2029")
    hover = kwargs.pop("hover_color", "#2a2d3a")
    text_color = kwargs.pop("text_color", TEXT_LIGHT)
    corner_radius = kwargs.pop("corner_radius", 6)
    font = kwargs.pop("font", ("Segoe UI", 12, "bold"))
    return ctk.CTkButton(*args, fg_color=fg, hover_color=hover, text_color=text_color, corner_radius=corner_radius, font=font, **kwargs)

def green_button(*args, **kwargs):
    fg = kwargs.pop("fg_color", "#00c853")
    hover = kwargs.pop("hover_color", "#009624")
    text_color = kwargs.pop("text_color", "#ffffff")
    corner_radius = kwargs.pop("corner_radius", 6)
    font = kwargs.pop("font", ("Segoe UI", 12, "bold"))
    return ctk.CTkButton(*args, fg_color=fg, hover_color=hover, text_color=text_color, corner_radius=corner_radius, font=font, **kwargs)
