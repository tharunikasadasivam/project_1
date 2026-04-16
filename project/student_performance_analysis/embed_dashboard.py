"""
embed_dashboard.py
Embeds all chart PNGs as base64 into a single self-contained HTML file
that can be shared with anyone — no server or extra files needed.
"""

import base64, os, re

CHARTS_DIR = "output/charts"
TEMPLATE   = "output/dashboard.html"
OUTPUT     = "output/dashboard_shareable.html"

# Read original HTML
with open(TEMPLATE, "r", encoding="utf-8") as f:
    html = f.read()

# Find all src="charts/..." references and embed them
def embed_image(match):
    src = match.group(1)
    img_path = os.path.join("output", src)
    if os.path.exists(img_path):
        with open(img_path, "rb") as img_file:
            b64 = base64.b64encode(img_file.read()).decode("utf-8")
        return f'src="data:image/png;base64,{b64}"'
    return match.group(0)

html = re.sub(r'src="(charts/[^"]+\.png)"', embed_image, html)

# Also remove the Google Fonts network request (embed fallback)
html = html.replace(
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(OUTPUT) / 1024
print(f"OK  Self-contained dashboard saved!")
print(f"    Path : {OUTPUT}")
print(f"    Size : {size_kb:.1f} KB")
print(f"\nShare this single file with anyone - it works in any browser.")
