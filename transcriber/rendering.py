"""Engrave a music21 Score to PDF and to a preview PNG.

No MuseScore/LilyPond install required: verovio (a pip-installable, self
contained notation engraving engine) renders MusicXML to SVG, svglib +
reportlab convert each page's SVG to a PDF page, and PyMuPDF (also self
contained, bundles its own renderer) rasterizes the first page for an
in-app preview.

verovio's font/resource loading is not thread-safe — calling it from any
thread other than a process's main thread reliably fails ("Bravura font
could not be loaded"), even with the resource path set explicitly. So the
actual rendering (`_render_musicxml_file`, invoked via this module's
`__main__` block) always runs in its own subprocess, which has its own real
main thread, regardless of which thread in the parent process calls
render_score().
"""

import io
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass

from music21 import stream
from PIL import Image, ImageChops

_VEROVIO_OPTIONS = {
    "pageWidth": 2100,
    "pageHeight": 2970,
    "pageMarginTop": 100,
    "pageMarginBottom": 100,
    "pageMarginLeft": 100,
    "pageMarginRight": 100,
    "scale": 40,
    "adjustPageHeight": False,
    "breaks": "auto",
    "footer": "none",
}


@dataclass
class RenderedScore:
    pdf_path: str
    page_count: int
    preview_png: bytes  # first page, cropped to content, for an in-app preview


def _crop_to_content(png_bytes: bytes, margin: int = 70) -> bytes:
    """Trim the mostly-blank page down to its actual notation, so a short
    transcription doesn't render as a tiny sliver on an otherwise-empty
    preview panel."""
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    diff = ImageChops.difference(image, background)
    bbox = diff.getbbox()
    if bbox is None:
        return png_bytes

    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(image.width, right + margin)
    bottom = min(image.height, bottom + margin)

    cropped = image.crop((left, top, right, bottom))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_musicxml_file(musicxml_path: str, pdf_path: str, preview_png_path: str,
                           preview_dpi: int) -> int:
    """Must run on a process's main thread. Returns the page count."""
    import fitz
    import verovio
    from reportlab.graphics import renderPDF
    from svglib.svglib import svg2rlg

    toolkit = verovio.toolkit()
    toolkit.setOptions(_VEROVIO_OPTIONS)
    if not toolkit.loadFile(musicxml_path):
        raise ValueError("Verovio could not read the generated score.")
    page_count = toolkit.getPageCount()
    if page_count < 1:
        raise ValueError("Nothing to render (no notes in the transcription).")

    with tempfile.TemporaryDirectory() as tmpdir:
        merged = fitz.open()
        for page_number in range(1, page_count + 1):
            svg_path = f"{tmpdir}/page{page_number}.svg"
            with open(svg_path, "w") as svg_file:
                svg_file.write(toolkit.renderToSVG(page_number))

            page_pdf_path = f"{tmpdir}/page{page_number}.pdf"
            drawing = svg2rlg(svg_path)
            renderPDF.drawToFile(drawing, page_pdf_path)

            with fitz.open(page_pdf_path) as page_doc:
                merged.insert_pdf(page_doc)

        merged.save(pdf_path)

        first_page_pixmap = merged[0].get_pixmap(dpi=preview_dpi)
        preview_png = _crop_to_content(first_page_pixmap.tobytes("png"))
        merged.close()

    with open(preview_png_path, "wb") as preview_file:
        preview_file.write(preview_png)

    return page_count


def render_score(score: stream.Score, pdf_path: str, preview_dpi: int = 220) -> RenderedScore:
    """Safe to call from any thread: the actual verovio rendering runs in a
    subprocess so it always gets a genuine main thread."""
    with tempfile.TemporaryDirectory() as tmpdir:
        musicxml_path = f"{tmpdir}/score.musicxml"
        score.write("musicxml", fp=musicxml_path)
        preview_png_path = f"{tmpdir}/preview.png"

        completed = subprocess.run(
            [sys.executable, __file__, musicxml_path, pdf_path, preview_png_path, str(preview_dpi)],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise ValueError(f"Rendering failed: {completed.stderr.strip() or completed.stdout.strip()}")

        page_count = json.loads(completed.stdout)["page_count"]
        with open(preview_png_path, "rb") as preview_file:
            preview_png = preview_file.read()

    return RenderedScore(pdf_path=pdf_path, page_count=page_count, preview_png=preview_png)


if __name__ == "__main__":
    _musicxml_path, _pdf_path, _preview_png_path, _preview_dpi = sys.argv[1:5]
    _page_count = _render_musicxml_file(_musicxml_path, _pdf_path, _preview_png_path, int(_preview_dpi))
    print(json.dumps({"page_count": _page_count}))
