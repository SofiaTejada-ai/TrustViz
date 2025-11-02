# trustviz/ingest/pdf_extract.py
from __future__ import annotations
from typing import List, Tuple, Optional
import io, os, base64, tempfile, httpx
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF

@dataclass
class FigureBlob:
    page: int
    image_b64: str     # base64 PNG
    caption: Optional[str] = None

async def _download_pdf(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as cx:
        r = await cx.get(url)
        r.raise_for_status()
        return r.content

def _png_b64(img: Image.Image) -> str:
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return base64.b64encode(bio.getvalue()).decode("ascii")

def _guess_caption(page_text: str) -> Optional[str]:
    # naive heuristic: first “Figure” sentence on the page
    for line in page_text.splitlines():
        if line.strip().lower().startswith(("figure", "fig.")):
            return line.strip()[:400]
    return None

async def extract_figures_from_pdf(pdf_url: str, max_images: int = 6) -> List[FigureBlob]:
    try:
        pdf_bytes = await _download_pdf(pdf_url)
    except Exception:
        return []
    figs: List[FigureBlob] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for pno in range(len(doc)):
            if len(figs) >= max_images: break
            page = doc[pno]
            page_text = page.get_text("text") or ""
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 4:  # grayscale / RGB
                    pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                else:          # has alpha
                    pix0 = fitz.Pixmap(fitz.csRGB, pix)
                    pil = Image.frombytes("RGB", [pix0.width, pix0.height], pix0.samples)
                    pix0 = None
                # light size filter (skip tiny icons)
                if pil.width < 180 or pil.height < 120:
                    continue
                figs.append(FigureBlob(page=pno+1, image_b64=_png_b64(pil), caption=_guess_caption(page_text)))
                if len(figs) >= max_images: break
    return figs
