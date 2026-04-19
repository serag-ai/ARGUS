import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

from my_agent.utils.mri.overlays import render_overlay_pngs
from my_agent.utils.pathology.inference import overlay_cell_heatmap
from my_agent.utils.oct.inference import overlay_oct_upper_boundaries

def create_mri_pdf_report(
    patient_id: str,
    image_path: str,
    segmentation_path: str,
    orientation: str,
    model_path: str,
    report_text: str,
    out_pdf_path: str | None = None,
) -> dict:

    if not segmentation_path or not os.path.exists(segmentation_path):
        return {"ok": False, "error": f"segmentation_path not found: {segmentation_path}"}

    if out_pdf_path is None:
        base = os.path.basename(image_path).replace(".nii.gz", "").replace(".nii", "")
        out_dir = os.path.join(os.path.dirname(image_path), "reports")
        os.makedirs(out_dir, exist_ok=True)
        out_pdf_path = os.path.join(out_dir, f"{base}_image_report.pdf")
    else:
        os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)

    asset_dir = os.path.join(os.path.dirname(out_pdf_path), "assets")
    pngs = render_overlay_pngs(image_path, segmentation_path, asset_dir, alpha=0.35)

    c = rl_canvas.Canvas(out_pdf_path, pagesize=letter)
    W, H = letter
    left = 0.8 * inch
    right = 0.8 * inch
    top = 0.8 * inch
    bottom = 0.8 * inch
    y = H - top

    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawString(left, y, "Brain MRI Segmentation Report")
    y -= 0.28 * inch

    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Patient ID: {patient_id}"); y -= 0.18 * inch
    c.drawString(left, y, f"Model: {os.path.basename(model_path)}"); y -= 0.18 * inch
    c.drawString(left, y, f"Orientation: {orientation}"); y -= 0.30 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Mid-slice views with segmentation overlay")
    y -= 0.18 * inch

    img_h = 2.2 * inch
    gap = 0.2 * inch
    usable_w = W - left - right
    img_w = (usable_w - 2 * gap) / 3.0

    x1 = left
    x2 = left + img_w + gap
    x3 = left + 2 * (img_w + gap)

    def _draw_img(path, x, y_top, w, h, label):
        c.setFont("Helvetica", 10)
        c.drawString(x, y_top, label)
        c.drawImage(ImageReader(path), x, y_top - h - 0.05 * inch,
                    width=w, height=h, preserveAspectRatio=True, anchor="n")

    _draw_img(pngs["SAG"], x1, y, img_w, img_h, "Sagittal")
    _draw_img(pngs["COR"], x2, y, img_w, img_h, "Coronal")
    _draw_img(pngs["AXI"], x3, y, img_w, img_h, "Axial")
    y -= (img_h + 0.45 * inch)

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Summary")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)
    txt = (report_text or "").strip()

    def _wrap_text(s: str, max_chars: int = 110) -> list[str]:
        if not s:
            return ["(No report text provided.)"]
        words = s.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + (1 if cur else 0) + len(w) <= max_chars:
                cur = f"{cur} {w}".strip()
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        return lines

    for line in _wrap_text(txt, max_chars=110)[:10]:
        c.drawString(left, y, line)
        y -= 0.15 * inch
        if y < bottom + 1.2 * inch:
            c.showPage()
            y = H - top
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()

    return {"ok": True, "pdf_path": out_pdf_path, "overlay_png_paths": pngs}

def create_pathology_pdf_report(
    patient_id: str,
    image_path: str,
    heatmap_path: str,
    model_path: str,
    report_text: str,
    out_pdf_path: str | None = None,
    alpha: float = 0.45,
) -> dict:

    if not image_path or not os.path.exists(image_path):
        return {"ok": False, "error": f"image_path not found: {image_path}"}

    if not heatmap_path or not os.path.exists(heatmap_path):
        return {"ok": False, "error": f"heatmap_path not found: {heatmap_path}"}

    if not model_path or not os.path.exists(model_path):
        return {"ok": False, "error": f"model_path not found: {model_path}"}

  
    # output path
    if out_pdf_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(os.path.dirname(image_path), "reports")
        os.makedirs(out_dir, exist_ok=True)
        out_pdf_path = os.path.join(out_dir, f"{base}_pathology_report.pdf")
    else:
        os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)

    #overlay
    asset_dir = os.path.join(os.path.dirname(out_pdf_path), "assets")
    os.makedirs(asset_dir, exist_ok=True)

    overlay_png_path = os.path.join(
        asset_dir,
        f"{os.path.splitext(os.path.basename(image_path))[0]}_overlay.png"
    )

    overlay_cell_heatmap(
        image_path=image_path,
        heatmap_path=heatmap_path,
        out_path=overlay_png_path,
        alpha=alpha,
    )

 
    #Build PDF

    c = rl_canvas.Canvas(out_pdf_path, pagesize=letter)
    W, H = letter

    left = 0.8 * inch
    right = 0.8 * inch
    top = 0.8 * inch
    bottom = 0.8 * inch
    y = H - top

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawString(left, y, "Pathology Cell Counting Report")
    y -= 0.30 * inch

    # Metadata
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Patient ID: {patient_id}"); y -= 0.18 * inch
    c.drawString(left, y, "Modality: Pathology"); y -= 0.18 * inch
    c.drawString(left, y, f"Model Used: {os.path.basename(model_path)}"); y -= 0.18 * inch
    c.drawString(left, y, f"Image: {os.path.basename(image_path)}"); y -= 0.28 * inch

    # Overlay Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Predicted Cell Probability Overlay")
    y -= 0.18 * inch

    img_w = W - left - right
    img_h = 4.5 * inch

    c.drawImage(
        ImageReader(overlay_png_path),
        left,
        y - img_h,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        anchor="n",
    )

    y -= (img_h + 0.35 * inch)

    # Report Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Report Summary")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)

    words = (report_text or "").strip().split()
    line = ""
    for w in words:
        if len(line) + len(w) + 1 <= 110:
            line = f"{line} {w}".strip()
        else:
            c.drawString(left, y, line)
            y -= 0.15 * inch
            line = w
            if y < bottom + 0.8 * inch:
                c.showPage()
                y = H - top
                c.setFont("Helvetica", 10)

    if line:
        c.drawString(left, y, line)

    c.showPage()
    c.save()

    return {
        "ok": True,
        "pdf_path": out_pdf_path,
        "overlay_png_path": overlay_png_path,
        "modality": "pathology",
        "model_used": model_path,
    }


def create_oct_pdf_report(
    patient_id: str,
    image_path: str,
    prob_path: str,
    model_path: str,
    report_text: str,
    out_pdf_path: str | None = None,
    n_classes: int = 11,
) -> dict:

    if not image_path or not os.path.exists(image_path):
        return {"ok": False, "error": f"image_path not found: {image_path}"}

    if not prob_path or not os.path.exists(prob_path):
        return {"ok": False, "error": f"prob_path not found: {prob_path}"}

    if not model_path or not os.path.exists(model_path):
        return {"ok": False, "error": f"model_path not found: {model_path}"}


    if out_pdf_path is None:
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_dir = os.path.join(os.path.dirname(image_path), "reports")
        os.makedirs(out_dir, exist_ok=True)
        out_pdf_path = os.path.join(out_dir, f"{base}_oct_report.pdf")
    else:
        os.makedirs(os.path.dirname(out_pdf_path), exist_ok=True)


    asset_dir = os.path.join(os.path.dirname(out_pdf_path), "assets")
    os.makedirs(asset_dir, exist_ok=True)

    overlay_png_path = os.path.join(
        asset_dir,
        f"{os.path.splitext(os.path.basename(image_path))[0]}_upperlines_overlay.png"
    )

    overlay_oct_upper_boundaries(
        image_path=image_path,
        prob_path=prob_path,
        n_classes=n_classes,
        out_path=overlay_png_path,
    )

  
    # Build PDF
    c = rl_canvas.Canvas(out_pdf_path, pagesize=letter)
    W, H = letter

    left = 0.8 * inch
    right = 0.8 * inch
    top = 0.8 * inch
    bottom = 0.8 * inch
    y = H - top

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawString(left, y, "OCT Retinal Layer Analysis Report")
    y -= 0.30 * inch

    # Metadata
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Patient ID: {patient_id}"); y -= 0.18 * inch
    c.drawString(left, y, "Modality: OCT"); y -= 0.18 * inch
    c.drawString(left, y, f"Model Used: {os.path.basename(model_path)}"); y -= 0.18 * inch
    c.drawString(left, y, f"Image: {os.path.basename(image_path)}"); y -= 0.28 * inch

    # Overlay section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Upper Retinal Layer Boundaries")
    y -= 0.18 * inch

    img_w = W - left - right
    img_h = 4.5 * inch

    c.drawImage(
        ImageReader(overlay_png_path),
        left,
        y - img_h,
        width=img_w,
        height=img_h,
        preserveAspectRatio=True,
        anchor="n",
    )

    y -= (img_h + 0.35 * inch)

    # Report text
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Report Summary")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)

    words = (report_text or "").strip().split()
    line = ""

    for w in words:
        if len(line) + len(w) + 1 <= 110:
            line = f"{line} {w}".strip()
        else:
            c.drawString(left, y, line)
            y -= 0.15 * inch
            line = w
            if y < bottom + 0.8 * inch:
                c.showPage()
                y = H - top
                c.setFont("Helvetica", 10)

    if line:
        c.drawString(left, y, line)

    c.showPage()
    c.save()

    return {
        "ok": True,
        "pdf_path": out_pdf_path,
        "overlay_png_path": overlay_png_path,
        "modality": "OCT",
        "model_used": model_path,
    }

def make_simple_pdf(
    patient_id: str,
    modality: str,
    report_text: str,
    overlay_png: str,
    out_pdf: str,
):
    """
    Page 1: title + report text
    Page 2: overlay image
    """
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

    c = rl_canvas.Canvas(out_pdf, pagesize=letter)
    W, H = letter


    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75 * inch, H - 0.9 * inch, "Clinical Report")
    c.setFont("Helvetica", 11)
    c.drawString(0.75 * inch, H - 1.2 * inch, f"Patient ID: {patient_id}")
    c.drawString(0.75 * inch, H - 1.4 * inch, f"Modality: {modality}")

    y = H - 1.8 * inch
    c.setFont("Helvetica", 10)


    max_chars = 105
    for line in (report_text or "").splitlines():
        if not line.strip():
            y -= 0.14 * inch
            continue

        s = line
        while len(s) > max_chars:
            if y < 1.0 * inch:
                c.showPage()
                y = H - 0.9 * inch
                c.setFont("Helvetica", 10)
            c.drawString(0.75 * inch, y, s[:max_chars])
            s = s[max_chars:]
            y -= 0.18 * inch

        if y < 1.0 * inch:
            c.showPage()
            y = H - 0.9 * inch
            c.setFont("Helvetica", 10)
        c.drawString(0.75 * inch, y, s)
        y -= 0.18 * inch

    # Page 2
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, H - 0.9 * inch, "Overlay")

    img = ImageReader(overlay_png)
    max_w = W - 1.5 * inch
    max_h = H - 1.8 * inch
    c.drawImage(img, 0.75 * inch, 0.9 * inch, width=max_w, height=max_h, preserveAspectRatio=True, anchor="c")

    c.save()
    return {"ok": True, "pdf_path": out_pdf}
