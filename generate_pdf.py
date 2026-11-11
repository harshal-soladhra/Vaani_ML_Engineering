import pymupdf
from pathlib import Path

def generate_pdf():
    md_path = Path(__file__).resolve().parent / "assignment_report.md"
    pdf_path = Path(__file__).resolve().parent / "assignment_report.pdf"

    if not md_path.exists():
        print("Missing assignment_report.md")
        return

    text = md_path.read_text(encoding="utf-8")

    doc = pymupdf.open()
    
    # Story-based or text box layout
    # Create pages and insert text cleanly
    lines = text.split("\n")
    
    # Simple multi-page text rendering with clean margins
    page_width, page_height = 612, 792 # Standard Letter
    margin = 40
    rect = pymupdf.Rect(margin, margin, page_width - margin, page_height - margin)
    
    current_page = doc.new_page(width=page_width, height=page_height)
    y = margin
    
    for line in lines:
        if line.startswith("# "):
            fontsize = 15
            color = (0.1, 0.2, 0.5)
            line_str = line.replace("# ", "")
        elif line.startswith("## "):
            fontsize = 12
            color = (0.15, 0.3, 0.6)
            line_str = line.replace("## ", "")
        elif line.startswith("### "):
            fontsize = 10
            color = (0.2, 0.2, 0.2)
            line_str = line.replace("### ", "")
        elif line.startswith("|"):
            fontsize = 7.5
            color = (0.1, 0.1, 0.1)
            line_str = line
        else:
            fontsize = 8.5
            color = (0.05, 0.05, 0.05)
            line_str = line

        # Check page break
        if y > page_height - margin - 20:
            current_page = doc.new_page(width=page_width, height=page_height)
            y = margin

        if line_str.strip():
            # insert text
            rc = current_page.insert_text(
                pymupdf.Point(margin, y),
                line_str[:115], # prevent horizontal overflow
                fontsize=fontsize,
                color=color,
                fontname="helv"
            )
            y += fontsize + 3
        else:
            y += 5

    doc.save(pdf_path)
    doc.close()
    print(f"Generated PDF successfully at {pdf_path} ({pdf_path.stat().st_size} bytes)")

if __name__ == "__main__":
    generate_pdf()
