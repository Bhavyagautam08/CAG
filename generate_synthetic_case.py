import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

OUTPUT_FILE = Path(__file__).with_name("synthetic_property_case_bundle.pdf")
IMAGE_OUTPUT_DIR = Path(__file__).with_name("synthetic_image_bundle")
FONT_PATH = Path(r"C:\Windows\Fonts\Nirmala.ttc")


def _register_hindi_font() -> str:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Install or provide a Devanagari TTF/TTC font; expected {FONT_PATH}"
        )
    pdfmetrics.registerFont(TTFont("HindiFont", str(FONT_PATH), subfontIndex=3))
    return "HindiFont"


def _draw_page(
    document: canvas.Canvas,
    page_number: int,
    title: str,
    lines: list[str],
    hindi_font: str,
    page_width: float,
    page_height: float,
) -> None:
    document.saveState()
    document.setFont("Helvetica-Bold", 12)
    document.setFillColorRGB(0.8, 0.2, 0.2, alpha=0.28)
    document.translate(page_width / 2, page_height / 2)
    document.rotate(45)
    document.drawCentredString(0, 0, "SYNTHETIC TEST CASE FILE - NOT AN OFFICIAL RECORD")
    document.restoreState()

    if random.random() > 0.5:
        document.rotate(random.uniform(-0.5, 0.5))
    document.setFillColorRGB(0, 0, 0)
    document.setFont(hindi_font if any("\u0900" <= char <= "\u097f" for char in title) else "Helvetica-Bold", 16)
    document.drawString(48, page_height - 48, title)
    document.setFont("Helvetica", 9)
    document.drawRightString(page_width - 48, 28, f"Synthetic test only | Page {page_number} of 16")

    y = page_height - 96
    for line in lines:
        has_hindi = any("\u0900" <= char <= "\u097f" for char in line)
        document.setFont(hindi_font if has_hindi else "Helvetica", 12)
        document.drawString(52, y, line)
        y -= 28

    document.setFont("Helvetica-Bold", 9)
    document.setFillColorRGB(0.7, 0.1, 0.1)
    document.drawString(
        52,
        48,
        "SYNTHETIC TEST DOCUMENT - NOT A VALID LEGAL, IDENTITY, OR FINANCIAL RECORD",
    )
    document.showPage()


def create_synthetic_pdf() -> Path:
    random.seed(2026)
    hindi_font = _register_hindi_font()
    document = canvas.Canvas(str(OUTPUT_FILE), pagesize=A4)
    page_width, page_height = A4
    pages: list[tuple[str, list[str]]] = []

    for page_number in range(1, 5):
        lines = [
            f"Document A - Page {page_number} of 4",
            "Donor / दाता: Rajesh Kumar Sharma",
            "Donee / प्राप्तकर्ता: Anil Kumar Sharma",
            "Property ID: FLAT-402-IMPERIAL",
            "Relationship / संबंध: Father-Son",
        ]
        if page_number == 2:
            lines.extend(
                [
                    "Area / क्षेत्रफल: 1250 Sq Ft",
                    "Registration Ref: REG-NM-26-44321",
                ]
            )
        if page_number == 4:
            lines.append("Execution and witness schedule - synthetic continuation")
        pages.append(("GIFT DEED / दानपत्र", lines))

    pages.extend(
        [
            (
                "मूल्यांकन पत्र / Valuation Record",
                [
                    "Valuation Ref: VAL-MH-2026-9921",
                    "संपत्ति आईडी / Property ID: PLOT-45-SEC9",
                    "संपत्ति का क्षेत्रफल / Property Area: 125.0 Sq M",
                    "मूल्यांकन राशि / Assessed Value: INR 45,00,000",
                    "नोट: क्षेत्रफल का विवरण दानपत्र से भिन्न है।",
                ],
            ),
            (
                "DOCUMENT HANDLING CHARGES / दस्तावेज़ संचालन शुल्क",
                [
                    "Receipt No: TEST-RCPT-2026-006",
                    "Received for: Document handling and processing charges",
                    "Amount: INR 750.00",
                    "भुगतानकर्ता: Anil Kumar Sharma",
                    "Synthetic receipt; no real payment was made.",
                ],
            ),
            (
                "IDENTITY VERIFICATION RECORD / पहचान विवरण",
                [
                    "Name: Anil Kumar Sharma",
                    "ID Number: TEST-ID-0001",
                    "SYNTHETIC TEST DOCUMENT - NOT A VALID ID",
                    "केवल परीक्षण हेतु - वास्तविक पहचान पत्र नहीं",
                ],
            ),
            (
                "PAYMENT CHALLAN / भुगतान चालान",
                [
                    "Challan No: TEST-CHALLAN-2026-008",
                    "Registration Ref: REG-NM-26-44321",
                    "Amount Paid: INR 2,500.00",
                    "भुगतान स्थिति: परीक्षण हेतु",
                ],
            ),
        ]
    )

    for page_number in (9, 10):
        lines = [
            f"Document F - Page {page_number - 8} of 2",
            "Property: FLAT-402-IMPERIAL",
            "Building: Imperial Residency Condominium",
            "Owner of record: Anil Kumar Sharma",
        ]
        if page_number == 10:
            lines.append("Common-area statement and synthetic continuation")
        pages.append(("IMPERIAL RESIDENCY CONDOMINIUM RECORD", lines))

    pages.extend(
        [
            (
                "संपत्ति सर्वेक्षण अभिलेख / Property Survey Record",
                [
                    "सर्वेक्षण क्रमांक / Survey No: SURV-TEST-2026-11",
                    "मालिक का नाम / Owner: Rajesh K. Sarma",
                    "संपत्ति आईडी / Property ID: FLAT-402-IMPERIAL",
                    "अभिलेख स्थिति / Record Status: Synthetic sample",
                ],
            ),
            (
                "कार्यालयीन पत्राचार / Official Correspondence - Letter 1",
                [
                    "Subject: Request to confirm property particulars",
                    "Dear Rajesh Kumar Sharma, please confirm the listed property details.",
                    "This correspondence is fictional and for testing only.",
                ],
            ),
            (
                "कार्यालयीन पत्राचार / Official Correspondence - Letter 2",
                [
                    "Subject: Missing supporting document",
                    "Please provide Annexure B (Clearance Certificate), which is not included.",
                    "This correspondence is fictional and for testing only.",
                ],
            ),
            (
                "कार्यालयीन पत्राचार / Official Correspondence - Letter 3",
                [
                    "Subject: Acknowledgement",
                    "Your synthetic test response has been received for review.",
                    "This correspondence is fictional and for testing only.",
                ],
            ),
            (
                "REGISTRATION RECORD / पंजीकरण विवरण",
                [
                    "Registration Ref: REG-NM-26-44321",
                    "Document Type: Gift Deed",
                    "Property ID: FLAT-402-IMPERIAL",
                    "Status: SYNTHETIC TEST ENTRY",
                ],
            ),
            (
                "PROPERTY TAX EXTRACT / संपत्ति कर रसीद",
                [
                    "Owner: Vikram Singh",
                    "Property: FLAT-102-IMPERIAL",
                    "Tax Receipt No: TEST-TAX-2026-016",
                    "Unrelated synthetic property; keep association unresolved.",
                ],
            ),
        ]
    )

    for page_number, (title, lines) in enumerate(pages, 1):
        _draw_page(document, page_number, title, lines, hindi_font, page_width, page_height)
    document.save()

    reader = PdfReader(str(OUTPUT_FILE))
    extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(reader.pages) != 16:
        raise ValueError(f"Expected 16 pages, generated {len(reader.pages)}.")
    for expected in (
        "REG-NM-26-44321",
        "VAL-MH-2026-9921",
        "TEST-ID-0001",
        "Annexure B",
        "Vikram Singh",
        "दाता",
    ):
        if expected not in extracted_text:
            raise ValueError(f"Generated PDF is missing expected text: {expected}")
    return OUTPUT_FILE


def create_synthetic_image_bundle() -> list[Path]:
    IMAGE_OUTPUT_DIR.mkdir(exist_ok=True)
    pages = [
        [
            "SYNTHETIC TEST PAGE - NOT AN OFFICIAL RECORD",
            "GIFT DEED",
            "Registration Ref: SYNTH-REG-2026-001",
            "Property ID: FLAT-TEST-001",
            "Donor: Synthetic Alice Example",
        ],
        [
            "SYNTHETIC TEST PAGE - NOT AN OFFICIAL RECORD",
            "GIFT DEED - CONTINUATION",
            "Registration Ref: SYNTH-REG-2026-001",
            "Property ID: FLAT-TEST-002",
            "Reviewer: compare the differing property ID with the source pages.",
        ],
    ]
    font = ImageFont.load_default(size=44)
    output_paths = []
    for page_number, lines in enumerate(pages, 1):
        image = Image.new("RGB", (1800, 520), "white")
        draw = ImageDraw.Draw(image)
        y = 28
        for line in lines:
            draw.text((40, y), line, font=font, fill="black")
            y += 92
        path = IMAGE_OUTPUT_DIR / f"synthetic-page-{page_number}.png"
        image.save(path, format="PNG")
        output_paths.append(path)
    return output_paths


if __name__ == "__main__":
    path = create_synthetic_pdf()
    print(f"Generated {path.resolve()} ({path.stat().st_size} bytes, 16 pages).")
    print("This is a digitally generated PDF, not an image-only scan.")
    image_paths = create_synthetic_image_bundle()
    print(f"Generated {len(image_paths)} synthetic image pages in {IMAGE_OUTPUT_DIR.resolve()}.")
