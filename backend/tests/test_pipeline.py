import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject, NumberObject

from backend.app.models import LogicalDocument, Page
from backend.app import ocr as ocr_module
from backend.app.ocr import _combine_words, _easyocr_mode, _paddle_words, _words_to_text, get_ocr_status
from backend.app.pdf_utils import LocalImageBundle, LocalPdf
from backend.app.processing import (
    INVOICE_FIELD_LABELS,
    _conflicting_field_values,
    _ocr_page,
    _starts_new_document,
    _unresolved_annexure_references,
    classify_text,
    extract_fields,
)


def test_classifies_english_document_with_rule_evidence():
    category, proxy, evidence = classify_text("GIFT DEED Registration No. 12345")
    assert category == "gift_deed"
    assert 0 < proxy < 1
    assert "gift deed" in evidence.casefold()


def test_classifies_hindi_valuation_heading():
    category, _, evidence = classify_text("मूल्यांकन दस्तावेज")
    assert category == "valuation"
    assert evidence


def test_classifies_identity_registration_and_property_tax_records():
    assert classify_text("IDENTITY VERIFICATION RECORD")[0] == "identity_document"
    assert classify_text("REGISTRATION RECORD Registration Ref: REG-001")[0] == "registration"
    assert classify_text("REGISTRATION RECORD Gift Deed Registration Ref: REG-001")[0] == "registration"
    assert classify_text("PROPERTY TAX EXTRACT Tax Receipt")[0] == "property_tax"
    category, _, evidence = classify_text("TAX INVOICE\nInvoice Serial Number: INV-001\nGSTIN:")
    assert category == "tax_invoice"
    assert "tax invoice" in evidence.casefold()


def test_unknown_is_explicit_when_no_rules_match():
    assert classify_text("unrelated handwritten notes")[0] == "unknown"


def test_paddle_result_parser_keeps_punctuation_and_word_confidence():
    words = _paddle_words(
        [
            {
                "res": {
                    "rec_texts": ["Registration No: REG-001:"],
                    "rec_scores": [0.91],
                    "rec_boxes": [[10, 20, 310, 60]],
                }
            }
        ],
        1,
        0,
    )

    assert [word["text"] for word in words] == ["Registration", "No:", "REG-001:"]
    assert all(word["confidence"] == pytest.approx(91) for word in words)
    assert all(word["engine"] == "paddleocr" for word in words)


def test_ocr_ensemble_uses_agreement_and_preserves_conflicts_as_alternatives():
    paddle_words = [
        {"text": "REG-001", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": 92, "engine": "paddleocr"},
        {"text": "GIFT", "x": 10, "y": 60, "width": 50, "height": 20, "confidence": 80, "engine": "paddleocr"},
    ]
    easy_words = [
        {"text": "REG-001", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": 75, "engine": "easyocr"},
        {"text": "DEED", "x": 10, "y": 60, "width": 50, "height": 20, "confidence": 88, "engine": "easyocr"},
        {"text": "DATE", "x": 10, "y": 100, "width": 50, "height": 20, "confidence": 88, "engine": "easyocr"},
    ]

    combined = _combine_words(paddle_words, easy_words)

    assert len(combined) == 3
    assert combined[0]["agreement"] is True
    assert [score["engine"] for score in combined[0]["engine_scores"]] == ["paddleocr", "easyocr"]
    assert combined[1]["text"] == "GIFT"
    assert combined[1]["alternatives"][0]["text"] == "DEED"
    assert _words_to_text(combined) == "REG-001\nGIFT\nDATE"


@pytest.mark.parametrize(
    ("paddle_confidence", "easy_confidence", "expected_text", "expected_engine"),
    [
        (42, 94, "INVOICE", "easyocr"),
        (96, 58, "INV0ICE", "paddleocr"),
    ],
)
def test_ocr_ensemble_selects_a_confident_reading_and_keeps_the_other(
    paddle_confidence: float,
    easy_confidence: float,
    expected_text: str,
    expected_engine: str,
):
    paddle_words = [
        {"text": "INV0ICE", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": paddle_confidence, "engine": "paddleocr"},
    ]
    easy_words = [
        {"text": "INVOICE", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": easy_confidence, "engine": "easyocr"},
    ]

    combined = _combine_words(paddle_words, easy_words)

    assert len(combined) == 1
    assert combined[0]["text"] == expected_text
    assert combined[0]["engine"] == expected_engine
    assert combined[0]["alternatives"][0]["engine"] != expected_engine
    assert combined[0]["agreement"] is False


def test_ocr_ensemble_keeps_conflicting_readings_when_confidence_is_not_decisive():
    paddle_words = [
        {"text": "INVOICE", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": 82, "engine": "paddleocr"},
    ]
    easy_words = [
        {"text": "INV0ICE", "x": 10, "y": 20, "width": 80, "height": 20, "confidence": 80, "engine": "easyocr"},
    ]

    combined = _combine_words(paddle_words, easy_words)

    assert combined[0]["text"] == "INVOICE"
    assert combined[0]["selection_reason"] == "confidence_not_decisive"
    assert combined[0]["alternatives"][0]["text"] == "INV0ICE"


def test_secondary_ocr_is_skipped_for_dense_high_confidence_paddle_pages():
    confident_words = [{"text": f"word-{index}", "confidence": 96} for index in range(20)]
    uncertain_words = confident_words[:-1] + [{"text": "uncertain", "confidence": 60}]
    poor_words = [{"text": f"word-{index}", "confidence": 70} for index in range(20)]

    assert _easyocr_mode(confident_words) == "skip"
    assert _easyocr_mode(uncertain_words) == "targeted"
    assert _easyocr_mode(poor_words) == "full"
    assert _easyocr_mode(confident_words[:2]) == "full"
    assert _easyocr_mode(confident_words[:3]) == "skip"


def test_run_ocr_uses_both_engines_and_records_agreement(monkeypatch, tmp_path: Path):
    class FakePaddleReader:
        def predict(self, image_path):
            assert Path(image_path).name == "scan.png"
            return [
                {
                    "res": {
                        "rec_texts": ["GIFT DEED"],
                        "rec_scores": [0.9],
                        "rec_boxes": [[10, 10, 110, 30]],
                    }
                }
            ]

    class FakeEasyReader:
        def readtext(self, image_path, detail, paragraph):
            assert Path(image_path).name == "scan.png"
            assert detail == 1
            assert paragraph is False
            return [([[10, 10], [110, 10], [110, 30], [10, 30]], "GIFT DEED", 0.8)]

    selected_languages = []
    reported_stages = []
    monkeypatch.setattr(ocr_module, "_configured_languages", lambda: ("en", "mr"))
    monkeypatch.setattr(
        ocr_module,
        "_paddle_reader",
        lambda language: selected_languages.append(language) or FakePaddleReader(),
    )
    monkeypatch.setattr(
        ocr_module,
        "_easy_reader",
        lambda languages: (selected_languages.append(languages) or FakeEasyReader()),
    )

    text, words, confidence, engine, errors = ocr_module.run_ocr(
        tmp_path / "scan.png",
        1,
        0,
        lambda stage, _fraction: reported_stages.append(stage),
    )

    assert selected_languages == ["devanagari", ("en", "mr")]
    assert text == "GIFT DEED"
    assert all(word["agreement"] is True for word in words)
    assert confidence == pytest.approx(90)
    assert engine == "paddleocr+easyocr"
    assert errors == []
    assert "loading_easyocr" in reported_stages
    assert reported_stages[-1] == "ocr_complete"


def test_run_ocr_skips_full_easyocr_on_dense_high_confidence_page(monkeypatch, tmp_path: Path):
    class FakePaddleReader:
        def predict(self, _image_path):
            return [{
                "res": {
                    "rec_texts": [" ".join(f"word{index}" for index in range(20))],
                    "rec_scores": [0.99],
                    "rec_boxes": [[10, 10, 210, 30]],
                }
            }]

    monkeypatch.setattr(ocr_module, "_configured_languages", lambda: ("en",))
    monkeypatch.setattr(ocr_module, "_paddle_reader", lambda _language: FakePaddleReader())
    monkeypatch.setattr(
        ocr_module,
        "_easy_reader",
        lambda _languages: (_ for _ in ()).throw(AssertionError("EasyOCR should be skipped")),
    )

    text, words, confidence, engine, errors = ocr_module.run_ocr(tmp_path / "scan.png", 1, 0)

    assert len(words) == 20
    assert "word19" in text
    assert confidence == pytest.approx(99)
    assert engine == "paddleocr"
    assert errors == []


def test_run_ocr_targets_easyocr_to_uncertain_paddle_lines(monkeypatch, tmp_path: Path):
    line = " ".join(f"word{index}" for index in range(20))

    class FakePaddleReader:
        def predict(self, _image_path):
            return [{
                "res": {
                    "rec_texts": [line],
                    "rec_scores": [0.9],
                    "rec_boxes": [[10, 10, 210, 30]],
                }
            }]

    class FakeEasyReader:
        def recognize(self, image_path, horizontal_list, free_list, detail, paragraph):
            assert Path(image_path).is_file()
            assert horizontal_list
            assert free_list == []
            assert detail == 1
            assert paragraph is False
            x1, x2, y1, y2 = horizontal_list[0]
            return [([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], line, 0.99)]

    image_path = tmp_path / "scan.png"
    Image.new("RGB", (240, 50), "white").save(image_path)
    monkeypatch.setattr(ocr_module, "_configured_languages", lambda: ("en",))
    monkeypatch.setattr(ocr_module, "_paddle_reader", lambda _language: FakePaddleReader())
    monkeypatch.setattr(ocr_module, "_easy_reader", lambda _languages: FakeEasyReader())

    text, words, _confidence, engine, errors = ocr_module.run_ocr(image_path, 1, 0)

    assert line in text
    assert len(words) >= 20
    assert engine == "paddleocr+easyocr"
    assert errors == []


def test_extracts_only_labelled_registration_reference():
    document = LogicalDocument(id="doc-1", ingestion_id="ing-1", start_page=1, end_page=1)
    page = Page(id="page-1", ingestion_id="ing-1", page_number=1, digital_text="Registration No: AB-1234\nDate: 12/03/2024")
    fields = extract_fields(document, [page])
    by_name = {field.field_name: field for field in fields}
    assert by_name["registration_number"].extracted_value == "AB-1234"
    assert by_name["document_date"].source_page == 1
    assert "AB-1234" in by_name["registration_number"].evidence


def test_extracts_labelled_fields_without_treating_headings_as_values():
    document = LogicalDocument(id="doc-fields", ingestion_id="ing-1", start_page=1, end_page=1)
    page = Page(
        id="page-fields",
        ingestion_id="ing-1",
        page_number=1,
        digital_text=(
            "REGISTRATION RECORD\n"
            "Registration Ref: REG-NM-26-44321\n"
            "Survey Record\n"
            "Survey No: SURV-TEST-2026-11\n"
            "Property ID: FLAT-402-IMPERIAL\n"
            "Property: FLAT-102-IMPERIAL\n"
            "Area / क्षेत्रफल: 1250 Sq Ft\n"
            "Assessed Value: INR 45,00,000\n"
            "ID Number: TEST-ID-0001\n"
            "Donor / दाता: Rajesh Kumar Sharma\n"
            "Donee / प्राप्तकर्ता: Anil Kumar Sharma"
        ),
    )

    fields = {field.field_name: field for field in extract_fields(document, [page])}

    assert fields["registration_number"].extracted_value == "REG-NM-26-44321"
    assert fields["survey_number"].extracted_value == "SURV-TEST-2026-11"
    assert fields["property_id"].extracted_value == "FLAT-402-IMPERIAL"
    assert fields["property_area"].extracted_value == "1250 Sq Ft"
    assert fields["assessed_value"].extracted_value == "45,00,000"
    assert fields["identity_number"].extracted_value == "TEST-ID-0001"
    assert fields["identity_number"].sensitive is True
    assert fields["donor_name"].extracted_value == "Rajesh Kumar Sharma"
    assert "Donor" in fields["donor_name"].evidence
    assert fields["donee_name"].extracted_value == "Anil Kumar Sharma"

    tax_document = LogicalDocument(id="doc-tax", ingestion_id="ing-1", start_page=1, end_page=1)
    tax_page = Page(
        id="page-tax",
        ingestion_id="ing-1",
        page_number=1,
        digital_text="Owner: Vikram Singh\nProperty: FLAT-102-IMPERIAL",
    )
    tax_fields = {field.field_name: field for field in extract_fields(tax_document, [tax_page])}
    assert tax_fields["owner_name"].extracted_value == "Vikram Singh"
    assert tax_fields["property_id"].extracted_value == "FLAT-102-IMPERIAL"


def test_tax_invoice_extraction_saves_labeled_values_and_marks_missing_values():
    document = LogicalDocument(
        id="tax-invoice-doc",
        ingestion_id="tax-invoice-ing",
        start_page=1,
        end_page=1,
        category="tax_invoice",
    )
    page = Page(
        id="tax-invoice-page",
        ingestion_id=document.ingestion_id,
        page_number=1,
        ocr_text_original=(
            "TAX INVOICE\n"
            "Logo 3MXYZ\n"
            "Registered Office:\nTel-No:\nCIN:\n"
            "GSTIN: 29ABCDE1234F1Z5\n"
            "IRN: abcdef0123456789abcdef0123456789\n"
            "Invoice Serial Number: INV-1001 Veh. No:\n"
            "Invoice Date: 02-07-2020 Date & Time of Supply:\n"
            "Invoice Currency Code: INR Place of Supply: Karnataka\n"
            "Details of Customer (Bill to) Details of Consignee (Ship to)\n"
            "Name: HDFC Name: Receiver Ltd\n"
            "Address: 234, Karnataka Address: Tech Park\n"
            "State Name: DELHI State Name: KARNATAKA\n"
            "State Code: 07 State Code: 29\n"
            "GSTIN: 07ABCDE1234F1Z5 GSTIN: 29ABCDE1234F1Z5\n"
            "Description of Goods HSN Code\n"
            "item15 9401 0.00 19.00 1,000.00 6,400.00\n"
            "Invoice Value (In Words) Total ₹10,000.00\n"
            "Ten Thousand Rupees Only"
        ),
    )

    fields = {field.field_name: field for field in extract_fields(document, [page])}

    assert fields["supplier_name"].extracted_value == "3MXYZ"
    assert fields["invoice_number"].extracted_value == "INV-1001"
    assert fields["invoice_date"].extracted_value == "02-07-2020"
    assert fields["invoice_currency"].extracted_value == "INR"
    assert fields["supplier_gstin"].extracted_value == "29ABCDE1234F1Z5"
    assert fields["customer_name"].extracted_value == "HDFC"
    assert fields["consignee_name"].extracted_value == "Receiver Ltd"
    assert fields["customer_state"].extracted_value == "DELHI"
    assert fields["consignee_state"].extracted_value == "KARNATAKA"
    assert fields["hsn_sac_code"].extracted_value == "9401"
    assert fields["line_item_description"].extracted_value == "item15"
    assert fields["invoice_line_item_raw"].extracted_value.startswith("item15 9401")
    assert fields["invoice_total"].extracted_value == "10,000.00"
    assert fields["amount_in_words"].extracted_value == "Ten Thousand Rupees Only"
    assert fields["reverse_charge"].extracted_value is None
    assert fields["reverse_charge"].verification_status == "not_available"
    assert fields["reverse_charge"].evidence.startswith("Expected invoice field not found")
    assert len(fields) == len(INVOICE_FIELD_LABELS)


def test_tax_invoice_labels_do_not_capture_values_from_later_lines():
    document = LogicalDocument(
        id="tax-invoice-boundary-doc",
        ingestion_id="tax-invoice-boundary-ing",
        start_page=1,
        end_page=1,
        category="tax_invoice",
    )
    irn = "abcdef0123456789" * 4
    page = Page(
        id="tax-invoice-boundary-page",
        ingestion_id=document.ingestion_id,
        page_number=1,
        ocr_text_original=(
            "TAX INVOICE\n"
            "Logo 3MXYZ\n"
            "Registered Office:\nTel-No:\nCIN:\n"
            "IRN:\n"
            f"{irn[:40]}\n{irn[40:]}\n"
            "Tax is Payable on Reverse No Transportation Mode:\n"
            "Charge:\n"
            "Invoice Serial Number: INV-2001 Veh. No:\n"
            "Order Reference:\n"
            "Terms & Conditions of sale\n"
            "Invoice Value (In Words) Total ₹10,000.00\n"
            "Freight Charges\nLoading and Packaging Charges\n"
            "Ten Thousand Rupees Only Insurance Charges\n"
        ),
    )

    fields = {field.field_name: field for field in extract_fields(document, [page])}

    assert fields["invoice_irn"].extracted_value == irn
    assert fields["invoice_number"].extracted_value == "INV-2001"
    assert fields["supplier_cin"].verification_status == "not_available"
    assert fields["transportation_mode"].verification_status == "not_available"
    assert fields["vehicle_number"].verification_status == "not_available"
    assert fields["order_reference"].verification_status == "not_available"
    assert fields["amount_in_words"].extracted_value == "Ten Thousand Rupees Only"


def test_identifies_referenced_annexures_without_a_matching_page_heading():
    letter = Page(
        id="letter",
        ingestion_id="ing-1",
        page_number=1,
        digital_text="Please provide Annexure B (Clearance Certificate).",
    )
    annexure_a = Page(
        id="annexure-a",
        ingestion_id="ing-1",
        page_number=2,
        digital_text="Annexure A - Application form",
    )
    annexure_b = Page(
        id="annexure-b",
        ingestion_id="ing-1",
        page_number=3,
        digital_text="Annexure B - Clearance Certificate",
    )

    assert _unresolved_annexure_references([letter], [letter, annexure_a]) == [
        (1, "B", "Please provide Annexure B (Clearance Certificate).")
    ]
    assert _unresolved_annexure_references([letter], [letter, annexure_b]) == []


def test_flags_conflicting_property_and_registration_values_for_review():
    pages = [
        Page(
            id="conflict-1",
            ingestion_id="ing-1",
            page_number=1,
            digital_text="Registration Ref: REG-001\nProperty ID: FLAT-101",
        ),
        Page(
            id="conflict-2",
            ingestion_id="ing-1",
            page_number=2,
            digital_text="Registration Ref: REG-002\nProperty ID: FLAT-102",
        ),
    ]

    conflicts = dict(_conflicting_field_values(pages))

    assert conflicts["registration_number"] == [(1, "REG-001"), (2, "REG-002")]
    assert conflicts["property_id"] == [(1, "FLAT-101"), (2, "FLAT-102")]


def test_image_bundle_rejects_paths_outside_its_page_directory(tmp_path: Path):
    manifest = tmp_path / "image-manifest.json"
    manifest.write_text('["../../outside.png"]', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid page path"):
        LocalImageBundle(manifest)


def test_explicit_letter_numbers_split_but_document_pages_stay_together():
    previous_letter = Page(
        id="letter-1",
        ingestion_id="ing-1",
        page_number=12,
        category="official_letter",
        digital_text="Official Correspondence - Letter 1",
    )
    next_letter = Page(
        id="letter-2",
        ingestion_id="ing-1",
        page_number=13,
        category="official_letter",
        digital_text="Official Correspondence - Letter 2",
    )
    same_document_page = Page(
        id="gift-page-2",
        ingestion_id="ing-1",
        page_number=2,
        category="gift_deed",
        digital_text="Document A - Page 2 of 4",
    )
    first_document_page = Page(
        id="gift-page-1",
        ingestion_id="ing-1",
        page_number=1,
        category="gift_deed",
        digital_text="Document A - Page 1 of 4",
    )

    assert _starts_new_document(previous_letter, next_letter)
    assert not _starts_new_document(first_document_page, same_document_page)


def _image_only_pdf() -> bytes:
    image = Image.new("RGB", (1800, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 60), "GIFT DEED Registration No: AB-1234", font=ImageFont.load_default(size=80), fill="black")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG", quality=95)

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    image_object = DecodedStreamObject()
    image_object.set_data(image_bytes.getvalue())
    image_object.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(image.width),
            NameObject("/Height"): NumberObject(image.height),
            NameObject("/ColorSpace"): NameObject("/DeviceRGB"),
            NameObject("/BitsPerComponent"): NumberObject(8),
            NameObject("/Filter"): NameObject("/DCTDecode"),
        }
    )
    image_reference = writer._add_object(image_object)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im0"): image_reference})}
    )
    content = DecodedStreamObject()
    content.set_data(b"q 550 0 0 250 30 450 cm /Im0 Do Q")
    page[NameObject("/Contents")] = writer._add_object(content)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_ocr_reads_embedded_image_pdf_and_keeps_source_image(tmp_path: Path):
    if not get_ocr_status()["configured"]:
        pytest.skip("PaddleOCR and EasyOCR are required for the OCR integration test")

    pdf_path = tmp_path / "scanned.pdf"
    pdf_path.write_bytes(_image_only_pdf())
    image_dir = tmp_path / "page-images"
    image_dir.mkdir()
    with LocalPdf(pdf_path) as pdf:
        if pdf.engine != "pypdf":
            pytest.skip("This test covers the pypdf embedded-image fallback")
        image_sources = pdf.embedded_images(0)
    assert len(image_sources) == 1

    text, engine, error, confidence, word_boxes, image_paths = _ocr_page(
        image_sources, "", "pypdf", image_dir, 1, 0
    )

    assert engine in {"paddleocr", "paddleocr+easyocr"}
    assert error is None
    assert "gift deed" in text.casefold()
    assert confidence is not None
    assert json.loads(word_boxes or "[]")
    assert image_paths is not None
    assert Path(image_paths[0]).read_bytes() == image_sources[0][0]
    assert Path(image_paths[1]).is_file()


def test_ocr_reports_unsupported_page_when_no_render_or_embedded_image_exists(tmp_path: Path):
    text, engine, error, confidence, word_boxes, image_paths = _ocr_page(
        [], "", "pypdf", tmp_path / "page-images", 1, 0
    )

    assert text == ""
    assert engine == "pypdf"
    assert error is not None
    assert "no full-page pdf renderer is available" in error.casefold()
    assert "no extractable embedded scan image" in error.casefold()
    assert "OCR was skipped" in error
    assert confidence is None
    assert word_boxes is None
    assert image_paths is None
