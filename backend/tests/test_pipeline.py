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
    assess_stamp_duty_article_34,
    classify_text,
    evaluate_article_34_relationship,
    extract_fields,
    extract_market_value,
    extract_property_schedule,
    extract_stamp_duty_paid,
    parse_indian_currency_amount,
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
    monkeypatch.setattr(ocr_module, "_secondary_ocr_enabled", lambda: True)
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
    monkeypatch.setattr(ocr_module, "_secondary_ocr_enabled", lambda: True)
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
    monkeypatch.setattr(ocr_module, "_secondary_ocr_enabled", lambda: True)
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



def test_donor_donee_relationship_english_recital_extraction():
    document = LogicalDocument(id="doc-gift-en", ingestion_id="ing-1", category="gift_deed", start_page=1, end_page=1)
    page = Page(
        id="page-gift-en",
        ingestion_id="ing-1",
        page_number=1,
        digital_text=(
            "DEED OF GIFT\n"
            "Donor: Ramesh Kumar Sharma\n"
            "Donee: Anil Kumar Sharma\n"
            "WHEREAS the Donor is the father of the Donee and out of natural love and affection desires to gift the property.\n"
            "NOW THIS DEED WITNESSETH that the Donor hereby conveys the schedule premises."
        ),
    )
    fields = {f.field_name: f for f in extract_fields(document, [page])}
    assert "donor_donee_relationship" in fields
    rel_field = fields["donor_donee_relationship"]
    assert rel_field.extracted_value == "Father and Son"
    assert rel_field.normalized_value == "father and son"
    assert rel_field.source_page == 1
    assert "WHEREAS the Donor is the father of the Donee and out of natural love and affection" in rel_field.evidence
    assert rel_field.verification_status == "unreviewed"
    assert rel_field.extraction_method == "recital_rule"


def test_donor_donee_relationship_hindi_recital_extraction():
    document = LogicalDocument(id="doc-gift-hi", ingestion_id="ing-1", category="gift_deed", start_page=2, end_page=2)
    page = Page(
        id="page-gift-hi",
        ingestion_id="ing-1",
        page_number=2,
        digital_text=(
            "दानपत्र\n"
            "दाता / Donor: राम कुमार\n"
            "प्राप्तकर्ता / Donee: श्याम कुमार\n"
            "चूंकि दाता और प्राप्तकर्ता के बीच पिता और पुत्र का संबंध है और स्वाभाविक प्रेम व स्नेह के कारण यह दानपत्र निष्पादित किया जा रहा है।"
        ),
    )
    fields = {f.field_name: f for f in extract_fields(document, [page])}
    assert "donor_donee_relationship" in fields
    rel_field = fields["donor_donee_relationship"]
    assert rel_field.extracted_value == "पिता और पुत्र"
    assert rel_field.normalized_value == "father and son"
    assert rel_field.source_page == 2
    assert "चूंकि दाता और प्राप्तकर्ता के बीच पिता और पुत्र का संबंध है" in rel_field.evidence
    assert rel_field.verification_status == "unreviewed"


def test_donor_donee_relationship_marathi_recital_extraction():
    document = LogicalDocument(id="doc-gift-mr", ingestion_id="ing-1", category="gift_deed", start_page=1, end_page=1)
    page = Page(
        id="page-gift-mr",
        ingestion_id="ing-1",
        page_number=1,
        digital_text=(
            "बक्षीसपत्र\n"
            "देणार: विजय दत्तात्रय पाटील\n"
            "घेणार: संजय दत्तात्रय पाटील\n"
            "देणार व घेणार हे सख्खे बंधू असून त्यांच्यात नैसर्गिक प्रेम व जिव्हाळा आहे."
        ),
    )
    fields = {f.field_name: f for f in extract_fields(document, [page])}
    assert "donor_donee_relationship" in fields
    rel_field = fields["donor_donee_relationship"]
    assert rel_field.extracted_value == "सख्खे बंधू"
    assert rel_field.normalized_value == "brothers"
    assert rel_field.source_page == 1
    assert "देणार व घेणार हे सख्खे बंधू असून" in rel_field.evidence
    assert rel_field.verification_status == "unreviewed"


def test_donor_donee_relationship_shared_surname_not_inferred():
    # Important requirement: system must NOT infer a relationship merely because donor and donee share a surname
    document = LogicalDocument(id="doc-gift-surname", ingestion_id="ing-1", category="gift_deed", start_page=1, end_page=1)
    page = Page(
        id="page-gift-surname",
        ingestion_id="ing-1",
        page_number=1,
        digital_text=(
            "GIFT DEED\n"
            "Donor: Rajesh Kumar Sharma\n"
            "Donee: Anil Kumar Sharma\n"
            "WHEREAS the Donor is the sole and absolute owner of Flat 101.\n"
            "AND WHEREAS the Donor desires to convey the property to the Donee without monetary consideration."
        ),
    )
    fields = {f.field_name: f for f in extract_fields(document, [page])}
    assert "donor_donee_relationship" in fields
    rel_field = fields["donor_donee_relationship"]
    assert rel_field.extracted_value is None
    assert rel_field.verification_status == "not_available"
    assert "No explicit donor-donee relationship found" in rel_field.evidence


def test_donor_donee_relationship_ambiguous_recital_flagged():
    # Vague relationship terms in recital flagged as ambiguous
    document = LogicalDocument(id="doc-gift-amb", ingestion_id="ing-1", category="gift_deed", start_page=1, end_page=1)
    page = Page(
        id="page-gift-amb",
        ingestion_id="ing-1",
        page_number=1,
        digital_text=(
            "GIFT DEED\n"
            "Donor: Rajesh Kumar Sharma\n"
            "Donee: Anil Kumar Sharma\n"
            "WHEREAS the Donor and Donee are near relatives and out of natural love and affection execute this deed."
        ),
    )
    fields = {f.field_name: f for f in extract_fields(document, [page])}
    assert "donor_donee_relationship" in fields
    rel_field = fields["donor_donee_relationship"]
    assert rel_field.extraction_method == "ambiguous_recital"
    assert rel_field.verification_status == "unreviewed"
    assert rel_field.extracted_value == "Near Relatives"


def test_article_34_config_and_stamp_duty_assessment_model():
    from decimal import Decimal
    from backend.app.config import (
        STAMP_DUTY_RULE_VERSION,
        STAMP_DUTY_LEGAL_REFERENCE,
        STAMP_DUTY_EFFECTIVE_FROM,
        STAMP_DUTY_EFFECTIVE_TO,
        STAMP_DUTY_BASE_AMOUNT,
        STAMP_DUTY_METRO_CESS_RATE,
        ARTICLE_34_ELIGIBLE_RELATIONSHIPS,
        ARTICLE_34_INELIGIBLE_RELATIONSHIPS,
        METRO_CESS_JURISDICTIONS,
        NON_METRO_INDICATORS,
        get_settings,
    )
    from backend.app.models import Document, LogicalDocument, StampDutyAssessment

    # 1. Config validation
    settings = get_settings()
    assert STAMP_DUTY_RULE_VERSION == "MH-ART34-2024.1"
    assert settings.stamp_duty_rule_version == "MH-ART34-2024.1"
    assert "Article 34" in STAMP_DUTY_LEGAL_REFERENCE
    assert STAMP_DUTY_EFFECTIVE_FROM == "2017-05-16"
    assert STAMP_DUTY_EFFECTIVE_TO is None
    assert STAMP_DUTY_BASE_AMOUNT == Decimal("200")
    assert STAMP_DUTY_METRO_CESS_RATE == Decimal("0.01")

    # Controlled eligible relationships (normalized and multilingual)
    for term in ["father", "mother", "son", "daughter", "husband", "wife", "brother", "sister", "grandson", "granddaughter"]:
        assert term in ARTICLE_34_ELIGIBLE_RELATIONSHIPS
    for term in ["पिता", "पुत्र", "माता", "पुत्री", "वडील", "आई", "मुलगा", "मुलगी", "पती", "पत्नी", "भाऊ", "बहीण"]:
        assert term in ARTICLE_34_ELIGIBLE_RELATIONSHIPS

    # Ineligibles
    for term in ["uncle", "aunt", "nephew", "niece", "cousin", "friend", "काका", "मामा", "पुतण्या", "भाचा", "मित्र"]:
        assert term in ARTICLE_34_INELIGIBLE_RELATIONSHIPS

    # Jurisdictions
    for jur in ["mumbai", "pune", "thane", "nagpur", "nashik", "मुंबई", "पुणे"]:
        assert jur in METRO_CESS_JURISDICTIONS
    for non_metro in ["gram panchayat", "ग्रामपंचायत"]:
        assert non_metro in NON_METRO_INDICATORS

    # 2. Model & relationship validation
    assert Document is LogicalDocument
    doc = LogicalDocument(id="doc-test-art34", ingestion_id="ing-art34", start_page=1, end_page=2, category="gift_deed")
    assessment = StampDutyAssessment(
        document_id=doc.id,
        donor_name="Rajesh Sharma",
        donee_name="Kabir Sharma",
        relationship="Father and Son",
        relationship_normalized="father_and_son",
        relationship_eligibility="eligible",
        property_type="residential_property",
        property_description="Flat 402",
        property_location="Worli, Mumbai",
        jurisdiction="maharashtra_metro_cess",
        market_value=126480000,
        base_duty=200,
        metro_cess_rate=0.01,
        metro_cess_amount=1264800,
        expected_total=1265000,
        stamp_duty_paid=200,
        metro_cess_paid=1264800,
        total_stamp_duty_paid=1265000,
        difference=0,
        status="compliant",
        legal_rule_version="MH-ART34-2024.1",
        review_status="unreviewed",
    )
    doc.stamp_duty_assessment = assessment
    assert doc.stamp_duty_assessment is assessment
    assert assessment.document is doc
    assert assessment.status == "compliant"
    assert assessment.legal_rule_version == "MH-ART34-2024.1"


def test_parse_indian_currency_amount_formats_and_devanagari():
    # Devanagari translation and various formats
    assert parse_indian_currency_amount("₹25,00,000") == 2500000
    assert parse_indian_currency_amount("₹ 25,00,000/-") == 2500000
    assert parse_indian_currency_amount("Rs. 25,00,000") == 2500000
    assert parse_indian_currency_amount("Rs 2500000") == 2500000
    assert parse_indian_currency_amount("INR 25,00,000.00") == 2500000
    assert parse_indian_currency_amount("रु. १५,००,०००/-") == 1500000
    assert parse_indian_currency_amount("रुपये २५,००,०००") == 2500000
    assert parse_indian_currency_amount("₹ 200/-") == 200
    assert parse_indian_currency_amount("Market Value: ₹ 25,200/-") == 25200
    # Digits preceding currency symbol on same line
    assert parse_indian_currency_amount("Gat No. 42, Valuation: ₹ 30,00,000/-") == 3000000
    # None and invalid
    assert parse_indian_currency_amount(None) is None
    assert parse_indian_currency_amount("") is None
    assert parse_indian_currency_amount("No amount here") is None


def test_extract_property_schedule_classification_and_ambiguity():
    # 1. Agricultural primary terms
    t1 = [(1, "Schedule of Property: Agricultural land bearing Survey No. 45/2 at Village Haveli")]
    prop_type1, _, _, surv1, _, _, is_amb1, ev1, pg1 = extract_property_schedule(t1)
    assert prop_type1 == "agricultural_land"
    assert not is_amb1
    assert surv1 == "45/2"
    assert "Agricultural primary terms" in ev1

    # 2. Marathi agricultural primary terms
    t2 = [(1, "परिशिष्ट: मौजे हवेली येथील शेती जमीन गट क्रमांक १०२")]
    prop_type2, _, _, surv2, _, _, is_amb2, ev2, _ = extract_property_schedule(t2)
    assert prop_type2 == "agricultural_land"
    assert not is_amb2
    assert surv2 == "१०२"

    # 3. Supporting indicators ONLY (survey / gat number alone MUST NEVER classify as agricultural)
    t3 = [(1, "Schedule of Property: Property bearing Survey Number 105, Gat Number 34 at Pune")]
    prop_type3, _, _, surv3, _, _, is_amb3, ev3, _ = extract_property_schedule(t3)
    assert prop_type3 == "unknown"
    assert not is_amb3
    assert "Supporting indicators only" in ev3

    # 4. Residential primary terms
    t4 = [(1, "Schedule of Property: Residential flat no 402, Building A, Pune")]
    prop_type4, _, _, _, _, _, is_amb4, ev4, _ = extract_property_schedule(t4)
    assert prop_type4 == "residential_property"
    assert not is_amb4

    # 5. Marathi residential primary terms
    t5 = [(1, "मिळकतीचे वर्णन: सदनिका क्र. ३०२, तिसरा मजला, पुणे")]
    prop_type5, _, _, _, _, _, is_amb5, ev5, _ = extract_property_schedule(t5)
    assert prop_type5 == "residential_property"
    assert not is_amb5

    # 6. Ambiguous / Mixed primary terms (both agricultural and residential)
    t6 = [
        (1, "The property comprises agricultural land bearing Survey 42 and residential flat no 5."),
    ]
    prop_type6, _, _, _, _, _, is_amb6, ev6, _ = extract_property_schedule(t6)
    assert prop_type6 == "unknown"
    assert is_amb6
    assert "Mixed primary terms" in ev6

    # 7. Other property (commercial shop)
    t7 = [(1, "Schedule: Commercial shop no 12, Market Yard, Pune")]
    prop_type7, _, _, _, _, _, is_amb7, _, _ = extract_property_schedule(t7)
    assert prop_type7 == "other_property"
    assert not is_amb7


def test_extract_market_value_and_assessed_fallback():
    # 1. Explicit market value
    t1 = [(1, "The market value of property is ₹ 25,00,000/- only.")]
    val1, conf1, ev1, pg1, fb1 = extract_market_value(t1)
    assert val1 == 2500000
    assert not conf1
    assert not fb1

    # 2. Devanagari ready reckoner value
    t2 = [(1, "सदर मिळकतीचे रेडी रेकनर मूल्य: रुपये ३०,००,०००/-")]
    val2, conf2, ev2, pg2, fb2 = extract_market_value(t2)
    assert val2 == 3000000
    assert not conf2
    assert not fb2

    # 3. Conflicting market values across pages
    t3 = [
        (1, "Market Value: ₹ 20,00,000/-"),
        (2, "Government valuation / Ready Reckoner Value: ₹ 25,00,000/-")
    ]
    val3, conf3, ev3, pg3, fb3 = extract_market_value(t3)
    assert val3 is None
    assert conf3
    assert "Conflicting market values" in ev3

    # 4. Assessed value fallback only (do not use as market value)
    t4 = [(1, "Municipal tax assessed value: ₹ 10,00,000/-")]
    val4, conf4, ev4, pg4, fb4 = extract_market_value(t4)
    assert val4 is None
    assert not conf4
    assert fb4
    assert "assessed_value_fallback_only" in ev4


def test_evaluate_article_34_relationship_eligibility():
    # 1. Eligible English
    t1 = [(1, "WHEREAS the donor is the father of the donee.")]
    v1, norm1, elig1, pg1, ev1, amb1 = evaluate_article_34_relationship(t1)
    assert elig1 == "eligible"
    assert not amb1

    # 2. Eligible Marathi
    t2 = [(1, "देणार व घेणार यांचे नातेसंबंध: वडील आणि मुलगा.")]
    v2, norm2, elig2, pg2, ev2, amb2 = evaluate_article_34_relationship(t2)
    assert elig2 == "eligible"
    assert not amb2

    # 3. Ineligible relationship (uncle to nephew)
    t3 = [(1, "WHEREAS the donor is the uncle of the donee.")]
    v3, norm3, elig3, pg3, ev3, amb3 = evaluate_article_34_relationship(t3)
    assert elig3 == "not_eligible"
    assert not amb3

    # 4. Ineligible Marathi (काका आणि पुतण्या)
    t4 = [(1, "देणार घेणाऱ्याचा काका आहे.")]
    v4, norm4, elig4, pg4, ev4, amb4 = evaluate_article_34_relationship(t4)
    assert elig4 == "not_eligible"

    # 5. Vague term ("near relatives") -> ambiguous
    t5 = [(1, "The donor and donee are near relatives.")]
    v5, norm5, elig5, pg5, ev5, amb5 = evaluate_article_34_relationship(t5)
    assert elig5 == "ambiguous"
    assert amb5

    # 6. Shared surname alone without recital relationship -> unknown
    t6 = [(1, "Donor: Ramesh Patil. Donee: Suresh Patil. Gifting immovable property.")]
    v6, norm6, elig6, pg6, ev6, amb6 = evaluate_article_34_relationship(
        t6, donor_name="Ramesh Patil", donee_name="Suresh Patil"
    )
    assert elig6 == "unknown"

    # 7. Conflicting relationships across pages -> ambiguous
    t7 = [
        (1, "The donor is the brother of the donee."),
        (2, "The donor is the uncle of the donee."),
    ]
    v7, norm7, elig7, pg7, ev7, amb7 = evaluate_article_34_relationship(t7)
    assert elig7 == "ambiguous"
    assert amb7


def test_extract_stamp_duty_paid_and_challan():
    # 1. Split stamp duty and metro cess
    t1 = [(1, "Stamp Duty Paid: ₹ 200/-\nMetro Cess Paid: ₹ 25,000/-\nChallan No: MH20240506ABC123")]
    b1, m1, tot1, ch1, conf1, ev1, pg1 = extract_stamp_duty_paid(t1)
    assert b1 == 200
    assert m1 == 25000
    assert tot1 == 25200
    assert ch1 == "MH20240506ABC123"
    assert not conf1

    # 2. Total stamp duty only
    t2 = [(1, "Total Stamp Duty Paid: ₹ 25,200/-")]
    b2, m2, tot2, ch2, conf2, ev2, pg2 = extract_stamp_duty_paid(t2)
    assert tot2 == 25200
    assert not conf2

    # 3. Conflicting payments across pages
    t3 = [
        (1, "Total Duty Paid: ₹ 200/-"),
        (2, "Total Duty Paid: ₹ 25,200/-")
    ]
    b3, m3, tot3, ch3, conf3, ev3, pg3 = extract_stamp_duty_paid(t3)
    assert conf3


def test_assess_stamp_duty_article_34_calculations_and_findings():
    # Case 1: Compliant Metro Gift Deed
    # Father to Son in Pune (metro), market value 25L, expected 200 + 25,000 = 25,200, paid 25,200
    doc1 = LogicalDocument(id="doc-comp-1", ingestion_id="ing-1", start_page=1, end_page=2, category="gift_deed")
    text_p1 = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Residential flat no 402, Pune."),
        (2, "Market Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-\nChallan No: CH-12345"),
    ]
    assess1 = assess_stamp_duty_article_34(doc1, text_p1)
    assert assess1.status == "compliant"
    assert assess1.relationship_eligibility == "eligible"
    assert assess1.property_type == "residential_property"
    assert assess1.jurisdiction == "metro"
    assert assess1.market_value == 2500000
    assert assess1.base_duty == 200
    assert assess1.metro_cess_amount == 25000
    assert assess1.expected_total == 25200
    assert assess1.total_stamp_duty_paid == 25200
    assert assess1.difference == 0

    # Case 2: Short Paid Metro Gift Deed
    # Mother to Daughter in Mumbai, market value 50L, expected 200 + 50,000 = 50,200, paid only 200
    doc2 = LogicalDocument(id="doc-short-1", ingestion_id="ing-2", start_page=1, end_page=2, category="gift_deed")
    text_p2 = [
        (1, "GIFT DEED\nWHEREAS the donor is the mother of the donee.\nSchedule of Property: Residential flat at Bandra, Mumbai."),
        (2, "Ready Reckoner Value: ₹ 50,00,000/-\nStamp Duty Paid: ₹ 200/-"),
    ]
    assess2 = assess_stamp_duty_article_34(doc2, text_p2)
    assert assess2.status == "short_paid"
    assert assess2.expected_total == 50200
    assert assess2.total_stamp_duty_paid == 200
    assert assess2.difference == -50000

    # Case 3: Non-Metro / Rural Gift Deed
    # Father to Son in Gram Panchayat, market value 15L, expected 200 (metro cess 0), paid 200
    doc3 = LogicalDocument(id="doc-rural-1", ingestion_id="ing-3", start_page=1, end_page=2, category="gift_deed")
    text_p3 = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Agricultural land at Gram Panchayat village."),
        (2, "Market Value: ₹ 15,00,000/-\nTotal Stamp Duty Paid: ₹ 200/-"),
    ]
    assess3 = assess_stamp_duty_article_34(doc3, text_p3)
    assert assess3.status == "compliant"
    assert assess3.jurisdiction == "non_metro"
    assert assess3.metro_cess_amount == 0
    assert assess3.expected_total == 200
    assert assess3.difference == 0

    # Case 4: Ineligible Family Relationship (Uncle to Nephew)
    doc4 = LogicalDocument(id="doc-inelig-1", ingestion_id="ing-4", start_page=1, end_page=2, category="gift_deed")
    text_p4 = [
        (1, "GIFT DEED\nWHEREAS the donor is the uncle of the donee.\nSchedule: Residential flat, Pune."),
        (2, "Market Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess4 = assess_stamp_duty_article_34(doc4, text_p4)
    assert assess4.status == "not_applicable"
    assert assess4.relationship_eligibility == "not_eligible"

    # Case 5: Conflicting Market Values
    doc5 = LogicalDocument(id="doc-conf-1", ingestion_id="ing-5", start_page=1, end_page=2, category="gift_deed")
    text_p5 = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nResidential flat in Pune. Market Value: ₹ 20,00,000/-"),
        (2, "Government valuation: ₹ 25,00,000/-"),
    ]
    assess5 = assess_stamp_duty_article_34(doc5, text_p5)
    assert assess5.status == "requires_review"

    # Case 6: Ambiguous Property Type (Mixed Agricultural and Residential)
    doc6 = LogicalDocument(id="doc-amb-1", ingestion_id="ing-6", start_page=1, end_page=2, category="gift_deed")
    text_p6 = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Agricultural land and residential flat in Pune."),
        (2, "Market Value: ₹ 25,00,000/-\nStamp Duty Paid: ₹ 25,200/-"),
    ]
    assess6 = assess_stamp_duty_article_34(doc6, text_p6)
    assert assess6.status == "requires_review"


def test_extract_fields_persists_gift_assessment_fields():
    doc = LogicalDocument(id="doc-ef-1", ingestion_id="ing-ef-1", start_page=1, end_page=1, category="gift_deed")
    page = Page(
        id="p-ef-1",
        ingestion_id="ing-ef-1",
        page_number=1,
        digital_text=(
            "GIFT DEED\n"
            "Donor: Ramesh Sharma\n"
            "Donee: Suresh Sharma\n"
            "WHEREAS the donor is the father of the donee.\n"
            "Schedule of Property: Residential flat 101, Pune.\n"
            "Market Value: ₹ 20,00,000/-\n"
            "Stamp Duty Paid: ₹ 200/-\n"
            "Metro Cess Paid: ₹ 20,000/-\n"
            "Total Stamp Duty Paid: ₹ 20,200/-\n"
            "Challan Number: CH98765"
        ),
    )
    fields = extract_fields(doc, [page])
    field_dict = {f.field_name: f.extracted_value for f in fields}

    assert field_dict["donor_name"] == "Ramesh Sharma"
    assert field_dict["donee_name"] == "Suresh Sharma"
    assert field_dict["donor_donee_relationship"] == "Father and Son"
    assert field_dict["property_type"] == "residential_property"
    assert field_dict["market_value"] == "2000000"
    assert field_dict["stamp_duty_paid"] == "200"
    assert field_dict["metro_cess_paid"] == "20000"
    assert field_dict["total_stamp_duty_paid"] == "20200"


def test_article_34_scenario_01_father_to_son_residential_gift():
    # 1. Father-to-son residential gift: Eligible relationship, residential flat in Pune (metro),
    # market value ₹25,00,000 -> base ₹200 + metro cess ₹25,000 = expected ₹25,200.
    doc = LogicalDocument(id="doc-s1", ingestion_id="ing-s1", start_page=1, end_page=2, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Residential flat no 501, Pune."),
        (2, "Market Value: ₹ 25,00,000/-\nStamp Duty Paid: ₹ 200/-\nMetro Cess Paid: ₹ 25,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.property_type == "residential_property"
    assert assess.jurisdiction == "metro"
    assert assess.market_value == 2500000
    assert assess.base_duty == 200
    assert assess.metro_cess_amount == 25000
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 25200
    assert assess.difference == 0
    assert assess.status == "compliant"


def test_article_34_scenario_02_mother_to_daughter_agricultural_gift():
    # 2. Mother-to-daughter agricultural gift: Eligible relationship, agricultural land (शेती जमीन),
    # proper calculation verified (metro cess 0 on agricultural land).
    doc = LogicalDocument(id="doc-s2", ingestion_id="ing-s2", start_page=1, end_page=2, category="gift_deed")
    text_by_page = [
        (1, "बक्षीसपत्र\nदेणारा घेणाराची आई आहे.\nमिळकतीचे वर्णन: शेती जमीन गट नंबर 88, तालुका बारामती, जिल्हा पुणे."),
        (2, "बाजारभाव: ₹ 20,00,000/-\nमुद्रांक शुल्क भरले: ₹ 200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.property_type == "agricultural_land"
    assert assess.base_duty == 200
    assert assess.metro_cess_amount == 0
    assert assess.expected_total == 200
    assert assess.total_stamp_duty_paid == 200
    assert assess.difference == 0
    assert assess.status == "compliant"


def test_article_34_scenario_03_eligible_relationship_missing_market_value():
    # 3. Eligible relationship with missing market value: Emits missing_market_value,
    # status is not_available (never compliant).
    doc = LogicalDocument(id="doc-s3", ingestion_id="ing-s3", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Residential apartment in Mumbai.\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.market_value is None
    assert assess.status == "not_available"
    assert assess.status != "compliant"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "missing_market_value" in finding_rules


def test_article_34_scenario_04_unknown_property_type():
    # 4. Unknown property type: Schedule exists without residential/agricultural terms;
    # status is requires_review, emits unknown_property_type.
    doc = LogicalDocument(id="doc-s4", ingestion_id="ing-s4", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Open non-descript parcel situated at Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.property_type == "unknown"
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "unknown_property_type" in finding_rules


def test_article_34_scenario_05_survey_number_alone_guard():
    # 5. Survey number alone guard: Schedule has Survey Number 104/2 without primary agricultural terms
    # -> must NOT classify as agricultural_land; classifies as unknown and emits unknown_property_type.
    doc = LogicalDocument(id="doc-s5", ingestion_id="ing-s5", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the mother of the donee.\nSchedule of Property: Survey Number 104/2 at Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.property_type != "agricultural_land"
    assert assess.property_type == "unknown"
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "unknown_property_type" in finding_rules


def test_article_34_scenario_06_ambiguous_residential_agricultural_wording():
    # 6. Ambiguous residential/agricultural wording: Schedule mentions both residential flat and agricultural farm land
    # -> status is requires_review, emits ambiguous_property_type.
    doc = LogicalDocument(id="doc-s6", ingestion_id="ing-s6", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Residential flat and agricultural land at Pune.\nMarket Value: ₹ 25,00,000/-\nStamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "ambiguous_property_type" in finding_rules


def test_article_34_scenario_07_non_eligible_relationship():
    # 7. Non-eligible relationship: Explicit uncle-to-nephew or cousin gift
    # -> relationship_eligibility == "not_eligible", emits relationship_not_eligible_article_34.
    doc = LogicalDocument(id="doc-s7", ingestion_id="ing-s7", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the uncle of the donee.\nSchedule of Property: Residential flat, Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "not_eligible"
    assert assess.status == "not_applicable"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "relationship_not_eligible_article_34" in finding_rules


def test_article_34_scenario_08_missing_relationship_shared_surname_guard():
    # 8. Missing relationship (with shared surname guard): Donor 'Ramesh Patil' and Donee 'Suresh Patil'
    # with no relationship in recital -> surname does NOT infer eligibility;
    # relationship_eligibility == "unknown", emits missing_donor_donee_relationship.
    doc = LogicalDocument(id="doc-s8", ingestion_id="ing-s8", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nDonor: Ramesh Patil\nDonee: Suresh Patil\nSchedule: Residential flat in Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "unknown"
    assert assess.status == "not_available"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "missing_donor_donee_relationship" in finding_rules


def test_article_34_scenario_09_missing_actual_stamp_duty():
    # 9. Missing actual stamp duty: Eligible gift with market value ₹25,00,000 but no stamp duty paid in text
    # -> status is not_available, emits missing_stamp_duty_paid.
    doc = LogicalDocument(id="doc-s9", ingestion_id="ing-s9", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Residential flat in Pune.\nMarket Value: ₹ 25,00,000/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.total_stamp_duty_paid is None
    assert assess.status == "not_available"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "missing_stamp_duty_paid" in finding_rules


def test_article_34_scenario_10_exact_compliant_payment():
    # 10. Exact compliant payment: Expected ₹25,200, Paid ₹25,200 -> status == "compliant", difference == 0.
    doc = LogicalDocument(id="doc-s10", ingestion_id="ing-s10", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the son of the donee.\nSchedule: Residential premises at Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 25200
    assert assess.difference == 0
    assert assess.status == "compliant"


def test_article_34_scenario_11_short_payment():
    # 11. Short payment: Market value ₹25,00,000, Expected ₹25,200, Paid ₹20,000
    # -> status == "short_paid", difference == -5200, emits stamp_duty_short_paid.
    doc = LogicalDocument(id="doc-s11", ingestion_id="ing-s11", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Residential flat in Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 20,000/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 20000
    assert assess.difference == -5200
    assert assess.status == "short_paid"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "stamp_duty_short_paid" in finding_rules


def test_article_34_scenario_12_extra_payment():
    # 12. Extra payment: Expected ₹25,200, Paid ₹30,000 -> status == "compliant", difference == 4800,
    # no stamp_duty_short_paid finding.
    doc = LogicalDocument(id="doc-s12", ingestion_id="ing-s12", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Residential flat in Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 30,000/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 30000
    assert assess.difference == 4800
    assert assess.status == "compliant"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "stamp_duty_short_paid" not in finding_rules


def test_article_34_scenario_13_devanagari_hindi_ocr_text():
    # 13. Devanagari (Hindi) OCR text: बाजार मूल्य: ₹ २५,००,०००, मुद्रांक शुल्क: ₹ २५,२००, निवासी फ्लैट, पिता और पुत्र
    # -> parsed, normalized, and assessed accurately.
    doc = LogicalDocument(id="doc-s13", ingestion_id="ing-s13", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "दानपत्र\nदाता और आदाता का संबंध: पिता और पुत्र.\nसंपत्ति विवरण: निवासी फ्लैट, पुणे.\nबाजार मूल्य: ₹ २५,००,०००\nमुद्रांक शुल्क: ₹ २५,२००"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.property_type == "residential_property"
    assert assess.market_value == 2500000
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 25200
    assert assess.difference == 0
    assert assess.status == "compliant"


def test_article_34_scenario_14_marathi_ocr_text():
    # 14. Marathi OCR text: शेत जमीन, बाजारभाव रु. १५,००,०००/-, मुद्रांक शुल्क रु. १५,२००/-, वडील आणि मुलगा
    # -> parsed, normalized, and assessed accurately.
    doc = LogicalDocument(id="doc-s14", ingestion_id="ing-s14", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "बक्षीसपत्र\nदेणार व घेणार यांचे नातेसंबंध: वडील आणि मुलगा.\nपरिशिष्ट मिळकतीचे वर्णन: शेत जमीन बारामती.\nबाजारभाव रु. १५,००,०००/-\nमुद्रांक शुल्क रु. १५,२००/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.property_type == "agricultural_land"
    assert assess.market_value == 1500000
    assert assess.expected_total == 200
    assert assess.total_stamp_duty_paid == 15200
    assert assess.difference == 15000
    assert assess.status == "compliant"


def test_article_34_scenario_15_indian_number_formatting_variations():
    # 15. Indian-number formatting variations: Verify ₹25,00,000, Rs. 25,00,000, INR 2500000,
    # and Devanagari digits all normalize to 2500000.
    assert parse_indian_currency_amount("₹25,00,000") == 2500000
    assert parse_indian_currency_amount("Rs. 25,00,000") == 2500000
    assert parse_indian_currency_amount("INR 2500000") == 2500000
    assert parse_indian_currency_amount("₹ २५,००,०००") == 2500000
    assert parse_indian_currency_amount("रु. १५,००,०००/-") == 1500000
    assert parse_indian_currency_amount("रुपये २५,००,०००") == 2500000
    assert parse_indian_currency_amount("INR 25,00,000.00") == 2500000
    assert parse_indian_currency_amount("₹ 25,00,000/-") == 2500000


def test_article_34_scenario_16_metro_jurisdiction_unknown():
    # 16. Metro jurisdiction unknown: Deed only says 'Maharashtra' without a recognizable city/district
    # -> status == "requires_review", emits metro_cess_jurisdiction_unknown.
    doc = LogicalDocument(id="doc-s16", ingestion_id="ing-s16", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property: Residential flat situated in Maharashtra state.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.jurisdiction == "unknown"
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "metro_cess_jurisdiction_unknown" in finding_rules


def test_article_34_scenario_17_conflicting_market_values():
    # 17. Conflicting market values: Page 2 states Market value: ₹20,00,000 and Page 5 states Market value: ₹30,00,000
    # -> preserves both candidates in evidence, sets status == "requires_review", emits stamp_duty_calculation_requires_review.
    doc = LogicalDocument(id="doc-s17", ingestion_id="ing-s17", start_page=1, end_page=2, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule: Residential apartment at Pune.\nMarket value: ₹20,00,000/-"),
        (2, "Government Market value: ₹30,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.status == "requires_review"
    assert "20,00,000" in assess.evidence
    assert "30,00,000" in assess.evidence
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "stamp_duty_calculation_requires_review" in finding_rules


def test_article_34_scenario_18_conflicting_relationship_statements():
    # 18. Conflicting relationship statements: Page 1 says real brother and Page 3 says cousin brother
    # -> relationship_eligibility == "ambiguous", emits ambiguous_donor_donee_relationship, status == "requires_review".
    doc = LogicalDocument(id="doc-s18", ingestion_id="ing-s18", start_page=1, end_page=2, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donee is the real brother of the donor.\nSchedule: Residential flat at Pune."),
        (2, "WHEREAS the donee is also referenced as cousin brother of the donor.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "ambiguous"
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "ambiguous_donor_donee_relationship" in finding_rules


def test_article_34_scenario_19_donor_donee_names_ocr_errors():
    # 19. Deed where donor/donee names have OCR errors: Donor/donee names contain OCR noise
    # while recital relationship is explicit; verify extraction and review flags.
    doc = LogicalDocument(id="doc-s19", ingestion_id="ing-s19", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "G!FT DEED\nDon0r: R@m3sh Sh4rma\nD0nee: Su-resh Sh4rma\nWHEREAS the donor is the father of the donee.\nSchedule: Residential flat at Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.relationship_eligibility == "eligible"
    assert assess.property_type == "residential_property"
    assert assess.status == "compliant"
    assert assess.expected_total == 25200


def test_article_34_scenario_20_separate_challan_page():
    # 20. Deed with a separate challan page: Property schedule & market value on Page 3,
    # stamp duty paid & challan number on Page 12 -> both pages appear in source_pages and calculation succeeds.
    doc = LogicalDocument(id="doc-s20", ingestion_id="ing-s20", start_page=1, end_page=12, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee."),
        (3, "Schedule of Property: Residential flat at Mumbai.\nMarket Value: ₹ 25,00,000/-"),
        (12, "MAHARASHTRA e-GRAS TREASURY CHALLAN\nTotal Stamp Duty Paid: ₹ 25,200/-\nChallan No: MH00987654321"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.expected_total == 25200
    assert assess.total_stamp_duty_paid == 25200
    assert assess.challan_number == "MH00987654321"
    assert assess.status == "compliant"
    pages = [int(p) for p in assess.source_pages.split(",")]
    assert 3 in pages
    assert 12 in pages


def test_article_34_scenario_21_multiple_properties():
    # 21. Deed with multiple properties: Schedule lists Property 1 and Property 2 with mixed types or multiple values
    # -> sets status == "requires_review" and emits ambiguous_property_type / stamp_duty_calculation_requires_review.
    doc = LogicalDocument(id="doc-s21", ingestion_id="ing-s21", start_page=1, end_page=1, category="gift_deed")
    text_by_page = [
        (1, "GIFT DEED\nWHEREAS the donor is the father of the donee.\nSchedule of Property:\nProperty 1: Agricultural farm land.\nProperty 2: Residential flat at Pune.\nMarket Value: ₹ 25,00,000/-\nTotal Stamp Duty Paid: ₹ 25,200/-"),
    ]
    assess = assess_stamp_duty_article_34(doc, text_by_page)
    assert assess.status == "requires_review"
    finding_rules = [f.rule_id for f in assess.findings_list]
    assert "ambiguous_property_type" in finding_rules or "stamp_duty_calculation_requires_review" in finding_rules


