import logging
import re
import tempfile
import io
import json
import shutil
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps
from pypdf.errors import PdfReadError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import (
    ARTICLE_34_ELIGIBLE_RELATIONSHIPS,
    ARTICLE_34_INELIGIBLE_RELATIONSHIPS,
    ARTICLE_34_NORMALIZED_ELIGIBLE_RELATIONSHIPS,
    ARTICLE_34_RELATIONSHIP_LEXICAL_MAPPINGS,
    METRO_CESS_JURISDICTIONS,
    NON_METRO_INDICATORS,
    STAMP_DUTY_BASE_AMOUNT,
    STAMP_DUTY_EFFECTIVE_FROM,
    STAMP_DUTY_LEGAL_REFERENCE,
    STAMP_DUTY_METRO_CESS_RATE,
    STAMP_DUTY_RULE_VERSION,
    get_settings,
)
from .ocr import OcrEngineUnavailableError, run_ocr
from .pdf_utils import LocalImageBundle, LocalPdf
from .models import (
    CasePerson,
    CaseProperty,
    CaseRecord,
    DocumentPerson,
    DocumentProperty,
    ExtractedField,
    Ingestion,
    LogicalDocument,
    Page,
    Person,
    ProcessingJob,
    PropertyRecord,
    StampDutyAssessment,
    ValidationFinding,
)

logger = logging.getLogger(__name__)

CATEGORIES = {
    "gift_deed": (
        "gift deed", "दानपत्र", "बक्षीसपत्र", "bakshispatra",
        "indenture of gift", "deed of gift", "gift of property",
        "the donor", "the donee", "executing donor", "claiming donee",
        "donors", "donees", "दानकर्ता", "दानग्रहीता",
        "irrevocably gift", "doth hereby gift",
    ),
    "valuation": ("mulyankan", "मूल्यांकन", "valuation", "assessment"),
    "receipt": ("processing charges", "handling charges", "receipt", "पावती"),
    "identity_document": ("aadhaar", "आधार", "unique identification", "identity document", "identity verification record", "पहचान विवरण"),
    "challan": ("challan", "चालान", "stamp duty", "e-challan"),
    "condominium": ("condominium", "society", "अपार्टमेंट", "संस्था"),
    # survey_record: English + Hindi + Marathi land-record terms
    "survey_record": (
        "survey number", "land record", "survey no",
        # Hindi
        "गट क्रमांक",
        # Marathi — 7/12 and crop-area registers
        "गा. नं.", "गा.नं.", "ग.नं.",  # Gat / village number
        "जमीन करणाराचे नाव",             # Name of landholder
        "पिकाखाली क्षेत्र",              # Area under crops
        "मिश्र पिकाचे",                  # Mixed crops
        "अमिश्र पिकाचे",                 # Non-mixed crops
        "हंगाम", "खरीप", "रब्बी",       # Season / Kharif / Rabi
        "जलसिंचन", "अजल सिंचन",         # Irrigated / unirrigated
        "तलाठी", "कामगार तलाठी",        # Revenue officer
        "सातबारा", "७/१२",               # 7/12 extract
        "फेरफार", "हक्क",               # Mutation / rights
        "भोगवटादार",                     # Occupant class
    ),
    # Dedicated category for Marathi administrative / revenue records
    "marathi_land_record": (
        "गा. नं.", "गा.नं.", "ग.नं.",
        "जमीन करणाराचे नाव",
        "पिकाखाली क्षेत्र",
        "मिश्र पिकाचे",
        "अमिश्र पिकाचे",
        "तलाठी", "कामगार तलाठी",
        "सातबारा", "७/१२",
        "हंगाम",
        "भोगवटादार",
        "फेरफार",
    ),
    "official_letter": ("to whomsoever", "विषय:", "subject:", "कार्यालय", "dear sir"),
    "registration": ("sub-registrar", "registration no", "registration ref", "registration record", "पंजीकरण विवरण", "दस्त नोंदणी", "reg. no"),
    "property_tax": ("property tax", "tax receipt", "संपत्ति कर"),
    "tax_invoice": ("tax invoice", "gst invoice", "invoice serial number", "invoice currency code", "gstin:", "invoice value"),
}

FIELD_PATTERNS = {
    "registration_number": r"(?:\bregistration\s+(?:no\.?|number|ref(?:erence)?)\b|(?<!\w)registration(?=\s*[:#-])|reg\.?\s*no\.?|नोंदणी\s*(?:क्रमांक|क्र))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    "document_date": r"(?:date|दिनांक|तारीख)\s*[:#-]?\s*([0-9०-९./-]{6,})",
    # Survey/Gat number — English, Hindi and Marathi (गा.नं. / ग.नं.)
    "survey_number": r"(?:\bsurvey\s+(?:no\.?|number)\b|(?<!\w)survey(?=\s*[:#-])|गट\s*क्रमांक|गा\.?\s*नं\.?|ग\.\s*नं\.?)\s*[:#-]?\s*([A-Z0-9०-९][A-Z0-9०-९/-]*)",
    "property_id": r"(?:property\s*(?:id|identifier)|property(?=\s*[:#-])|संपत्ति\s*आईडी)\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    # Property area — extended for Marathi क्षेत्र / हेक्टर / are
    "property_area": r"(?:property\s*area|area|क्षेत्रफल|संपत्ति\s*का\s*क्षेत्रफल|क्षेत्र|एकूण\s*क्षेत्र)\s*(?:/\s*(?:क्षेत्रफल|property\s*area))?\s*[:#-]?\s*([0-9०-९,.]+\s*(?:sq\.?\s*(?:ft|m(?:et(?:er|re))?s?)|वर्ग\s*(?:फुट|मीटर)|हेक्टर|are|R|H))",
    "valuation_reference": r"(?:valuation\s*(?:ref(?:erence)?|no\.?|number)|मूल्यांकन\s*(?:संदर्भ|क्रमांक))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    "assessed_value": r"(?:assessed\s*value|valuation\s*amount|मूल्यांकन\s*राशि)\s*[:#-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.\d{1,2})?)",
    "receipt_number": r"(?:receipt\s*(?:no\.?|number)|रसीद\s*(?:क्रमांक|संख्या))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    "identity_number": r"(?:id\s*(?:no\.?|number)|identifier|पहचान\s*(?:संख्या|क्रमांक))\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    "amount": r"(?:amount|value|रक्कम|मूल्य)\s*[:#-]?\s*(?:INR|Rs\.?|₹)?\s*([0-9,]+(?:\.\d{1,2})?)",
    "challan_number": r"(?:challan|चालान)\s*(?:no\.?|number|क्रमांक)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9/-]{2,})",
    # Marathi land-record specific fields
    "landholder_name": r"(?:जमीन\s*करणाराचे\s*नाव|भोगवटादार)\s*[:#-]?\s*([\u0900-\u097F\w .'-]{3,80})",
    "crop_area": r"(?:पिकाखाली\s*क्षेत्र|मिश्र\s*पिकाचे\s*एकूण\s*क्षेत्र|अमिश्र\s*पिकाचे\s*क्षेत्र)\s*[:#-]?\s*([0-9०-९,.]+)",
    "season_year": r"(?:हंगाम|वर्ष)\s*[:#-]?\s*([0-9०-९]{4}(?:[/-][0-9०-९]{2,4})?)",
    "talathi_name": r"(?:कामगार\s*तलाठी|तलाठी\s*सही|गाव\s*कामगार\s*तलाठी\s*सही)\s*[:#-]?\s*([\u0900-\u097F\w .'-]{3,80})",
}

INVOICE_FIELD_LABELS: dict[str, tuple[tuple[str, ...], int]] = {
    "supplier_name": (("tax invoice",), 0),
    "invoice_number": (("invoice serial number", "invoice number", "invoice no"), 0),
    "invoice_date": (("invoice date",), 0),
    "invoice_currency": (("invoice currency code", "currency code"), 0),
    "invoice_irn": (("irn",), 0),
    "supplier_gstin": (("gstin",), 0),
    "customer_gstin": (("gstin",), 1),
    "consignee_gstin": (("gstin",), 2),
    "supplier_phone": (("tel-no", "telephone number", "phone number"), 0),
    "supplier_cin": (("cin", "corporate identity number"), 0),
    "supplier_registered_address": (("registered office",), 0),
    "reverse_charge": (("tax is payable on reverse charge", "reverse charge"), 0),
    "transportation_mode": (("transportation mode", "mode of transport"), 0),
    "vehicle_number": (("veh. no", "vehicle number", "vehicle no"), 0),
    "date_time_of_supply": (("date & time of supply", "date and time of supply"), 0),
    "place_of_supply": (("place of supply",), 0),
    "customer_name": (("name",), 0),
    "consignee_name": (("name",), 1),
    "customer_address": (("address",), 0),
    "consignee_address": (("address",), 1),
    "customer_state": (("state name",), 0),
    "consignee_state": (("state name",), 1),
    "customer_state_code": (("state code",), 0),
    "consignee_state_code": (("state code",), 1),
    "order_reference": (("order reference",), 0),
    "line_item_description": (("description of goods", "description"), 0),
    "hsn_sac_code": (("hsn code", "hsn/sac"), 0),
    "invoice_line_item_raw": (("description of goods", "description"), 0),
    "quantity": (("qty", "quantity"), 0),
    "unit_of_measure": (("uom", "unit of measure"), 0),
    "unit_rate": (("rate",), 0),
    "discount": (("disc", "discount"), 0),
    "taxable_value": (("taxable value",), 0),
    "cgst_rate": (("cgst rate",), 0),
    "cgst_amount": (("cgst amount",), 0),
    "sgst_rate": (("sgst rate",), 0),
    "sgst_amount": (("sgst amount",), 0),
    "igst_rate": (("igst rate",), 0),
    "igst_amount": (("igst amount",), 0),
    "invoice_total": (("invoice total",), 0),
    "amount_in_words": (("invoice value (in words)", "amount in words"), 0),
    "freight_charges": (("freight charges",), 0),
    "loading_packaging_charges": (("loading and packaging charges",), 0),
    "insurance_charges": (("insurance charges",), 0),
    "other_charges": (("other charges",), 0),
    "terms_and_conditions": (("terms & conditions of sale", "terms and conditions of sale"), 0),
    "authorized_signatory": (("authorized signatory", "authorised signatory"), 0),
}


DONOR_DONEE_RELATIONSHIP_TERMS: dict[str, tuple[str, str]] = {
    # English
    "parents to son": ("Parents to Son", "parents and son"),
    "parents and son": ("Parents to Son", "parents and son"),
    "parents-son": ("Parents to Son", "parents and son"),
    "parent to son": ("Parent and Son", "parent and son"),
    "parent and son": ("Parent and Son", "parent and son"),
    "parents to daughter": ("Parents to Daughter", "parents and daughter"),
    "parents and daughter": ("Parents to Daughter", "parents and daughter"),
    "parents-daughter": ("Parents to Daughter", "parents and daughter"),
    "parent to daughter": ("Parent and Daughter", "parent and daughter"),
    "parent and daughter": ("Parent and Daughter", "parent and daughter"),
    "parents to children": ("Parents and Children", "parents and children"),
    "parents and children": ("Parents and Children", "parents and children"),
    "father to son": ("Father and Son", "father and son"),
    "father to daughter": ("Father and Daughter", "father and daughter"),
    "mother to son": ("Mother and Son", "mother and son"),
    "mother to daughter": ("Mother and Daughter", "mother and daughter"),
    "brother to brother": ("Brothers", "brothers"),
    "brother to sister": ("Brother and Sister", "brother and sister"),
    "sister to brother": ("Brother and Sister", "brother and sister"),
    "sister to sister": ("Sisters", "sisters"),
    "uncle to nephew": ("Uncle and Nephew", "uncle and nephew"),
    "uncle to niece": ("Uncle and Niece", "uncle and niece"),
    "paternal uncle to nephew": ("Paternal Uncle and Nephew", "uncle and nephew"),
    "paternal aunt to nephew": ("Paternal Aunt and Nephew", "aunt and nephew"),
    "husband to wife": ("Husband and Wife", "husband and wife"),
    "wife to husband": ("Husband and Wife", "husband and wife"),
    "parents": ("Parents and Children", "parents and children"),
    "parent": ("Parent and Child", "parent and child"),
    "father and son": ("Father and Son", "father and son"),
    "father and daughter": ("Father and Daughter", "father and daughter"),
    "mother and son": ("Mother and Son", "mother and son"),
    "mother and daughter": ("Mother and Daughter", "mother and daughter"),
    "brother and sister": ("Brother and Sister", "brother and sister"),
    "brothers": ("Brothers", "brothers"),
    "brother and brother": ("Brothers", "brothers"),
    "sisters": ("Sisters", "sisters"),
    "sister and sister": ("Sisters", "sisters"),
    "husband and wife": ("Husband and Wife", "husband and wife"),
    "father-son": ("Father and Son", "father and son"),
    "father-daughter": ("Father and Daughter", "father and daughter"),
    "mother-son": ("Mother and Son", "mother and son"),
    "mother-daughter": ("Mother and Daughter", "mother and daughter"),
    "brother-sister": ("Brother and Sister", "brother and sister"),
    "husband-wife": ("Husband and Wife", "husband and wife"),
    "father": ("Father and Son", "father and son"),
    "mother": ("Mother and Son", "mother and son"),
    "son": ("Father and Son", "father and son"),
    "daughter": ("Father and Daughter", "father and daughter"),
    "brother": ("Brothers", "brothers"),
    "sister": ("Brother and Sister", "brother and sister"),
    "husband": ("Husband and Wife", "husband and wife"),
    "wife": ("Husband and Wife", "husband and wife"),
    "grandfather and grandson": ("Grandfather and Grandson", "grandfather and grandson"),
    "grandfather and granddaughter": ("Grandfather and Granddaughter", "grandfather and granddaughter"),
    "grandmother and grandson": ("Grandmother and Grandson", "grandmother and grandson"),
    "grandson": ("Grandfather and Grandson", "grandfather and grandson"),
    "granddaughter": ("Grandfather and Granddaughter", "grandfather and granddaughter"),
    "nephew": ("Uncle and Nephew", "uncle and nephew"),
    "niece": ("Uncle and Niece", "uncle and niece"),
    "uncle": ("Uncle and Nephew", "uncle and nephew"),
    "aunt": ("Aunt and Nephew", "aunt and nephew"),
    "cousin": ("Cousin", "cousin"),
    "cousin brother": ("Cousin", "cousin"),
    "cousin sister": ("Cousin", "cousin"),
    "real brother": ("Brothers", "brothers"),
    "real sister": ("Brother and Sister", "brother and sister"),
    "daughter-in-law": ("Parent-in-law and Daughter-in-law", "daughter-in-law"),
    "son-in-law": ("Parent-in-law and Son-in-law", "son-in-law"),

    # Hindi
    "माता-पिता और पुत्र": ("माता-पिता और पुत्र", "parents and son"),
    "माता-पिता से पुत्र": ("माता-पिता और पुत्र", "parents and son"),
    "माता-पिता व पुत्र": ("माता-पिता और पुत्र", "parents and son"),
    "माता-पिता और पुत्री": ("माता-पिता और पुत्री", "parents and daughter"),
    "माता-पिता से पुत्री": ("माता-पिता और पुत्री", "parents and daughter"),
    "माता-पिता व पुत्री": ("माता-पिता और पुत्री", "parents and daughter"),
    "पिता से पुत्र": ("पिता और पुत्र", "father and son"),
    "माता से पुत्र": ("माता और पुत्र", "mother and son"),
    "पिता से पुत्री": ("पिता और पुत्री", "father and daughter"),
    "माता से पुत्री": ("माता और पुत्री", "mother and daughter"),
    "माता-पिता": ("माता-पिता और संतान", "parents and children"),
    "पिता और पुत्र": ("पिता और पुत्र", "father and son"),
    "पिता-पुत्र": ("पिता और पुत्र", "father and son"),
    "पिता व पुत्र": ("पिता और पुत्र", "father and son"),
    "पिता और पुत्री": ("पिता और पुत्री", "father and daughter"),
    "पिता-पुत्री": ("पिता और पुत्री", "father and daughter"),
    "पिता व पुत्री": ("पिता और पुत्री", "father and daughter"),
    "माता और पुत्र": ("माता और पुत्र", "mother and son"),
    "माता-पुत्र": ("माता और पुत्र", "mother and son"),
    "माता और पुत्री": ("माता और पुत्री", "mother and daughter"),
    "माता-पुत्री": ("माता और पुत्री", "mother and daughter"),
    "भाई और बहन": ("भाई और बहन", "brother and sister"),
    "भाई-बहन": ("भाई और बहन", "brother and sister"),
    "सगे भाई": ("सगे भाई", "brothers"),
    "पति और पत्नी": ("पति और पत्नी", "husband and wife"),
    "पति-पत्नी": ("पति और पत्नी", "husband and wife"),
    "पिता": ("पिता और पुत्र", "father and son"),
    "माता": ("माता और पुत्र", "mother and son"),
    "पुत्र": ("पिता और पुत्र", "father and son"),
    "बेटा": ("पिता और पुत्र", "father and son"),
    "सुपुत्र": ("पिता और पुत्र", "father and son"),
    "पुत्री": ("पिता और पुत्री", "father and daughter"),
    "बेटी": ("पिता और पुत्री", "father and daughter"),
    "सुपुत्री": ("पिता और पुत्री", "father and daughter"),
    "भाई": ("सगे भाई", "brothers"),
    "बहन": ("भाई और बहन", "brother and sister"),
    "पत्नी": ("पति और पत्नी", "husband and wife"),
    "पति": ("पति और पत्नी", "husband and wife"),
    "दादा": ("दादा और पोता", "grandfather and grandson"),
    "पोता": ("दादा और पोता", "grandfather and grandson"),
    "पोती": ("दादा और पोती", "grandfather and granddaughter"),
    "भतीजा": ("चाचा और भतीजा", "uncle and nephew"),
    "बहू": ("ससुर और बहू", "daughter-in-law"),

    # Marathi
    "आई-वडील आणि मुलगा": ("आई-वडील आणि मुलगा", "parents and son"),
    "आई-वडील व मुलगा": ("आई-वडील आणि मुलगा", "parents and son"),
    "आई-वडील ते मुलगा": ("आई-वडील आणि मुलगा", "parents and son"),
    "आई-वडील आणि मुलगी": ("आई-वडील आणि मुलगी", "parents and daughter"),
    "आई-वडील व मुलगी": ("आई-वडील आणि मुलगी", "parents and daughter"),
    "आई-वडील ते मुलगी": ("आई-वडील आणि मुलगी", "parents and daughter"),
    "वडील ते मुलगा": ("वडील आणि मुलगा", "father and son"),
    "आई ते मुलगा": ("आई आणि मुलगा", "mother and son"),
    "वडील ते मुलगी": ("वडील आणि मुलगी", "father and daughter"),
    "आई ते मुलगी": ("आई आणि मुलगी", "mother and daughter"),
    "आई-वडील": ("आई-वडील आणि मुले", "parents and children"),
    "वडील आणि मुलगा": ("वडील आणि मुलगा", "father and son"),
    "वडील व मुलगा": ("वडील आणि मुलगा", "father and son"),
    "वडील-मुलगा": ("वडील आणि मुलगा", "father and son"),
    "वडील आणि मुलगी": ("वडील आणि मुलगी", "father and daughter"),
    "वडील व मुलगी": ("वडील आणि मुलगी", "father and daughter"),
    "वडील-मुलगी": ("वडील आणि मुलगी", "father and daughter"),
    "आई आणि मुलगा": ("आई आणि मुलगा", "mother and son"),
    "आई व मुलगा": ("आई आणि मुलगा", "mother and son"),
    "आई-मुलगा": ("आई आणि मुलगा", "mother and son"),
    "आई आणि मुलगी": ("आई आणि मुलगी", "mother and daughter"),
    "आई व मुलगी": ("आई आणि मुलगी", "mother and daughter"),
    "आई-मुलगी": ("आई आणि मुलगी", "mother and daughter"),
    "भाऊ आणि बहीण": ("भाऊ आणि बहीण", "brother and sister"),
    "भाऊ-बहीण": ("भाऊ आणि बहीण", "brother and sister"),
    "सख्खे बंधू": ("सख्खे बंधू", "brothers"),
    "पती आणि पत्नी": ("पती आणि पत्नी", "husband and wife"),
    "पती-पत्नी": ("पती आणि पत्नी", "husband and wife"),
    "वडील": ("वडील आणि मुलगा", "father and son"),
    "आई": ("आई आणि मुलगा", "mother and son"),
    "मुलगा": ("वडील आणि मुलगा", "father and son"),
    "चिरंजीव": ("वडील आणि मुलगा", "father and son"),
    "मुलगी": ("वडील आणि मुलगी", "father and daughter"),
    "कन्या": ("वडील आणि मुलगी", "father and daughter"),
    "सुकन्या": ("वडील आणि मुलगी", "father and daughter"),
    "भाऊ": ("सख्खे बंधू", "brothers"),
    "बंधू": ("सख्खे बंधू", "brothers"),
    "बहीण": ("भाऊ आणि बहीण", "brother and sister"),
    "पत्नी": ("पती आणि पत्नी", "husband and wife"),
    "पती": ("पती आणि पत्नी", "husband and wife"),
    "आजोबा": ("आजोबा आणि नातू", "grandfather and grandson"),
    "नातू": ("आजोबा आणि नातू", "grandfather and grandson"),
    "नात": ("आजोबा आणि नात", "grandfather and granddaughter"),
    "काका": ("काका आणि पुतण्या", "uncle and nephew"),
    "मामा": ("मामा आणि भाचा", "uncle and nephew"),
    "चाचा": ("चाचा और भतीजा", "uncle and nephew"),
    "पुतण्या": ("काका आणि पुतण्या", "uncle and nephew"),
    "भाचा": ("मामा आणि भाचा", "uncle and nephew"),
    "भतीजा": ("चाचा और भतीजा", "uncle and nephew"),
    "सून": ("सासरा आणि सून", "daughter-in-law"),
}

DONOR_DONEE_EXPLICIT_PATTERNS = [
    # Explicit header / field label pattern (e.g. "RELATIONSHIP: Parents to Son" or "RELATION: Father and Son")
    r"\b(?:relationship|relation|related\s+as)\s*(?:between\s+(?:the\s+)?(?:donor\s+and\s+donee|parties))?\s*[:：is-]*\s*([a-zA-Z\s/-]+?)(?=\s*(?:[;.\n]|\bwhereas\b|\band\b|\bnow\b|$))",
    # Hindi explicit label
    r"(?:संबंध|रिश्ता|नाता|नातेसंबंध)\s*[:：-]\s*([\u0900-\u097F\s/-]+?)(?=\s*(?:[;।\n]|$))",
    # Marathi explicit label
    r"(?:नातेसंबंध|नाते|संबंध)\s*[:：-]\s*([\u0900-\u097F\s/-]+?)(?=\s*(?:[;।\n]|$))",

    # Mutual / explicit relationship phrase (English)
    r"\b(?:related\s+as|relationship\s*(?:between\s+(?:the\s+)?(?:donor\s+and\s+donee|parties))?\s*[:：is]*)\s*(father\s+and\s+son|father\s+and\s+daughter|mother\s+and\s+son|mother\s+and\s+daughter|brother\s+and\s+sister|brothers|sisters|husband\s+and\s+wife|father-son|father-daughter|mother-son|mother-daughter|brother-sister|husband-wife)\b",
    r"\b(?:parties\s+(?:hereto\s+)?(?:are\s+)?(?:related\s+as\s+)?|are\s+)(brothers|sisters|husband\s+and\s+wife|father\s+and\s+son)\b",
    r"\b(?:the\s+)?donor\s+(?:is|being|referenced\s+as|is\s+also\s+referenced\s+as)\s+(?:the\s+)?(?:real\s+|full\s+|blood\s+|elder\s+|younger\s+)?(cousin\s+brother|cousin\s+sister|cousin|father|mother|son|daughter|brother|sister|husband|wife|grandfather|grandmother|grandson|granddaughter|uncle|aunt|nephew|niece)\s+of\s+(?:the\s+)?(?:donee|recipient)\b",
    r"\b(?:the\s+)?donee\s+(?:is|being|referenced\s+as|is\s+also\s+referenced\s+as)\s+(?:the\s+)?(?:real\s+|full\s+|blood\s+|elder\s+|younger\s+)?(cousin\s+brother|cousin\s+sister|cousin|father|mother|son|daughter|brother|sister|wife|husband|grandfather|grandmother|grandson|granddaughter|nephew|niece)\s+of\s+(?:the\s+)?(?:donor|transferor)\b",
    r"\bout\s+of\s+(?:natural\s+)?love\s+and\s+affection\s+(?:for|towards)\s+(?:(?:his|her|the\s+donor'?s)\s+)?(son|daughter|brother|sister|wife|husband|mother|father|grandson|granddaughter|nephew|niece)\b",
    r"\b(?:donor|donee)\s*[:：.-]\s*[^,\n()]+\((father|mother|son|daughter|brother|sister|husband|wife|grandson|granddaughter)\)",

    # Hindi
    r"(?:दाता|देने\s*वाले?)\s+(?:और|व)\s+(?:प्राप्तकर्ता|लेने\s*वाले?)\s*(?:के\s+बीच\s+)?\s*(पिता\s*(?:और|व|-)\s*पुत्र|पिता\s*(?:और|व|-)\s*पुत्री|माता\s*(?:और|व|-)\s*पुत्र|माता\s*(?:और|व|-)\s*पुत्री|भाई\s*(?:और|व|-)\s*बहन|सगे\s*भाई|पति\s*(?:और|व|-)\s*पत्नी)",
    r"(?:दाता|देने\s*वाले?)\s+(?:और|व)\s+(?:प्राप्तकर्ता|लेने\s*वाले?)\s*(?:के\s+बीच\s+)?(?:का\s+)?(?:संबंध|रिश्ता|नाता|नातेसंबंध)\s*[:：-]?\s*(पिता\s*(?:और|व|-)\s*पुत्र|पिता\s*(?:और|व|-)\s*पुत्री|माता\s*(?:और|व|-)\s*पुत्र|माता\s*(?:और|व|-)\s*पुत्री|भाई\s*(?:और|व|-)\s*बहन|सगे\s*भाई|पति\s*(?:और|व|-)\s*पत्नी)",
    r"(?:संबंध|रिश्ता|नाता)\s*[:：-]?\s*(पिता\s*(?:और|व|-)\s*पुत्र|पिता\s*(?:और|व|-)\s*पुत्री|माता\s*(?:और|व|-)\s*पुत्र|माता\s*(?:और|व|-)\s*पुत्री|भाई\s*(?:और|व|-)\s*बहन|सगे\s*भाई|पति\s*(?:और|व|-)\s*पत्नी)",
    r"(?:दाता|देने\s*वाले?)\s+(?:प्राप्तकर्ता|लेने\s*वाले?)\s+के?\s*(पिता|माता|भाई|बहन|पति|दादा|नाना|चाचा|मामा)\s*(?:हैं?|होने\s*के\s*नाते)",
    r"(?:प्राप्तकर्ता|लेने\s*वाले?)\s+(?:दाता|देने\s*वाले?)\s+क[ाेी]?\s*(पुत्र|बेटा|पुत्री|बेटी|सुपुत्र|सुपुत्री|भाई|बहन|पत्नी|पति|पोता|पोती|नाती|नातिन)\s*(?:हैं?|होने\s*के\s*नाते)",
    r"(?:स्वाभाविक|प्राकृतिक)\s+(?:प्रेम\s*(?:और|व)\s*स्नेह|प्रेम|स्नेह)\s*(?:के\s+कारण|वश)\s+(?:अपने|अपनी)\s*(पुत्र|बेटा|पुत्री|बेटी|सुपुत्र|सुपुत्री|भाई|बहन|पत्नी|पति|पोता|पोती)",
    r"(?:दाता|प्राप्तकर्ता)\s*[:：-]?\s*[^,\n()]+\((पिता|माता|पुत्र|पुत्री|बेटा|बेटी|भाई|बहन|पति|पत्नी)\)",

    # Marathi
    r"देणार\s*(?:\(दाता\))?\s*(?:व|आणि)\s*घेणार\s*(?:\(स्वीकारणारा\))?\s*(?:यांचे|यांच्यातील|यांच्यात|यांच्या)\s*(?:नाते|नातेसंबंध|संबंध)?\s*[:：-]?\s*(वडील\s*(?:आणि|व|-)\s*मुलगा|वडील\s*(?:आणि|व|-)\s*मुलगी|आई\s*(?:आणि|व|-)\s*मुलगा|आई\s*(?:आणि|व|-)\s*मुलगी|भाऊ\s*(?:आणि|व|-)\s*बहीण|सख्खे\s*बंधू|पती\s*(?:आणि|व|-)\s*पत्नी)",
    r"देणार\s*(?:व|आणि)\s*घेणार\s*(?:हे\s+)?(सख्खे\s+बंधू|भाऊ|पती-पत्नी)",
    r"(?:नातेसंबंध|नाते|संबंध)\s*[:：-]?\s*(वडील\s*(?:आणि|व|-)\s*मुलगा|वडील\s*(?:आणि|व|-)\s*मुलगी|आई\s*(?:आणि|व|-)\s*मुलगा|आई\s*(?:आणि|व|-)\s*मुलगी|भाऊ\s*(?:आणि|व|-)\s*बहीण|सख्खे\s*बंधू|पती\s*(?:आणि|व|-)\s*पत्नी)",
    r"देणाराचे\s+घेणाऱ्याशी\s+नाते\s*[:：-]?\s*(वडील\s*(?:आणि|व|-)\s*मुलगा|वडील\s*(?:आणि|व|-)\s*मुलगी|आई\s*(?:आणि|व|-)\s*मुलगा|आई\s*(?:आणि|व|-)\s*मुलगी|भाऊ\s*(?:आणि|व|-)\s*बहीण|सख्खे\s*बंधू|पती\s*(?:आणि|व|-)\s*पत्नी)",
    r"देणार[ाी]?\s*(?:हा\s+|ही\s+)?(?:घेणाऱ्या|घेणारा)च[ाीे]?\s*(वडील|पिता|आई|माता|भाऊ|बहीण|पती|पत्नी|आजोबा|आजी|काका|मामा|चाचा|आत्या|मावशी)\s*(?:आहे|असून|असल्याने)",
    r"घेणार[ाी]?\s*(?:हा\s+|ही\s+)?(?:देणाऱ्या|देणारा)च[ाीे]?\s*(मुलगा|मुलगी|पुत्र|कन्या|चिरंजीव|सुपुत्र|सुकन्या|भाऊ|बहीण|पत्नी|पती|नातू|नात|पुतण्या|भाचा|भतीजा|भांजा)\s*(?:आहे|असून|असल्याने)",
    r"(?:नैसर्गिक|स्वाभाविक)\s+(?:प्रेम\s*(?:व|आणि)\s*जिव्हाळा|प्रेम|जिव्हाळा|लोभ|ममते)(?:पोटी|मुळे|खातर)\s+(?:आपल[ाीे]|त्यांच[ाीे])?\s*(मुलगा|मुलगी|पुत्र|कन्या|चिरंजीव|सुपुत्र|सुकन्या|भाऊ|बहीण|पत्नी|पती|नातू|नात)",
    r"देणार\s*[:：-]?\s*[^,\n()]+\((वडील|आई|भाऊ|बहीण|पती|पत्नी)\)",
    r"घेणार\s*[:：-]?\s*[^,\n()]+\((मुलगा|मुलगी|भाऊ|बहीण|पत्नी|पती)\)",
]

DONOR_DONEE_VAGUE_PATTERNS = [
    r"\b(?:near\s+relatives?|relatives?|family\s+members?|related)\b",
    r"(?:नातेवाईक|नातेसंबंधी|कुटुंबातील\s*सदस्य)",
    r"(?:रिश्तेदार|संबंधी|सगे-संबंधी|परिवार\s*के\s*सदस्य)",
]


def _extract_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.;।!?\n])\s+", text)
    sentences: list[str] = []
    for s in raw:
        cleaned = re.sub(r"\s+", " ", s).strip()
        if len(cleaned) >= 5:
            sentences.append(cleaned)
    return sentences


def extract_donor_donee_relationship(
    text_by_page: list[tuple[int, str]],
    donor_name: str | None = None,
    donee_name: str | None = None,
) -> tuple[str, str | None, str, int, bool]:
    """
    Search recital text for explicit relationship phrases near donor and donee names.
    Supports English, Hindi, and Marathi phrases.
    Does not infer relationship merely from shared surnames.
    Returns: (extracted_value, normalized_value, evidence_sentence, source_page, is_ambiguous)
    If no explicit relationship is stated, returns extracted_value='not_available'.
    """
    if isinstance(text_by_page, str):
        text_by_page = [(1, text_by_page)]
    default_page = text_by_page[0][0] if text_by_page else 1
    found_candidates: list[tuple[str, str, str, int, bool]] = []

    d_first = donor_name.split()[0].casefold() if donor_name and len(donor_name.split()) > 1 else None
    e_first = donee_name.split()[0].casefold() if donee_name and len(donee_name.split()) > 1 else None

    for page_number, page_text in text_by_page:
        sentences = _extract_sentences(page_text)
        for sentence in sentences:
            sentence_cf = sentence.casefold()

            matched_in_sentence = False
            for pat in DONOR_DONEE_EXPLICIT_PATTERNS:
                m = re.search(pat, sentence, re.IGNORECASE)
                if m:
                    raw_term = m.group(1).strip().casefold()
                    alt_term = re.sub(r"\s*[/\\-]\s*", " and ", raw_term)
                    alt_to = re.sub(r"\s+to\s+", " and ", raw_term)
                    norm_entry = (
                        DONOR_DONEE_RELATIONSHIP_TERMS.get(raw_term)
                        or DONOR_DONEE_RELATIONSHIP_TERMS.get(alt_term)
                        or DONOR_DONEE_RELATIONSHIP_TERMS.get(alt_to)
                        or next((v for k, v in DONOR_DONEE_RELATIONSHIP_TERMS.items() if k in raw_term), None)
                    )
                    if norm_entry:
                        disp_val, norm_val = norm_entry
                        found_candidates.append((disp_val, norm_val, sentence, page_number, False))
                        matched_in_sentence = True
                        break

            if matched_in_sentence:
                continue

            # If donor_name and donee_name given, check if sentence mentions both with a relation
            if d_first and e_first and d_first in sentence_cf and e_first in sentence_cf:
                name_rel = re.search(
                    r"\b(father|mother|son|daughter|brother|sister|husband|wife|वडील|आई|मुलगा|मुलगी|भाऊ|बहीण|पिता|माता|पुत्र|पुत्री)\b",
                    sentence,
                    re.IGNORECASE,
                )
                if name_rel:
                    raw_term = name_rel.group(1).strip().casefold()
                    norm_entry = DONOR_DONEE_RELATIONSHIP_TERMS.get(raw_term)
                    if norm_entry:
                        disp_val, norm_val = norm_entry
                        found_candidates.append((disp_val, norm_val, sentence, page_number, False))
                        continue

            # Check vague terms (only if sentence context mentions donor/donee/parties/gift)
            if any(term in sentence_cf for term in ["donor", "donee", "parties", "दाता", "प्राप्तकर्ता", "देणार", "घेणार"]):
                for vpat in DONOR_DONEE_VAGUE_PATTERNS:
                    vm = re.search(vpat, sentence, re.IGNORECASE)
                    if vm:
                        found_candidates.append(("Near Relatives", "near relatives", sentence, page_number, True))
                        break

    if not found_candidates:
        return (
            "not_available",
            None,
            "No explicit donor-donee relationship found in recital text; verify the source page.",
            default_page,
            False,
        )

    non_vague = [c for c in found_candidates if not c[4]]
    if non_vague:
        norm_values = {c[1] for c in non_vague}
        if len(norm_values) > 1:
            all_disp = ", ".join(dict.fromkeys(c[0] for c in non_vague))
            ev = non_vague[0][2]
            pg = non_vague[0][3]
            return (f"Ambiguous: {all_disp}", "ambiguous", ev, pg, True)
        return non_vague[0]

    return found_candidates[0]


DEVANAGARI_DIGITS_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def parse_indian_currency_amount(raw_text: str | None) -> int | None:
    """
    Translates Devanagari digits (०-९) to ASCII (0-9) and extracts numeric rupee amount.
    Supports formats: ₹25,00,000, ₹ 25,00,000/-, Rs. 25,00,000, Rs 2500000,
    INR 25,00,000.00, रु. १५,००,०००/-, रुपये २५,००,०००.
    Strips commas and trailing .00 / /-, returning integer rupees.
    """
    if not raw_text:
        return None
    translated = str(raw_text).translate(DEVANAGARI_DIGITS_MAP)
    m = re.search(
        r"(?:₹|Rs\.?|INR|रु\.?|रुपये)\s*([0-9][0-9,]*(?:\.\d{1,2})?)\s*(?:/-)?",
        translated,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(r"([0-9][0-9,]*(?:\.\d{1,2})?)\s*/-", translated)
    if not m:
        m = re.search(r"\b([0-9][0-9,]*(?:\.\d{1,2})?)\b", translated)
    if not m:
        return None
    num_str = m.group(1).replace(",", "").strip()
    if not num_str:
        return None
    try:
        val = Decimal(num_str)
        return int(val.to_integral_value(rounding=ROUND_HALF_UP))
    except (ValueError, ArithmeticError):
        return None


AGRICULTURAL_PRIMARY_TERMS: tuple[str, ...] = (
    "agricultural land", "agriculture land", "agricultural property",
    "farm land", "cultivable land", "dry land", "irrigated land",
    "शेती जमीन", "कृषी जमीन", "शेत जमीन", "बागायत जमीन", "जिरायत जमीन",
)

AGRICULTURAL_SUPPORTING_INDICATORS: tuple[str, ...] = (
    "survey number", "gat number", "7/12 extract", "crop area",
    "kharif", "rabi", "जमीन", "गट क्रमांक", "सातबारा", "पिकाखाली क्षेत्र", "सर्व्हे नंबर",
)

RESIDENTIAL_PRIMARY_TERMS: tuple[str, ...] = (
    "residential property", "residential premises", "residential flat",
    "flat", "apartment", "house", "bungalow", "row house", "dwelling", "tenement",
    "घर", "निवासी", "फ्लॅट", "अपार्टमेंट", "सदनिका", "बंगला",
)

OTHER_PROPERTY_TERMS: tuple[str, ...] = (
    "commercial", "office", "shop", "industrial", "warehouse",
    "कार्यालय", "दुकान", "व्यावसायिक", "गाळा", "गोदाम",
)

SCHEDULE_HEADER_PATTERNS = [
    r"(?:\bschedule\s*(?:of\s*(?:the\s*)?property)?\b|परिशिष्ट|मिळकतीचे\s*वर्णन|मिळकत\s*वर्णन|चतुःसीमा|description\s+of\s+(?:the\s+)?property)",
]

MARKET_VALUE_PATTERNS = [
    r"(?:\bmarket\s*value(?:\s*of\s*(?:the\s*)?property)?\b|\bready\s*reckoner\s*value\b|\breckoner\s*value\b|\bgovernment\s*valuation\b|\bvaluation(?:\s*amount|\s*value)?\b|\bvalue\s*of\s*(?:the\s*)?property\b|बाजार\s*मूल्य|बाजारभाव|मूल्यांकन\s*मूल्य|मूल्यांकन\s*रक्कम|रेडी\s*रेकनर\s*मूल्य)(?:\s+is\s*|\s*[:：=-]?\s*)[^0-9०-९\n]*?([0-9०-९][0-9०-९,]*(?:\.\d{1,2})?(?:\s*/-)?)"
]

ASSESSED_VALUE_PATTERNS = [
    r"(?:\bassessed\s*value\b|\bvaluation\s*amount\b|\btax\s*assessment\s*value\b|कर\s*आकारणी\s*मूल्य|मूल्यांकन\s*राशि)(?:\s+is\s*|\s*[:：=-]?\s*)[^0-9०-९\n]*?([0-9०-९][0-9०-९,]*(?:\.\d{1,2})?(?:\s*/-)?)"
]

TOTAL_STAMP_DUTY_PATTERNS = [
    r"(?:\btotal\s*stamp\s*duty(?:\s*paid)?\b|\btotal\s*duty\s*paid\b|एकूण\s*मुद्रांक\s*शुल्क|एकूण\s*शुल्क)(?:\s+is\s*|\s*[:：=-]?\s*)[^0-9०-९\n]*?([0-9०-९][0-9०-९,]*(?:\.\d{1,2})?(?:\s*/-)?)"
]

METRO_CESS_PATTERNS = [
    r"(?:\bMumbai\s*Metro\s*Cess(?:\s*paid)?\b|\bmetro\s*cess(?:\s*paid)?\b|\bmetro-cess(?:\s*paid)?\b|मेट्रो\s*सेस(?:\s*भरले)?|अधिभार|\bcess(?:\s*paid)?\b)(?:\s+is\s*|\s*[:：=-]?\s*)[^0-9०-९\n]*?([0-9०-९][0-9०-९,]*(?:\.\d{1,2})?(?:\s*/-)?)"
]

BASE_STAMP_DUTY_PATTERNS = [
    r"(?<!total\s)(?<!एकूण\s)(?:\bstamp\s*duty\s*paid\b|\bduty\s*paid\b|\bstamp\s*duty\b|\bduty\s*payable\b|\bSD\b|भरलेले\s*मुद्रांक\s*शुल्क|मुद्रांक\s*शुल्क|stamp\s*शुल्क)(?:\s+is\s*|\s*[:：=-]?\s*)[^0-9०-९\n]*?([0-9०-९][0-9०-९,]*(?:\.\d{1,2})?(?:\s*/-)?)"
]

CHALLAN_PATTERNS = [
    r"(?:\bchallan\s*(?:no\.?|number)\b|\be-sbtr\s*(?:no\.?|number)?\b|\bGRN\b|चलन\s*क्रमांक|चालान\s*(?:क्रमांक|संख्या|no\.?|number)?)(?:\s+is\s*|\s*[:：=-]?\s*)([A-Z0-9][A-Z0-9/-]{3,})"
]


def extract_property_schedule(
    text_by_page: list[tuple[int, str]],
) -> tuple[str, str | None, str | None, str | None, str | None, int | None, bool, str, int]:
    """
    Scans page texts for Schedule of Property and recital descriptions.
    Returns:
    (property_type, property_description, property_location, survey_number, area, market_value, is_ambiguous_property_type, evidence, source_page)
    """
    if isinstance(text_by_page, str):
        text_by_page = [(1, text_by_page)]
    default_page = text_by_page[0][0] if text_by_page else 1
    agri_matches: list[tuple[int, str, str]] = []
    res_matches: list[tuple[int, str, str]] = []
    other_matches: list[tuple[int, str, str]] = []
    supporting_agri_matches: list[tuple[int, str, str]] = []

    descriptions: list[tuple[int, str]] = []
    locations: list[tuple[int, str]] = []
    survey_numbers: list[tuple[int, str]] = []
    areas: list[tuple[int, str]] = []
    market_values: list[tuple[int, int, str]] = []

    for page_num, text in text_by_page:
        lines = text.splitlines()
        for idx, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue
            line_cf = line_str.casefold()

            is_header = any(re.search(pat, line_str, re.IGNORECASE) for pat in SCHEDULE_HEADER_PATTERNS)
            if is_header:
                desc_snippet = " ".join(lines[idx:idx+4]).strip()
                if desc_snippet:
                    descriptions.append((page_num, desc_snippet[:500]))

            for term in AGRICULTURAL_PRIMARY_TERMS:
                if term in line_cf:
                    agri_matches.append((page_num, term, line_str))

            for term in RESIDENTIAL_PRIMARY_TERMS:
                pattern = rf"\b{re.escape(term)}\b" if len(term) < 6 and term.isascii() else re.escape(term)
                if re.search(pattern, line_cf):
                    res_matches.append((page_num, term, line_str))

            for term in OTHER_PROPERTY_TERMS:
                pattern = rf"\b{re.escape(term)}\b" if len(term) < 6 and term.isascii() else re.escape(term)
                if re.search(pattern, line_cf):
                    other_matches.append((page_num, term, line_str))

            for term in AGRICULTURAL_SUPPORTING_INDICATORS:
                if term in line_cf:
                    supporting_agri_matches.append((page_num, term, line_str))

            survey_match = re.search(
                r"(?:\bsurvey\s*(?:no\.?|number)?|\bgat\s*(?:no\.?|number)?|गट\s*क्रमांक|गा\.?\s*नं\.?|ग\.\s*नं\.?|सर्व्हे\s*(?:नंबर|नं\.?))\s*[:#-]?\s*([A-Z0-9०-९][A-Z0-9०-९/-]*)",
                line_str,
                re.IGNORECASE,
            ) or re.search(FIELD_PATTERNS["survey_number"], line_str, re.IGNORECASE)
            if survey_match:
                survey_numbers.append((page_num, survey_match.group(1).strip()))

            area_match = re.search(FIELD_PATTERNS["property_area"], line_str, re.IGNORECASE)
            if area_match:
                areas.append((page_num, area_match.group(1).strip()))

            loc_match = re.search(
                r"(?:at\s+village|village|taluka|district|मौजे|गाव|तालुका|जिल्हा)\s*[:：-]?\s*([A-Za-z\u0900-\u097F\s,.-]{3,60})",
                line_str,
                re.IGNORECASE,
            )
            if loc_match:
                locations.append((page_num, loc_match.group(0).strip()[:200]))

            for mv_pat in MARKET_VALUE_PATTERNS:
                mvm = re.search(mv_pat, line_str, re.IGNORECASE)
                if mvm:
                    amt = parse_indian_currency_amount(mvm.group(1))
                    if amt:
                        market_values.append((page_num, amt, line_str[:300]))

    has_agri = len(agri_matches) > 0
    has_res = len(res_matches) > 0
    has_other = len(other_matches) > 0

    if has_agri and has_res:
        property_type = "unknown"
        is_ambiguous = True
        evidence = f"Mixed primary terms: Agri ({[m[1] for m in agri_matches]}) and Res ({[m[1] for m in res_matches]})"
        source_page = agri_matches[0][0]
    elif has_agri:
        property_type = "agricultural_land"
        is_ambiguous = False
        evidence = f"Agricultural primary terms: {[m[1] for m in agri_matches]}; evidence: {agri_matches[0][2][:300]}"
        source_page = agri_matches[0][0]
    elif has_res:
        property_type = "residential_property"
        is_ambiguous = False
        evidence = f"Residential primary terms: {[m[1] for m in res_matches]}; evidence: {res_matches[0][2][:300]}"
        source_page = res_matches[0][0]
    elif has_other:
        property_type = "other_property"
        is_ambiguous = False
        evidence = f"Other property terms: {[m[1] for m in other_matches]}; evidence: {other_matches[0][2][:300]}"
        source_page = other_matches[0][0]
    elif supporting_agri_matches:
        property_type = "unknown"
        is_ambiguous = False
        evidence = f"Supporting indicators only (survey/gat without primary terms): {[m[1] for m in supporting_agri_matches]}"
        source_page = supporting_agri_matches[0][0]
    else:
        property_type = "unknown"
        is_ambiguous = False
        evidence = "No property terms found in schedule or recitals"
        source_page = default_page

    prop_desc = descriptions[0][1] if descriptions else (agri_matches[0][2] if agri_matches else (res_matches[0][2] if res_matches else None))
    prop_loc = locations[0][1] if locations else None
    surv_no = survey_numbers[0][1] if survey_numbers else None
    prop_area = areas[0][1] if areas else None
    mv = market_values[0][1] if market_values else None

    return (property_type, prop_desc, prop_loc, surv_no, prop_area, mv, is_ambiguous, evidence, source_page)


def extract_market_value(
    text_by_page: list[tuple[int, str]],
) -> tuple[int | None, bool, str, int, bool]:
    """
    Extracts market value across pages.
    Returns:
    (market_value, has_conflicting_market_values, evidence, source_page, is_assessed_fallback)
    """
    if isinstance(text_by_page, str):
        text_by_page = [(1, text_by_page)]
    default_page = text_by_page[0][0] if text_by_page else 1
    mv_candidates: list[tuple[int, str, str, int]] = []
    assessed_candidates: list[tuple[int, str, str, int]] = []

    for page_num, text in text_by_page:
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            for pat in MARKET_VALUE_PATTERNS:
                m = re.search(pat, line_str, re.IGNORECASE)
                if m:
                    raw_val = m.group(1).strip()
                    parsed = parse_indian_currency_amount(raw_val)
                    if parsed is not None:
                        mv_candidates.append((page_num, line_str[:500], raw_val, parsed))
            for pat in ASSESSED_VALUE_PATTERNS:
                m = re.search(pat, line_str, re.IGNORECASE)
                if m:
                    raw_val = m.group(1).strip()
                    parsed = parse_indian_currency_amount(raw_val)
                    if parsed is not None:
                        assessed_candidates.append((page_num, line_str[:500], raw_val, parsed))

    if mv_candidates:
        distinct_amounts = {c[3] for c in mv_candidates}
        if len(distinct_amounts) > 1:
            ev = "Conflicting market values: " + "; ".join(f"Page {c[0]}: {c[1]}" for c in mv_candidates)
            return None, True, ev, mv_candidates[0][0], False
        return mv_candidates[0][3], False, mv_candidates[0][1], mv_candidates[0][0], False

    if assessed_candidates:
        ev = "assessed_value_fallback_only: " + "; ".join(f"Page {c[0]}: {c[1]}" for c in assessed_candidates)
        return None, False, ev, assessed_candidates[0][0], True

    return None, False, "No market value found in deed text", default_page, False


def evaluate_article_34_relationship(
    text_by_page: list[tuple[int, str]],
    donor_name: str | None = None,
    donee_name: str | None = None,
) -> tuple[str | None, str | None, str, int, str, bool]:
    """
    Evaluates Article 34 relationship eligibility.
    Returns:
    (relationship_value, relationship_normalized, relationship_eligibility, source_page, evidence, is_ambiguous)
    Where relationship_eligibility is 'eligible' | 'not_eligible' | 'unknown' | 'ambiguous'.
    """
    rel_val, rel_norm, rel_ev, rel_page, is_amb = extract_donor_donee_relationship(
        text_by_page, donor_name=donor_name, donee_name=donee_name
    )

    if rel_val == "not_available" or not rel_norm:
        return None, None, "unknown", rel_page, rel_ev, False

    if is_amb or rel_norm == "ambiguous" or rel_norm in ("near relatives", "relative"):
        return rel_val, "ambiguous", "ambiguous", rel_page, rel_ev, True

    norm_clean = rel_norm.casefold().strip().replace("-", "_").replace(" ", "_")
    mapped = ARTICLE_34_RELATIONSHIP_LEXICAL_MAPPINGS.get(rel_norm.casefold().strip(), norm_clean)
    mapped_clean = mapped.casefold().strip().replace("-", "_").replace(" ", "_")

    is_eligible = (
        mapped_clean in ARTICLE_34_NORMALIZED_ELIGIBLE_RELATIONSHIPS
        or norm_clean in ARTICLE_34_NORMALIZED_ELIGIBLE_RELATIONSHIPS
        or rel_norm.casefold() in [k.casefold() for k in ARTICLE_34_ELIGIBLE_RELATIONSHIPS]
    )

    is_ineligible = (
        mapped_clean in [k.replace("-", "_").replace(" ", "_").casefold() for k in ARTICLE_34_INELIGIBLE_RELATIONSHIPS]
        or norm_clean in [k.replace("-", "_").replace(" ", "_").casefold() for k in ARTICLE_34_INELIGIBLE_RELATIONSHIPS]
        or rel_norm.casefold() in [k.casefold() for k in ARTICLE_34_INELIGIBLE_RELATIONSHIPS]
    )

    if is_eligible and not is_ineligible:
        return rel_val, mapped_clean, "eligible", rel_page, rel_ev, False
    elif is_ineligible:
        return rel_val, mapped_clean, "not_eligible", rel_page, rel_ev, False
    else:
        return rel_val, mapped_clean, "not_eligible", rel_page, rel_ev, False


def extract_stamp_duty_paid(
    text_by_page: list[tuple[int, str]],
) -> tuple[int | None, int | None, int | None, str | None, bool, str, int]:
    """
    Extracts stamp duty paid, metro cess paid, total paid, and challan number across pages.
    Returns:
    (stamp_duty_paid, metro_cess_paid, total_stamp_duty_paid, challan_number, has_conflicting_stamp_duty, evidence, source_page)
    """
    if isinstance(text_by_page, str):
        text_by_page = [(1, text_by_page)]
    default_page = text_by_page[0][0] if text_by_page else 1
    base_duty_candidates: list[tuple[int, str, str, int]] = []
    metro_cess_candidates: list[tuple[int, str, str, int]] = []
    total_duty_candidates: list[tuple[int, str, str, int]] = []
    challan_candidates: list[tuple[int, str, str]] = []

    for page_num, text in text_by_page:
        for line in text.splitlines():
            line_str = line.strip()
            if not line_str:
                continue

            for pat in TOTAL_STAMP_DUTY_PATTERNS:
                m = re.search(pat, line_str, re.IGNORECASE)
                if m:
                    parsed = parse_indian_currency_amount(m.group(1))
                    if parsed is not None:
                        total_duty_candidates.append((page_num, line_str[:500], m.group(1).strip(), parsed))

            for pat in METRO_CESS_PATTERNS:
                m = re.search(pat, line_str, re.IGNORECASE)
                if m:
                    parsed = parse_indian_currency_amount(m.group(1))
                    if parsed is not None:
                        metro_cess_candidates.append((page_num, line_str[:500], m.group(1).strip(), parsed))

            if not any(re.search(p, line_str, re.IGNORECASE) for p in TOTAL_STAMP_DUTY_PATTERNS):
                for pat in BASE_STAMP_DUTY_PATTERNS:
                    m = re.search(pat, line_str, re.IGNORECASE)
                    if m:
                        parsed = parse_indian_currency_amount(m.group(1))
                        if parsed is not None:
                            base_duty_candidates.append((page_num, line_str[:500], m.group(1).strip(), parsed))

            for pat in CHALLAN_PATTERNS:
                m = re.search(pat, line_str, re.IGNORECASE)
                if m:
                    challan_candidates.append((page_num, line_str[:500], m.group(1).strip()))

    has_conflicts = False
    distinct_base = {c[3] for c in base_duty_candidates}
    distinct_total = {c[3] for c in total_duty_candidates}
    if len(distinct_base) > 1 or len(distinct_total) > 1:
        has_conflicts = True

    base_paid = base_duty_candidates[0][3] if base_duty_candidates else None
    metro_paid = metro_cess_candidates[0][3] if metro_cess_candidates else None

    if total_duty_candidates:
        total_paid = total_duty_candidates[0][3]
    elif base_paid is not None or metro_paid is not None:
        total_paid = (base_paid or 0) + (metro_paid or 0)
    else:
        total_paid = None

    challan_no = challan_candidates[0][2] if challan_candidates else None

    ev_parts = []
    if total_duty_candidates:
        ev_parts.append(f"Total Paid: {total_duty_candidates[0][1]}")
    if base_duty_candidates:
        ev_parts.append(f"Duty: {base_duty_candidates[0][1]}")
    if metro_cess_candidates:
        ev_parts.append(f"Metro Cess: {metro_cess_candidates[0][1]}")
    if challan_candidates:
        ev_parts.append(f"Challan: {challan_candidates[0][1]}")
    evidence = "; ".join(ev_parts) if ev_parts else "No stamp duty payment extracted"

    page = default_page
    for cand_list in (total_duty_candidates, base_duty_candidates, metro_cess_candidates, challan_candidates):
        if cand_list:
            page = cand_list[0][0]
            break

    return (base_paid, metro_paid, total_paid, challan_no, has_conflicts, evidence, page)


def assess_stamp_duty_article_34(
    document: LogicalDocument,
    text_by_page: list[tuple[int, str]],
    all_pages: list[Page] | None = None,
    db: Session | None = None,
) -> StampDutyAssessment:
    """
    Performs Article 34 Stamp Duty assessment for Maharashtra gift deeds.
    Calculates expected stamp duty and metro cess using Decimal ROUND_HALF_UP.
    Persists/updates StampDutyAssessment and generates ValidationFindings.
    """
    donor_name = next((f.extracted_value for f in document.fields if f.field_name == "donor_name"), None)
    donee_name = next((f.extracted_value for f in document.fields if f.field_name == "donee_name"), None)

    rel_val, rel_norm, rel_elig, rel_page, rel_ev, rel_is_amb = evaluate_article_34_relationship(
        text_by_page, donor_name=donor_name, donee_name=donee_name
    )
    if rel_elig == "unknown" and all_pages:
        all_text_by_page = [
            (p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "")
            for p in all_pages
        ]
        fb_val, fb_norm, fb_elig, fb_page, fb_ev, fb_amb = evaluate_article_34_relationship(
            all_text_by_page, donor_name=donor_name, donee_name=donee_name
        )
        if fb_elig != "unknown":
            rel_val, rel_norm, rel_elig, rel_page, rel_ev, rel_is_amb = fb_val, fb_norm, fb_elig, fb_page, fb_ev, fb_amb

    prop_type, prop_desc, prop_loc, surv_no, prop_area, sched_mv, prop_is_amb, prop_ev, prop_page = (
        extract_property_schedule(text_by_page)
    )
    if prop_type == "unknown" and not prop_is_amb and all_pages:
        all_text_by_page = [
            (p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "")
            for p in all_pages
        ]
        fb_type, fb_desc, fb_loc, fb_surv, fb_area, fb_mv, fb_amb, fb_ev, fb_page = extract_property_schedule(
            all_text_by_page
        )
        if fb_type != "unknown" or fb_amb:
            prop_type, prop_desc, prop_loc, surv_no, prop_area, sched_mv, prop_is_amb, prop_ev, prop_page = (
                fb_type, fb_desc, fb_loc, fb_surv, fb_area, fb_mv, fb_amb, fb_ev, fb_page
            )

    mv, has_conflicting_mv, mv_ev, mv_page, mv_is_assessed_fallback = extract_market_value(text_by_page)
    if mv is None and not has_conflicting_mv and all_pages:
        all_text_by_page = [
            (p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "")
            for p in all_pages
        ]
        fb_mv, fb_conf_mv, fb_mv_ev, fb_mv_page, fb_fallback = extract_market_value(all_text_by_page)
        if fb_mv is not None or fb_conf_mv:
            mv, has_conflicting_mv, mv_ev, mv_page, mv_is_assessed_fallback = fb_mv, fb_conf_mv, fb_mv_ev, fb_mv_page, fb_fallback
    if mv is None and sched_mv is not None and not has_conflicting_mv:
        mv = sched_mv

    combined_text = " ".join(t for _, t in text_by_page).casefold()
    loc_text = (prop_loc or "").casefold()

    is_non_metro = any(ind in loc_text or ind in combined_text for ind in NON_METRO_INDICATORS)
    is_metro = any(j in loc_text or j in combined_text for j in METRO_CESS_JURISDICTIONS)

    if prop_type == "agricultural_land":
        jurisdiction = "non_metro" if is_non_metro else ("metro" if is_metro else "non_metro")
        is_metro_cess_applicable = False
    elif is_non_metro:
        jurisdiction = "non_metro"
        is_metro_cess_applicable = False
    elif is_metro:
        jurisdiction = "metro"
        is_metro_cess_applicable = True
    else:
        jurisdiction = "unknown"
        is_metro_cess_applicable = False

    base_paid, metro_paid, total_paid, challan_no, has_conflicting_duty, duty_ev, duty_page = (
        extract_stamp_duty_paid(text_by_page)
    )
    if total_paid is None and all_pages:
        all_text_by_page = [
            (p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "")
            for p in all_pages
        ]
        fb_base, fb_metro, fb_total, fb_ch, fb_conf_duty, fb_duty_ev, fb_duty_page = extract_stamp_duty_paid(
            all_text_by_page
        )
        if fb_total is not None or fb_conf_duty:
            base_paid, metro_paid, total_paid, challan_no, has_conflicting_duty, duty_ev, duty_page = (
                fb_base, fb_metro, fb_total, fb_ch, fb_conf_duty, fb_duty_ev, fb_duty_page
            )

    base_duty = int(STAMP_DUTY_BASE_AMOUNT)
    metro_cess_rate = float(STAMP_DUTY_METRO_CESS_RATE) if is_metro_cess_applicable else 0.0
    metro_cess_amount: int | None = None
    expected_total: int | None = None
    difference: int | None = None
    status = "requires_review"
    status_reason = ""

    if rel_elig == "not_eligible" or prop_type == "other_property":
        status = "not_applicable"
        status_reason = (
            f"Article 34 concessional rate is not applicable: "
            f"Relationship '{rel_val}' (eligibility: {rel_elig}) or Property Type '{prop_type}' "
            "does not qualify for family gift deed concession. Standard conveyance stamp duty applies."
        )
        base_duty = None
        metro_cess_rate = None
        metro_cess_amount = None
        expected_total = None
        difference = None
    elif rel_is_amb or prop_is_amb or has_conflicting_mv or has_conflicting_duty:
        status = "requires_review"
        reasons = []
        if rel_is_amb:
            reasons.append(f"Ambiguous relationship in recital: {rel_val}")
        if prop_is_amb:
            reasons.append("Ambiguous / mixed property type in schedule")
        if has_conflicting_mv:
            reasons.append("Conflicting market values extracted across pages")
        if has_conflicting_duty:
            reasons.append("Conflicting stamp duty paid amounts across pages")
        status_reason = "; ".join(reasons)
        if mv is not None and not has_conflicting_mv:
            metro_cess = (Decimal(mv) * Decimal(str(metro_cess_rate))).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            metro_cess_amount = int(metro_cess)
            expected_total = int(STAMP_DUTY_BASE_AMOUNT) + metro_cess_amount
    elif rel_elig == "unknown":
        status = "not_available"
        status_reason = "No explicit donor-donee relationship found in recital text. Article 34 eligibility cannot be confirmed."
    elif prop_type == "unknown":
        status = "requires_review"
        status_reason = "Property type in Schedule of Property could not be classified as residential or agricultural."
        if mv is not None:
            metro_cess = (Decimal(mv) * Decimal(str(metro_cess_rate))).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            metro_cess_amount = int(metro_cess)
            expected_total = int(STAMP_DUTY_BASE_AMOUNT) + metro_cess_amount
    elif jurisdiction == "unknown" and prop_type != "agricultural_land":
        status = "requires_review"
        status_reason = "Metro Cess jurisdiction could not be determined from property location. Review whether metro cess applies."
        if mv is not None:
            metro_cess = (Decimal(mv) * Decimal("0.01")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
            metro_cess_amount = int(metro_cess)
            expected_total = int(STAMP_DUTY_BASE_AMOUNT) + metro_cess_amount
    elif mv is None:
        status = "not_available"
        status_reason = "Declared market value was not found in the property schedule or recitals. Required to calculate stamp duty and metro cess."
    elif total_paid is None:
        if is_metro_cess_applicable:
            metro_cess = (Decimal(mv) * Decimal("0.01")).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            metro_cess_amount = int(metro_cess)
            expected_total = int(STAMP_DUTY_BASE_AMOUNT) + metro_cess_amount
        else:
            metro_cess_amount = 0
            expected_total = int(STAMP_DUTY_BASE_AMOUNT)
        status = "not_available"
        status_reason = f"Expected total duty ₹{expected_total:,}, but no stamp duty payment was extracted to compare."
    else:
        if is_metro_cess_applicable:
            metro_cess = (Decimal(mv) * Decimal("0.01")).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
            metro_cess_amount = int(metro_cess)
            expected_total = int(STAMP_DUTY_BASE_AMOUNT) + metro_cess_amount
        else:
            metro_cess_amount = 0
            expected_total = int(STAMP_DUTY_BASE_AMOUNT)

        difference = total_paid - expected_total
        if difference >= 0:
            status = "compliant"
            status_reason = (
                f"Compliant under Article 34. Paid ₹{total_paid:,} meets or exceeds expected ₹{expected_total:,} "
                f"(Base: ₹{base_duty} + Metro Cess: ₹{metro_cess_amount:,})."
            )
        else:
            status = "short_paid"
            status_reason = (
                f"Short paid under Article 34 by ₹{abs(difference):,}. Expected ₹{expected_total:,} "
                f"(Base: ₹{base_duty} + Metro Cess: ₹{metro_cess_amount:,}), but actual paid is ₹{total_paid:,}."
            )

    source_pages_set = {rel_page, prop_page, mv_page, duty_page}
    source_pages_str = ",".join(str(p) for p in sorted(source_pages_set))

    evidence_summary = (
        f"Recital: {rel_ev}; Property: {prop_ev}; Valuation: {mv_ev}; Payment: {duty_ev}"
    )[:2000]

    assessment = None
    if db:
        assessment = db.scalar(
            select(StampDutyAssessment).where(StampDutyAssessment.document_id == document.id)
        )
    if assessment is None:
        assessment = getattr(document, "stamp_duty_assessment", None)
    if assessment is None:
        assessment = StampDutyAssessment(document_id=document.id)
        if db:
            db.add(assessment)

    assessment.donor_name = donor_name
    assessment.donee_name = donee_name
    assessment.relationship = rel_val
    assessment.relationship_normalized = rel_norm
    assessment.relationship_eligibility = rel_elig
    assessment.property_type = prop_type
    assessment.property_description = prop_desc
    assessment.property_location = prop_loc
    assessment.survey_number = surv_no
    assessment.area = prop_area
    assessment.jurisdiction = jurisdiction
    assessment.market_value = mv
    assessment.base_duty = base_duty
    assessment.metro_cess_rate = metro_cess_rate
    assessment.metro_cess_amount = metro_cess_amount
    assessment.expected_total = expected_total
    assessment.stamp_duty_paid = base_paid
    assessment.metro_cess_paid = metro_paid
    assessment.total_stamp_duty_paid = total_paid
    assessment.challan_number = challan_no
    assessment.difference = difference
    assessment.status = status
    assessment.legal_rule_version = STAMP_DUTY_RULE_VERSION
    assessment.evidence = evidence_summary
    assessment.source_pages = source_pages_str
    assessment.reviewer_notes = status_reason

    document.stamp_duty_assessment = assessment

    findings_list: list[ValidationFinding] = []
    case_id = document.case_id or (document.case.id if getattr(document, "case", None) else None)

    if status == "short_paid":
        for r_id in ("stamp_duty_short_paid", "short_paid_stamp_duty"):
            findings_list.append(
                ValidationFinding(
                    case_id=case_id,
                    document_id=document.id,
                    rule_id=r_id,
                    severity="high",
                    kind="compliance_deficit",
                    explanation=status_reason,
                    evidence=evidence_summary,
                    source_pages=source_pages_str,
                )
            )
    if has_conflicting_mv or "Conflicting market values" in status_reason:
        for r_id in ("stamp_duty_calculation_requires_review", "conflicting_market_values"):
            findings_list.append(
                ValidationFinding(
                    case_id=case_id,
                    document_id=document.id,
                    rule_id=r_id,
                    severity="medium",
                    kind="conflicting_information",
                    explanation="Conflicting market values were extracted across pages. Review the schedule of property and ready reckoner rate.",
                    evidence=mv_ev,
                    source_pages=str(mv_page),
                )
            )
    if prop_is_amb:
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="ambiguous_property_type",
                severity="medium",
                kind="classification_ambiguity",
                explanation="Property type in Schedule of Property is ambiguous or contains mixed residential and agricultural indicators. Concessional Article 34 duty requires manual review.",
                evidence=prop_ev,
                source_pages=str(prop_page),
            )
        )
    if prop_type == "unknown" and rel_elig != "not_eligible" and not prop_is_amb:
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="unknown_property_type",
                severity="medium",
                kind="classification_ambiguity",
                explanation="Property type in Schedule of Property could not be classified as residential or agricultural. Concessional Article 34 duty requires manual review.",
                evidence=prop_ev,
                source_pages=str(prop_page),
            )
        )
    if rel_elig == "not_eligible":
        for r_id in ("relationship_not_eligible_article_34", "ineligible_article_34_relationship"):
            findings_list.append(
                ValidationFinding(
                    case_id=case_id,
                    document_id=document.id,
                    rule_id=r_id,
                    severity="medium",
                    kind="ineligible_concession",
                    explanation=f"The donor-donee relationship ('{rel_val}') is not eligible for concessional stamp duty under Article 34 of Maharashtra Stamp Act. Standard conveyance duty applies.",
                    evidence=rel_ev,
                    source_pages=str(rel_page),
                )
            )
    if rel_elig == "unknown" and document.category == "gift_deed":
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="missing_donor_donee_relationship",
                severity="low",
                kind="missing_information",
                explanation="Expected donor-donee relationship was not explicitly stated in the recital text. Verify against the source document.",
                source_pages=str(rel_page),
            )
        )
    if rel_is_amb and document.category == "gift_deed":
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="ambiguous_donor_donee_relationship",
                severity="medium",
                kind="ambiguous_relationship",
                explanation=f"The donor-donee relationship in the recital text is ambiguous ({rel_val}). Verify against the source document.",
                evidence=rel_ev,
                source_pages=str(rel_page),
            )
        )
    if jurisdiction == "unknown" and rel_elig == "eligible" and prop_type != "agricultural_land":
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="metro_cess_jurisdiction_unknown",
                severity="medium",
                kind="jurisdiction_ambiguity",
                explanation="Metro Cess jurisdiction could not be determined from property location. Verify whether property falls within a municipal corporation / metro area.",
                evidence=prop_loc or "Location not specified",
                source_pages=str(prop_page),
            )
        )
    if mv is None and rel_elig == "eligible" and document.category == "gift_deed":
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="missing_market_value",
                severity="low",
                kind="missing_information",
                explanation="Market value / Ready Reckoner value was not found in the property schedule or recitals. Required to calculate metro cess under Article 34.",
                source_pages=source_pages_str,
            )
        )
    if total_paid is None and rel_elig == "eligible" and document.category == "gift_deed":
        findings_list.append(
            ValidationFinding(
                case_id=case_id,
                document_id=document.id,
                rule_id="missing_stamp_duty_paid",
                severity="low",
                kind="missing_information",
                explanation="Stamp duty payment receipt details were not found in the document.",
                source_pages=source_pages_str,
            )
        )

    assessment.findings_list = findings_list
    if db:
        existing_new_rules = {
            f.rule_id
            for f in db.new
            if isinstance(f, ValidationFinding) and f.document_id == document.id
        }
        for f in findings_list:
            if f.rule_id not in existing_new_rules:
                db.add(f)

    def _add_or_update_doc_field(
        fname: str,
        fval: str | None,
        spage: int,
        fev: str,
        fmethod: str = "stamp_duty_assessment",
    ):
        if fval is None:
            return
        existing = next((f for f in document.fields if f.field_name == fname), None)
        if not existing and db:
            existing = db.scalar(
                select(ExtractedField).where(
                    ExtractedField.document_id == document.id,
                    ExtractedField.field_name == fname,
                )
            )
        if not existing:
            new_f = ExtractedField(
                document_id=document.id,
                field_name=fname,
                extracted_value=str(fval),
                normalized_value=str(fval),
                source_page=spage,
                evidence=fev,
                extraction_method=fmethod,
                verification_status="unreviewed",
            )
            if db:
                db.add(new_f)
            document.fields.append(new_f)
        else:
            if existing.extracted_value is None and fval is not None:
                existing.extracted_value = str(fval)
                existing.normalized_value = str(fval)

    _add_or_update_doc_field("property_type", prop_type if prop_type != "unknown" else None, prop_page, prop_ev)
    _add_or_update_doc_field("property_description", prop_desc, prop_page, prop_ev)
    _add_or_update_doc_field("property_location", prop_loc, prop_page, prop_ev)
    _add_or_update_doc_field("survey_number", surv_no, prop_page, prop_ev)
    _add_or_update_doc_field("area", prop_area, prop_page, prop_ev)
    _add_or_update_doc_field("market_value", str(mv) if mv is not None else None, mv_page, mv_ev)
    _add_or_update_doc_field("relationship_eligibility", rel_elig, rel_page, rel_ev)
    _add_or_update_doc_field("stamp_duty_paid", str(base_paid) if base_paid is not None else None, duty_page, duty_ev)
    _add_or_update_doc_field("metro_cess_paid", str(metro_paid) if metro_paid is not None else None, duty_page, duty_ev)
    _add_or_update_doc_field("total_stamp_duty_paid", str(total_paid) if total_paid is not None else None, duty_page, duty_ev)
    _add_or_update_doc_field("challan_number", challan_no, duty_page, duty_ev)
    _add_or_update_doc_field("expected_stamp_duty", str(expected_total) if expected_total is not None else None, duty_page, "Expected stamp duty calculated under Article 34")
    _add_or_update_doc_field("stamp_duty_difference", str(difference) if difference is not None else None, duty_page, "Difference between total paid and expected stamp duty")
    _add_or_update_doc_field("stamp_duty_status", status, duty_page, status_reason)

    return assessment


_INVOICE_LABELS = tuple(
    dict.fromkeys(
        label
        for labels, _occurrence in INVOICE_FIELD_LABELS.values()
        for label in labels
    )
)


def classify_text(text: str) -> tuple[str, float, str]:
    normalized = text.casefold()
    scores = {
        category: sum(1 for keyword in keywords if keyword.casefold() in normalized)
        for category, keywords in CATEGORIES.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "unknown", 0.0, "No configured category keywords matched"
    confidence_proxy = min(0.9, 0.45 + 0.15 * (score - 1))
    evidence = ", ".join(keyword for keyword in CATEGORIES[category] if keyword.casefold() in normalized)
    return category, confidence_proxy, evidence


def _clean_field_evidence(text: str, pattern: str) -> tuple[str, str] | None:
    for line in text.splitlines():
        match = re.search(pattern, line, flags=re.IGNORECASE)
        if match:
            value = (match.group(1) if match.lastindex else line).strip(" :।\t")
            if value:
                return value[:500], line.strip()[:500]
    return None


def _label_pattern(label: str) -> str:
    return r"(?<!\w)" + re.escape(label).replace(r"\ ", r"[ \t]+") + r"[ \t]*[:：][ \t]*"


def _extract_invoice_label(text: str, labels: tuple[str, ...], occurrence: int) -> tuple[str, str] | None:
    pattern = re.compile("|".join(f"(?:{_label_pattern(label)})" for label in labels), re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if occurrence >= len(matches):
        return None
    match = matches[occurrence]
    next_label = re.compile(
        "|".join(f"(?:{_label_pattern(label)})" for label in _INVOICE_LABELS),
        re.IGNORECASE,
    )
    line_end = text.find("\n", match.end())
    value_end = line_end if line_end >= 0 else len(text)
    next_match = next_label.search(text, match.end(), value_end)
    if next_match:
        value_end = next_match.start()
    value = text[match.end():value_end].strip(" \t:：-–|")
    if not value:
        return None
    return value[:500], text[max(0, match.start() - 80):min(len(text), value_end + 80)].strip()[:500]


def _extract_invoice_irn(text_by_page: list[tuple[int, str]]) -> tuple[int, str, str] | None:
    irn_label = re.compile(_label_pattern("irn"), re.IGNORECASE)
    for page_number, text in text_by_page:
        match = irn_label.search(text)
        if not match:
            continue
        line_end = text.find("\n", match.end())
        if line_end < 0:
            line_end = len(text)
        segments = [text[match.end():line_end].strip()]
        segments.extend(text[line_end:].splitlines()[1:4])
        parts: list[str] = []
        for segment in segments:
            value = re.sub(r"\s+", "", segment)
            if not value:
                continue
            if not re.fullmatch(r"[a-f0-9]+", value, re.IGNORECASE):
                break
            parts.append(value)
            candidate = "".join(parts)
            if len(candidate) == 64:
                evidence = f"IRN: {candidate}"
                return page_number, candidate, evidence
            if len(candidate) > 64:
                break
    return None


def _extract_invoice_fields(document: LogicalDocument, text_by_page: list[tuple[int, str]]) -> list[ExtractedField]:
    found: list[ExtractedField] = []
    for field_name, (labels, occurrence) in INVOICE_FIELD_LABELS.items():
        result = None
        source_page = document.start_page
        for page_number, text in text_by_page:
            result = _extract_invoice_label(text, labels, occurrence)
            if result:
                source_page = page_number
                break

        if field_name == "supplier_name":
            result = None
            for page_number, text in text_by_page:
                match = re.search(
                    r"\bTAX\s+INVOICE\s+(?:Logo\s+)?([A-Z0-9][A-Z0-9&., _-]{1,60}?)\s+Registered\s+Office\b",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    result = (match.group(1).strip(" ,"), match.group(0).strip()[:500])
                    source_page = page_number
                    break

        if field_name == "invoice_irn":
            irn = _extract_invoice_irn(text_by_page)
            result = (irn[1], irn[2]) if irn else None
            if irn:
                source_page = irn[0]

        if field_name == "invoice_date" and result:
            date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", result[0])
            result = (date_match.group(0), result[1]) if date_match else None

        if field_name == "amount_in_words":
            for page_number, text in text_by_page:
                match = re.search(
                    r"\b((?:(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
                    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                    r"eighty|ninety|hundred|thousand|lakh|lakhs|crore|crores|and)"
                    r"\s+){1,8}rupees?(?:\s+only)?)\b",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    result = (re.sub(r"\s+", " ", match.group(1)).strip(), match.group(0)[:500])
                    source_page = page_number
                    break

        if field_name == "invoice_total":
            for page_number, text in text_by_page:
                matches = re.findall(
                    r"\bTotal\s*[₹]?\s*([0-9][0-9,]*(?:\.\d{1,2})?)",
                    text,
                    re.IGNORECASE,
                )
                if matches:
                    result = (matches[-1], f"Total {matches[-1]}")
                    source_page = page_number
                    break

        value = result[0] if result else None
        found.append(
            ExtractedField(
                document_id=document.id,
                field_name=field_name,
                extracted_value=value,
                normalized_value=value,
                source_page=source_page,
                evidence=result[1] if result else "Expected invoice field not found in OCR text; verify the source page.",
                extraction_method="invoice_label" if result else "expected_invoice_field",
                verification_status="unreviewed" if result else "not_available",
            )
        )

    for page_number, text in text_by_page:
        for line in text.splitlines():
            row = re.search(
                r"^\s*(?:\d{1,3}\s+)?(?P<description>[A-Za-z][A-Za-z0-9_-]{1,40})\s+(?P<hsn>\d{4,8})\b.*$",
                line.strip(),
            )
            if not row:
                continue
            for field_name, value in (
                ("line_item_description", row.group("description")),
                ("hsn_sac_code", row.group("hsn")),
                ("invoice_line_item_raw", line.strip()[:500]),
            ):
                field = next(item for item in found if item.field_name == field_name)
                field.extracted_value = value
                field.normalized_value = value
                field.source_page = page_number
                field.evidence = line.strip()[:500]
                field.extraction_method = "invoice_table_row"
                field.verification_status = "unreviewed"
            break
    return found


def extract_fields(document: LogicalDocument, pages: list[Page], all_pages: list[Page] | None = None) -> list[ExtractedField]:
    text_by_page = [(p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "") for p in pages]
    if document.category == "tax_invoice":
        return _extract_invoice_fields(document, text_by_page)
    found: list[ExtractedField] = []
    for field_name, pattern in FIELD_PATTERNS.items():
        for page_number, text in text_by_page:
            result = _clean_field_evidence(text, pattern)
            if result:
                value, evidence = result
                found.append(
                    ExtractedField(
                        document_id=document.id,
                        field_name=field_name,
                        extracted_value=value,
                        normalized_value=re.sub(r"\s+", "", value).casefold() if field_name.endswith("_number") else value,
                        source_page=page_number,
                        evidence=evidence,
                        extraction_method="label_rule",
                        verification_status="unreviewed",
                        sensitive=field_name == "identity_number",
                    )
                )
                break

    names: list[tuple[int, str, str, str]] = []
    for page_number, text in text_by_page:
        for line in text.splitlines():
            match = re.search(
                r"(donor\s*/\s*दाता|donee\s*/\s*प्राप्तकर्ता|देणार(?:\s*/\s*दाता)?|घेणार(?:\s*/\s*प्राप्तकर्ता)?|donor|donee|दाता|देणार|घेणारा?|प्राप्तकर्ता|transferor|recipient|owner|party|श्री|श्रीमती)\s*[:：.-]\s*([\w .'-]{3,100})",
                line,
                re.IGNORECASE,
            )
            if match:
                label = match.group(1).casefold().split("/", 1)[0].strip()
                name = match.group(2).strip(" .:-")
                if name and not re.search(r"\d{4}", name):
                    field_name = {
                        "donor": "donor_name",
                        "दाता": "donor_name",
                        "देणार": "donor_name",
                        "donee": "donee_name",
                        "प्राप्तकर्ता": "donee_name",
                        "घेणार": "donee_name",
                        "घेणारा": "donee_name",
                        "transferor": "transferor_name",
                        "recipient": "recipient_name",
                        "owner": "owner_name",
                    }.get(label, "")
                    names.append((page_number, name, line.strip(), field_name))
    emitted_names: set[str] = set()
    party_index = 0
    for page_number, name, evidence, field_name in names:
        if field_name and field_name in emitted_names:
            continue
        if not field_name:
            party_index += 1
            field_name = f"party_{party_index}_name"
        if field_name in emitted_names:
            continue
        emitted_names.add(field_name)
        found.append(
            ExtractedField(
                document_id=document.id,
                field_name=field_name,
                extracted_value=name,
                source_page=page_number,
                evidence=evidence,
                extraction_method="label_rule",
                verification_status="unreviewed",
            )
        )

    donor_name = next((f.extracted_value for f in found if f.field_name == "donor_name"), None)
    donee_name = next((f.extracted_value for f in found if f.field_name == "donee_name"), None)
    has_gift_context = (
        document.category == "gift_deed"
        or (donor_name is not None and donee_name is not None)
        or any(
            re.search(r"\b(?:gift\s*deed)\b|दानपत्र|बक्षीसपत्र", text, re.IGNORECASE)
            for _, text in text_by_page
        )
    )
    if has_gift_context:
        rel_val, rel_norm, rel_ev, rel_page, is_amb = extract_donor_donee_relationship(
            text_by_page, donor_name=donor_name, donee_name=donee_name
        )
        if rel_val == "not_available" and all_pages:
            all_text_by_page = [
                (p.page_number, p.ocr_text_corrected or p.ocr_text_original or p.digital_text or "")
                for p in all_pages
            ]
            fallback_val, fallback_norm, fallback_ev, fallback_page, fallback_amb = extract_donor_donee_relationship(
                all_text_by_page, donor_name=donor_name, donee_name=donee_name
            )
            if fallback_val != "not_available":
                rel_val, rel_norm, rel_ev, rel_page, is_amb = fallback_val, fallback_norm, fallback_ev, fallback_page, fallback_amb
        if rel_val != "not_available":
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="donor_donee_relationship",
                    extracted_value=rel_val,
                    normalized_value=rel_norm,
                    source_page=rel_page,
                    evidence=rel_ev,
                    extraction_method="ambiguous_recital" if is_amb else "recital_rule",
                    verification_status="unreviewed",
                )
            )
        elif document.category == "gift_deed" or (donor_name and donee_name):
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="donor_donee_relationship",
                    extracted_value=None,
                    normalized_value=None,
                    source_page=rel_page,
                    evidence=rel_ev,
                    extraction_method="expected_gift_deed_field",
                    verification_status="not_available",
                )
            )

        prop_type, prop_desc, prop_loc, surv_no, prop_area, sched_mv, prop_is_amb, prop_ev, prop_page = (
            extract_property_schedule(text_by_page)
        )
        if prop_type != "unknown":
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="property_type",
                    extracted_value=prop_type,
                    normalized_value=prop_type,
                    source_page=prop_page,
                    evidence=prop_ev,
                    extraction_method="schedule_rule",
                    verification_status="unreviewed",
                )
            )

        mv, has_conf_mv, mv_ev, mv_page, is_assessed_fallback = extract_market_value(text_by_page)
        if mv is None and sched_mv is not None:
            mv = sched_mv
        if mv is not None:
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="market_value",
                    extracted_value=str(mv),
                    normalized_value=str(mv),
                    source_page=mv_page,
                    evidence=mv_ev,
                    extraction_method="valuation_rule",
                    verification_status="unreviewed",
                )
            )
        elif document.category == "gift_deed":
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="market_value",
                    extracted_value=None,
                    normalized_value=None,
                    source_page=mv_page,
                    evidence=mv_ev,
                    extraction_method="expected_gift_deed_field",
                    verification_status="not_available",
                )
            )

        base_paid, metro_paid, total_paid, challan_no, has_conf_duty, duty_ev, duty_page = (
            extract_stamp_duty_paid(text_by_page)
        )
        if base_paid is not None:
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="stamp_duty_paid",
                    extracted_value=str(base_paid),
                    normalized_value=str(base_paid),
                    source_page=duty_page,
                    evidence=duty_ev,
                    extraction_method="stamp_duty_rule",
                    verification_status="unreviewed",
                )
            )
        if metro_paid is not None:
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="metro_cess_paid",
                    extracted_value=str(metro_paid),
                    normalized_value=str(metro_paid),
                    source_page=duty_page,
                    evidence=duty_ev,
                    extraction_method="stamp_duty_rule",
                    verification_status="unreviewed",
                )
            )
        if total_paid is not None:
            found.append(
                ExtractedField(
                    document_id=document.id,
                    field_name="total_stamp_duty_paid",
                    extracted_value=str(total_paid),
                    normalized_value=str(total_paid),
                    source_page=duty_page,
                    evidence=duty_ev,
                    extraction_method="stamp_duty_rule",
                    verification_status="unreviewed",
                )
            )

    return found


def refresh_ingestion_extraction(db: Session, ingestion_id: str) -> dict[str, int]:
    pages = db.scalars(
        select(Page).where(Page.ingestion_id == ingestion_id).order_by(Page.page_number)
    ).all()
    documents = db.scalars(
        select(LogicalDocument)
        .where(LogicalDocument.ingestion_id == ingestion_id)
        .order_by(LogicalDocument.start_page)
    ).all()
    refreshed_fields = 0
    refreshed_documents = 0

    for page in pages:
        text = page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
        page.category, page.classification_confidence_proxy, page.classification_evidence = classify_text(text)

    for document in documents:
        document_pages = [
            page for page in pages
            if document.start_page <= page.page_number <= document.end_page
        ]
        if not document_pages:
            continue
        combined_text = "\n".join(
            page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
            for page in document_pages
        )
        predicted_category, confidence, evidence = classify_text(combined_text)
        document.predicted_category = predicted_category
        document.confidence_proxy = confidence
        document.classification_evidence = evidence
        if document.review_status != "reviewed":
            document.category = predicted_category

        preserved_fields = [
            field for field in document.fields
            if field.verification_status in {"verified", "uncertain", "incorrect"}
            or field.verified_value is not None
        ]
        preserved_names = {field.field_name for field in preserved_fields}
        for field in list(document.fields):
            if field not in preserved_fields:
                db.delete(field)

        refreshed = extract_fields(document, document_pages, all_pages=pages)
        for field in refreshed:
            if field.field_name not in preserved_names:
                db.add(field)
                refreshed_fields += 1

        db.execute(
            delete(ValidationFinding).where(
                ValidationFinding.document_id == document.id,
                ValidationFinding.rule_id.in_([
                    "missing_invoice_fields",
                    "missing_donor_donee_relationship",
                    "ambiguous_donor_donee_relationship",
                    "short_paid_stamp_duty",
                    "stamp_duty_short_paid",
                    "conflicting_market_values",
                    "stamp_duty_calculation_requires_review",
                    "ambiguous_property_type",
                    "unknown_property_type",
                    "ineligible_article_34_relationship",
                    "relationship_not_eligible_article_34",
                    "missing_market_value",
                    "missing_stamp_duty_paid",
                    "metro_cess_jurisdiction_unknown",
                ]),
            )
        )
        if document.category == "tax_invoice":
            missing_invoice_fields = [
                field.field_name.replace("_", " ")
                for field in refreshed
                if field.verification_status == "not_available"
            ]
            if missing_invoice_fields:
                db.add(
                    ValidationFinding(
                        case_id=document.case_id,
                        document_id=document.id,
                        rule_id="missing_invoice_fields",
                        severity="low",
                        kind="missing_information",
                        explanation=(
                            "Expected tax-invoice values were not found in the OCR text: "
                            + ", ".join(missing_invoice_fields)
                            + ". Verify each item against the source page."
                        ),
                        source_pages=",".join(str(page.page_number) for page in document_pages),
                    )
                )
        if document.category == "gift_deed":
            rel_field = next((f for f in refreshed if f.field_name == "donor_donee_relationship"), None)
            if rel_field and rel_field.verification_status == "not_available":
                db.add(
                    ValidationFinding(
                        case_id=document.case_id,
                        document_id=document.id,
                        rule_id="missing_donor_donee_relationship",
                        severity="low",
                        kind="missing_information",
                        explanation="Expected donor-donee relationship was not explicitly stated in the recital text. Verify against the source document.",
                        source_pages=str(rel_field.source_page),
                    )
                )
            if rel_field and rel_field.extraction_method == "ambiguous_recital":
                db.add(
                    ValidationFinding(
                        case_id=document.case_id,
                        document_id=document.id,
                        rule_id="ambiguous_donor_donee_relationship",
                        severity="medium",
                        kind="ambiguous_relationship",
                        explanation=f"The donor-donee relationship in the recital text is ambiguous ({rel_field.extracted_value}). Verify against the source document.",
                        evidence=rel_field.evidence,
                        source_pages=str(rel_field.source_page),
                    )
                )
        if document.category == "gift_deed" or any(f.field_name == "donor_donee_relationship" for f in refreshed):
            text_by_page = [
                (page.page_number, page.ocr_text_corrected or page.ocr_text_original or page.digital_text or "")
                for page in document_pages
            ]
            assess_stamp_duty_article_34(document, text_by_page, all_pages=pages, db=db)
        refreshed_documents += 1

    return {"documents": refreshed_documents, "fields": refreshed_fields}


def _conflicting_field_values(pages: list[Page]) -> list[tuple[str, list[tuple[int, str]]]]:
    conflict_fields = ("registration_number", "survey_number", "property_id", "property_area", "assessed_value")
    conflicts = []
    for field_name in conflict_fields:
        pattern = FIELD_PATTERNS[field_name]
        values = []
        seen: set[str] = set()
        for page in pages:
            text = page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
            result = _clean_field_evidence(text, pattern)
            if not result:
                continue
            value, _ = result
            normalized = re.sub(r"\s+", "", value).casefold()
            if normalized not in seen:
                seen.add(normalized)
                values.append((page.page_number, value))
        if len(values) > 1:
            conflicts.append((field_name, values))
    return conflicts


def _document_marker(page: Page) -> str | None:
    text = page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
    match = re.search(r"\b(?:document|letter)\s+(?:no\.?\s*)?([A-Z0-9][A-Z0-9-]*)", text, re.IGNORECASE)
    return match.group(1).casefold() if match else None


def _unresolved_annexure_references(
    document_pages: list[Page],
    all_pages: list[Page],
) -> list[tuple[int, str, str]]:
    annexure_label = re.compile(r"^\s*annexure\s+([A-Z0-9]+)\b", re.IGNORECASE)
    annexure_reference = re.compile(r"\bannexure\s+([A-Z0-9]+)\b", re.IGNORECASE)
    available = {
        match.group(1).upper()
        for page in all_pages
        for line in (page.ocr_text_corrected or page.ocr_text_original or page.digital_text or "").splitlines()
        if (match := annexure_label.search(line))
    }
    unresolved = []
    for page in document_pages:
        page_text = page.ocr_text_corrected or page.ocr_text_original or page.digital_text or ""
        for line in page_text.splitlines():
            match = annexure_reference.search(line)
            if match and match.group(1).upper() not in available:
                unresolved.append((page.page_number, match.group(1).upper(), line.strip()[:500]))
    return unresolved


def _starts_new_document(previous: Page, current: Page) -> bool:
    if previous.category != current.category:
        return True
    previous_marker = _document_marker(previous)
    current_marker = _document_marker(current)
    return bool(previous_marker and current_marker and previous_marker != current_marker)


def _prepare_scan(source: Image.Image, rotation: int) -> Image.Image:
    image = ImageOps.exif_transpose(source)
    rotation = rotation % 360
    if rotation:
        image = image.rotate(-rotation, expand=True)
    return image


def _ocr_page(
    image_sources: list[tuple[bytes, str]],
    existing_text: str,
    digital_engine: str,
    image_dir: Path,
    page_number: int,
    rotation: int,
    progress_callback: Callable[[str, float], None] | None = None,
):
    if len(existing_text.strip()) >= 40:
        return existing_text, digital_engine, None, None, None, None
    if not image_sources:
        prefix = "A short text layer was found. " if existing_text.strip() else ""
        error = (
            f"{prefix}No full-page PDF renderer is available and no extractable embedded scan image "
            "was found; OCR was skipped. Use an organization-approved PDF renderer for this page."
        )
        return existing_text, digital_engine, error, None, None, None
    image_dir.mkdir(parents=True, exist_ok=True)
    page_texts = []
    all_boxes = []
    confidences = []
    engines_used = []
    errors = []
    original_page_path = None
    processed_page_path = None
    for image_index, (image_data, suffix) in enumerate(image_sources):
        try:
            original = Image.open(io.BytesIO(image_data))
            original.load()
            source = _prepare_scan(original, rotation)
            extension = suffix if re.fullmatch(r"\.[a-z0-9]{2,5}", suffix) else ".png"
            original_path = image_dir / f"page-{page_number}-image-{image_index}{extension}"
            if not original_page_path:
                original_page_path = original_path
            original_path.write_bytes(image_data)

            grayscale = source.convert("L")
            processed = source.copy()
            processed_path = image_dir / f"page-{page_number}-processed-{image_index}.png"
            processed.save(processed_path, optimize=True)
            if image_index == 0:
                processed_page_path = processed_path

            if progress_callback:
                progress_callback("page_preprocessing", 0.05)
            text, boxes, confidence, engine, engine_errors = run_ocr(
                processed_path, page_number, image_index, progress_callback
            )
            errors.extend(engine_errors)
            if not boxes and get_settings().ocr_retry_rotated_pages:
                angle = 90 if grayscale.width > grayscale.height else 180
                rotated = grayscale.rotate(-angle, expand=True)
                rotated_path = image_dir / f"page-{page_number}-rotated-{angle}-{image_index}.png"
                rotated.save(rotated_path, optimize=True)
                rotated_text, rotated_boxes, rotated_conf, rotated_engine, rotated_errors = run_ocr(
                    rotated_path, page_number, image_index, progress_callback
                )
                errors.extend(rotated_errors)
                if rotated_text and len(rotated_text.strip()) > len(text.strip() or ""):
                    text, boxes, confidence, engine = (
                        rotated_text,
                        rotated_boxes,
                        rotated_conf,
                        rotated_engine,
                    )
            if text:
                page_texts.append(text)
                engines_used.extend(engine.split("+"))
            all_boxes.extend(boxes)
            if confidence is not None:
                confidences.append(confidence)
        except (OcrEngineUnavailableError, OSError, ValueError, RuntimeError) as exc:
            errors.append(f"Image {image_index + 1}: {str(exc)[:200]}")
    confidence = sum(confidences) / len(confidences) if confidences else None
    if not page_texts:
        errors.append("No text was recognized by the configured OCR engines.")
    page_text = "\n".join(page_texts).strip() or existing_text
    error = "; ".join(errors) if errors else None
    page_engine = "+".join(dict.fromkeys(engines_used)) if page_texts else digital_engine
    return page_text, page_engine, error, confidence, json.dumps(all_boxes, ensure_ascii=False) if all_boxes else None, (str(original_page_path), str(processed_page_path) if processed_page_path else None)


def process_ingestion(ingestion_id: str, db_factory) -> None:
    db: Session = db_factory()
    temp_dir: Path | None = None
    try:
        ingestion = db.get(Ingestion, ingestion_id)
        job = db.scalar(select(ProcessingJob).where(ProcessingJob.ingestion_id == ingestion_id))
        if not ingestion or not job:
            return
        has_existing_pages = db.scalar(select(Page.id).where(Page.ingestion_id == ingestion_id).limit(1)) is not None
        if has_existing_pages:
            logger.warning("Resetting stale partial processing state for ingestion %s before reprocessing.", ingestion_id)
            existing_doc_ids = db.scalars(select(LogicalDocument.id).where(LogicalDocument.ingestion_id == ingestion_id)).all()
            if existing_doc_ids:
                db.execute(delete(ValidationFinding).where(ValidationFinding.document_id.in_(existing_doc_ids)))
                db.execute(delete(ExtractedField).where(ExtractedField.document_id.in_(existing_doc_ids)))
                db.execute(delete(StampDutyAssessment).where(StampDutyAssessment.document_id.in_(existing_doc_ids)))
            db.execute(delete(LogicalDocument).where(LogicalDocument.ingestion_id == ingestion_id))
            db.execute(delete(Page).where(Page.ingestion_id == ingestion_id))
            db.flush()
        ingestion.status = job.status = "processing"
        ingestion.progress = 0
        job.stage = "page_extraction"
        db.commit()
        temp_dir = Path(tempfile.mkdtemp(prefix=f"case-{ingestion_id}-"))
        image_dir = Path(get_settings().storage_dir) / ingestion_id / "page_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        page_results = []
        page_source = (
            LocalImageBundle(ingestion.storage_path)
            if ingestion.source_type == "image"
            else LocalPdf(ingestion.storage_path)
        )
        with page_source as pdf:
            ingestion.page_count = pdf.page_count
            if ingestion.page_count > get_settings().max_pdf_pages:
                raise ValueError(f"PDF has {ingestion.page_count} pages; limit is {get_settings().max_pdf_pages}.")
            for index in range(pdf.page_count):
                def report_ocr_progress(stage: str, fraction: float, page_index: int = index) -> None:
                    page_progress = int((page_index + fraction) * 70 / max(ingestion.page_count, 1))
                    job.progress = ingestion.progress = max(job.progress, page_progress)
                    job.stage = stage
                    db.commit()

                digital = pdf.text(index)
                rendered_page = pdf.render_png(index, dpi=get_settings().ocr_dpi) if len(digital.strip()) < 40 else None
                extraction_error = None
                if rendered_page:
                    image_sources = [(rendered_page, ".png")]
                else:
                    try:
                        image_sources = pdf.embedded_images(index)
                    except (KeyError, NotImplementedError, OSError, PdfReadError, TypeError, ValueError) as exc:
                        image_sources = []
                        extraction_error = (
                            "Could not extract an embedded scan image; no approved full-page PDF renderer "
                            f"is available (page {index + 1}, {type(exc).__name__})."
                        )
                        logger.warning(
                            "Embedded scan image extraction failed for page %s (%s)",
                            index + 1,
                            type(exc).__name__,
                        )
                text, engine, error, ocr_confidence, word_boxes, image_paths = _ocr_page(
                    image_sources,
                    digital,
                    pdf.engine,
                    image_dir,
                    index + 1,
                    pdf.rotation(index) if not rendered_page else 0,
                    report_ocr_progress,
                )
                if ingestion.source_type == "image" and image_paths is None:
                    image_paths = (str(pdf.source_image_path(index)), None)
                if extraction_error:
                    error = f"{error}; {extraction_error}" if error else extraction_error
                category, confidence, evidence = classify_text(text)
                page = Page(
                    ingestion_id=ingestion.id,
                    page_number=index + 1,
                    digital_text=digital or None,
                    ocr_text_original=text or None,
                    ocr_engine=engine,
                    ocr_languages=get_settings().ocr_languages if engine.startswith("paddleocr") else "",
                    ocr_error=error,
                    ocr_confidence_proxy=ocr_confidence,
                    ocr_word_boxes=word_boxes,
                    image_path=image_paths[0] if image_paths else None,
                    processed_image_path=image_paths[1] if image_paths else None,
                    category=category,
                    classification_evidence=evidence,
                    classification_confidence_proxy=confidence,
                    processing_status="completed_with_warnings" if error else "completed",
                )
                db.add(page)
                page_results.append(page)
                job.progress = ingestion.progress = int((index + 1) * 70 / max(ingestion.page_count, 1))
                job.stage = "ocr_and_classification"
                db.commit()

        job.stage = "document_segmentation"
        db.flush()
        groups: list[list[Page]] = []
        for page in page_results:
            if groups and not _starts_new_document(groups[-1][-1], page):
                groups[-1].append(page)
            else:
                groups.append([page])
        doc_pages: list[tuple[LogicalDocument, list[Page]]] = []
        for group in groups:
            category = group[0].category
            doc = LogicalDocument(
                ingestion_id=ingestion.id,
                start_page=group[0].page_number,
                end_page=group[-1].page_number,
                category=category,
                predicted_category=category,
                classification_method="keyword_rules",
                classification_evidence="; ".join(filter(None, (p.classification_evidence for p in group)))[:1000],
                confidence_proxy=min(p.classification_confidence_proxy for p in group),
                review_status="needs_review",
            )
            db.add(doc)
            db.flush()
            doc_pages.append((doc, group))
        job.stage = "field_extraction_and_case_linking"
        db.flush()
        case_ids: set[str] = set()
        for doc, pages in doc_pages:
            doc.fields.extend(extract_fields(doc, pages, all_pages=page_results))
            registration = next(
                (field.normalized_value for field in doc.fields if field.field_name == "registration_number"),
                None,
            )
            if registration:
                case = db.scalar(select(CaseRecord).where(CaseRecord.reference == registration))
                if not case:
                    case = CaseRecord(reference=registration, title=f"Reference {registration}", status="needs_review")
                    db.add(case)
                    db.flush()
                doc.case = case
                doc.case_link_status = "high_confidence_candidate"
                doc.case_link_evidence = f"Exact registration reference: {registration}; reviewer confirmation required."
            else:
                case = CaseRecord(title=f"Unlinked document, pages {doc.start_page}-{doc.end_page}")
                db.add(case)
                db.flush()
                doc.case = case
                doc.case_link_status = "insufficient_evidence"
                doc.case_link_evidence = "No exact transaction or registration identifier was extracted; kept separate."
            case_ids.add(case.id)
            if doc.category == "tax_invoice":
                missing_invoice_fields = [
                    field.field_name.replace("_", " ")
                    for field in doc.fields
                    if field.verification_status == "not_available"
                ]
                if missing_invoice_fields:
                    db.add(
                        ValidationFinding(
                            case_id=case.id,
                            document_id=doc.id,
                            rule_id="missing_invoice_fields",
                            severity="low",
                            kind="missing_information",
                            explanation=(
                                "Expected tax-invoice values were not found in the OCR text: "
                                + ", ".join(missing_invoice_fields)
                                + ". Verify each item against the source page."
                            ),
                            source_pages=",".join(str(page.page_number) for page in pages),
                        )
                    )
            if doc.category == "gift_deed":
                rel_field = next((f for f in doc.fields if f.field_name == "donor_donee_relationship"), None)
                if rel_field and rel_field.verification_status == "not_available":
                    db.add(
                        ValidationFinding(
                            case_id=case.id,
                            document_id=doc.id,
                            rule_id="missing_donor_donee_relationship",
                            severity="low",
                            kind="missing_information",
                            explanation="Expected donor-donee relationship was not explicitly stated in the recital text. Verify against the source document.",
                            source_pages=str(rel_field.source_page),
                        )
                    )
                if rel_field and rel_field.extraction_method == "ambiguous_recital":
                    db.add(
                        ValidationFinding(
                            case_id=case.id,
                            document_id=doc.id,
                            rule_id="ambiguous_donor_donee_relationship",
                            severity="medium",
                            kind="ambiguous_relationship",
                            explanation=f"The donor-donee relationship in the recital text is ambiguous ({rel_field.extracted_value}). Verify against the source document.",
                            evidence=rel_field.evidence,
                            source_pages=str(rel_field.source_page),
                        )
                    )
            if doc.category == "gift_deed" or any(f.field_name == "donor_donee_relationship" for f in doc.fields):
                text_by_page = [
                    (page.page_number, page.ocr_text_corrected or page.ocr_text_original or page.digital_text or "")
                    for page in pages
                ]
                assess_stamp_duty_article_34(doc, text_by_page, all_pages=page_results, db=db)
            for field in doc.fields:
                if field.field_name.endswith("_name") and field.extracted_value:
                    person = Person(display_name=field.extracted_value, name_original=field.extracted_value)
                    db.add(person)
                    db.flush()
                    db.add(DocumentPerson(document_id=doc.id, person_id=person.id, relationship_type="mentioned"))
                if field.field_name == "survey_number" and field.extracted_value:
                    prop = PropertyRecord(identifier=field.normalized_value or field.extracted_value, description=field.extracted_value)
                    db.add(prop)
                    db.flush()
                    db.add(DocumentProperty(document_id=doc.id, property_id=prop.id, relationship_type="references"))
            if doc.category in {"gift_deed", "valuation", "challan"} and not registration:
                db.add(
                    ValidationFinding(
                        case_id=case.id,
                        document_id=doc.id,
                        rule_id="missing_registration_reference",
                        severity="low",
                        kind="missing_information",
                        explanation="No registration number was extracted from this document. Verify whether one is present.",
                        source_pages=f"{doc.start_page}-{doc.end_page}",
                    )
                )
            for page_number, annexure_id, evidence in _unresolved_annexure_references(pages, page_results):
                db.add(
                    ValidationFinding(
                        case_id=case.id,
                        document_id=doc.id,
                        rule_id="possible_missing_annexure",
                        severity="low",
                        kind="missing_attachment",
                        explanation=(
                            f"Page {page_number} references Annexure {annexure_id}, but no page in this upload "
                            "is headed as that annexure. Verify against the source documents."
                        ),
                        evidence=evidence,
                        source_pages=str(page_number),
                    )
                )
            for field_name, values in _conflicting_field_values(pages):
                evidence = "; ".join(f"Page {page_number}: {value}" for page_number, value in values)
                db.add(
                    ValidationFinding(
                        case_id=case.id,
                        document_id=doc.id,
                        rule_id="conflicting_extracted_values",
                        severity="low",
                        kind="conflicting_information",
                        explanation=(
                            f"Different values were extracted for {field_name.replace('_', ' ')}. "
                            "Compare the cited pages and resolve through human review; no legal conclusion is implied."
                        ),
                        evidence=evidence,
                        source_pages=",".join(str(page_number) for page_number, _ in values),
                    )
                )
            for page in pages:
                if page.ocr_error:
                    db.add(
                        ValidationFinding(
                            case_id=case.id,
                            document_id=doc.id,
                            rule_id="ocr_unavailable_or_failed",
                            severity="medium",
                            kind="extraction_error",
                            explanation=page.ocr_error,
                            source_pages=str(page.page_number),
                        )
                    )
        job.progress = 95
        job.stage = "validation"
        db.flush()
        for case_id in case_ids:
            sync_case_entities(db, case_id)
        db.flush()
        has_warnings = any(page.ocr_error for page in page_results)
        final_status = "completed_with_warnings" if has_warnings else "completed"
        ingestion.status = job.status = final_status
        ingestion.progress = 100
        job.progress = 100
        job.stage = "completed"
        db.commit()
    except Exception as exc:
        logger.exception("Ingestion processing failed", extra={"ingestion_id": ingestion_id})
        db.rollback()
        ingestion = db.get(Ingestion, ingestion_id)
        job = db.scalar(select(ProcessingJob).where(ProcessingJob.ingestion_id == ingestion_id))
        if ingestion:
            ingestion.status = "failed"
            ingestion.error = str(exc)[:1000]
        if job:
            job.status = "failed"
            job.error = str(exc)[:1000]
            job.stage = "failed"
        db.commit()
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        db.close()


def sync_case_entities(db: Session, case_id: str) -> None:
    case = db.get(CaseRecord, case_id)
    if not case:
        return
    case.people.clear()
    case.properties.clear()
    people = db.scalars(
        select(Person)
        .join(DocumentPerson, DocumentPerson.person_id == Person.id)
        .join(LogicalDocument, LogicalDocument.id == DocumentPerson.document_id)
        .where(LogicalDocument.case_id == case_id)
    ).all()
    properties = db.scalars(
        select(PropertyRecord)
        .join(DocumentProperty, DocumentProperty.property_id == PropertyRecord.id)
        .join(LogicalDocument, LogicalDocument.id == DocumentProperty.document_id)
        .where(LogicalDocument.case_id == case_id)
    ).all()
    for person in people:
        relationship_type = db.scalar(
            select(DocumentPerson.relationship_type)
            .join(LogicalDocument, LogicalDocument.id == DocumentPerson.document_id)
            .where(LogicalDocument.case_id == case_id, DocumentPerson.person_id == person.id)
            .limit(1)
        ) or "mentioned"
        case.people.append(CasePerson(person=person, relationship_type=relationship_type))
    for property_record in properties:
        case.properties.append(CaseProperty(property=property_record))
