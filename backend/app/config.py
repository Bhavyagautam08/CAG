from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# Article 34 Versioned Legal Defaults & Standards
STAMP_DUTY_RULE_VERSION: str = "MH-ART34-2024.1"
STAMP_DUTY_LEGAL_REFERENCE: str = (
    "Maharashtra Stamp Act, Article 34 (Residential/Agricultural Gift to eligible family members: ₹200 base duty + 1% Metro Cess where applicable)"
)
STAMP_DUTY_EFFECTIVE_FROM: str = "2017-05-16"
STAMP_DUTY_EFFECTIVE_TO: str | None = None
STAMP_DUTY_BASE_AMOUNT: Decimal = Decimal("200")
STAMP_DUTY_METRO_CESS_RATE: Decimal = Decimal("0.01")

# Maharashtra Stamp Act Article 34 Concessional Categories (Donee's relation to donor)
ARTICLE_34_200_RELATIONS: frozenset[str] = frozenset({
    "husband",
    "wife",
    "son",
    "daughter",
    "grandson",
    "granddaughter",
    "wife_of_deceased_son",
})

ARTICLE_34_3_PERCENT_FAMILY_RELATIONS: frozenset[str] = frozenset({
    "father",
    "mother",
    "brother",
    "sister",
})

# Controlled normalized eligible relationships under Article 34
ARTICLE_34_NORMALIZED_ELIGIBLE_RELATIONSHIPS: frozenset[str] = frozenset({
    "father",
    "mother",
    "son",
    "daughter",
    "husband",
    "wife",
    "spouse",
    "brother",
    "sister",
    "grandson",
    "granddaughter",
    "grandfather",
    "grandmother",
    "wife_of_deceased_son",
    "father_and_son",
    "father_and_daughter",
    "mother_and_son",
    "mother_and_daughter",
    "husband_and_wife",
    "brother_and_sister",
    "brother_and_brother",
    "sister_and_sister",
    "grandparent_and_grandchild",
    "parents_and_son",
    "parents_and_daughter",
    "parents_and_children",
    "parent_and_child",
    "son_to_mother",
    "son_to_father",
    "daughter_to_mother",
    "daughter_to_father",
})

# English, Hindi, and Marathi lexical mappings to normalized forms
ARTICLE_34_RELATIONSHIP_LEXICAL_MAPPINGS: dict[str, str] = {
    # English lexical variants
    "father": "father",
    "mother": "mother",
    "son": "son",
    "daughter": "daughter",
    "husband": "husband",
    "wife": "wife",
    "spouse": "spouse",
    "brother": "brother",
    "sister": "sister",
    "brothers": "brother_and_brother",
    "sisters": "sister_and_sister",
    "grandfather": "grandfather",
    "grandmother": "grandmother",
    "grandson": "grandson",
    "granddaughter": "granddaughter",
    "wife of deceased son": "wife_of_deceased_son",
    "father and son": "father_and_son",
    "father-son": "father_and_son",
    "father to son": "father_and_son",
    "father and daughter": "father_and_daughter",
    "father-daughter": "father_and_daughter",
    "father to daughter": "father_and_daughter",
    "mother and son": "mother_and_son",
    "mother-son": "mother_and_son",
    "mother to son": "mother_and_son",
    "mother and daughter": "mother_and_daughter",
    "mother-daughter": "mother_and_daughter",
    "mother to daughter": "mother_and_daughter",
    "husband and wife": "husband_and_wife",
    "husband-wife": "husband_and_wife",
    "husband to wife": "husband_and_wife",
    "wife to husband": "husband_and_wife",
    "brother and sister": "brother_and_sister",
    "brother-sister": "brother_and_sister",
    "brother to sister": "brother_and_sister",
    "sister to brother": "brother_and_sister",
    "brother and brother": "brother_and_brother",
    "sister and sister": "sister_and_sister",
    "parents to son": "parents_and_son",
    "parents and son": "parents_and_son",
    "parents to daughter": "parents_and_daughter",
    "parents and daughter": "parents_and_daughter",
    "parent to son": "father_and_son",
    "parent to daughter": "father_and_daughter",
    "parents to children": "parents_and_children",
    "parents and children": "parents_and_children",
    "parent and child": "parent_and_child",
    "son to mother": "son_to_mother",
    "son and mother": "son_to_mother",
    "son to father": "son_to_father",
    "son and father": "son_to_father",
    "daughter to mother": "daughter_to_mother",
    "daughter and mother": "daughter_to_mother",
    "daughter to father": "daughter_to_father",
    "daughter and father": "daughter_to_father",

    # Hindi lexical variants
    "पिता": "father",
    "माता": "mother",
    "पुत्र": "son",
    "बेटा": "son",
    "सुपुत्र": "son",
    "पुत्री": "daughter",
    "बेटी": "daughter",
    "सुपुत्री": "daughter",
    "पति": "husband",
    "पत्नी": "wife",
    "भाई": "brother",
    "बहन": "sister",
    "सगे भाई": "brother_and_brother",
    "दादा": "grandfather",
    "दादी": "grandmother",
    "नाना": "grandfather",
    "नानी": "grandmother",
    "पोता": "grandson",
    "पोती": "granddaughter",
    "नाती": "grandson",
    "नातिन": "granddaughter",
    "पिता और पुत्र": "father_and_son",
    "पिता-पुत्र": "father_and_son",
    "पिता व पुत्र": "father_and_son",
    "पिता से पुत्र": "father_and_son",
    "पिता और पुत्री": "father_and_daughter",
    "पिता-पुत्री": "father_and_daughter",
    "पिता व पुत्री": "father_and_daughter",
    "पिता से पुत्री": "father_and_daughter",
    "माता और पुत्र": "mother_and_son",
    "माता-पुत्र": "mother_and_son",
    "माता व पुत्र": "mother_and_son",
    "माता से पुत्र": "mother_and_son",
    "माता और पुत्री": "mother_and_daughter",
    "माता-पुत्री": "mother_and_daughter",
    "माता व पुत्री": "mother_and_daughter",
    "माता से पुत्री": "mother_and_daughter",
    "पति और पत्नी": "husband_and_wife",
    "पति-पत्नी": "husband_and_wife",
    "भाई और बहन": "brother_and_sister",
    "भाई-बहन": "brother_and_sister",
    "माता-पिता और पुत्र": "parents_and_son",
    "माता-पिता से पुत्र": "parents_and_son",
    "माता-पिता व पुत्र": "parents_and_son",
    "माता-पिता और पुत्री": "parents_and_daughter",
    "माता-पिता से पुत्री": "parents_and_daughter",
    "पुत्र और माता": "son_to_mother",
    "पुत्र से माता": "son_to_mother",
    "पुत्र और पिता": "son_to_father",
    "पुत्र से पिता": "son_to_father",
    "पुत्री और माता": "daughter_to_mother",
    "पुत्री और पिता": "daughter_to_father",
    "मृत पुत्र की विधवा": "wife_of_deceased_son",

    # Marathi lexical variants
    "वडील": "father",
    "आई": "mother",
    "मुलगा": "son",
    "मुलगी": "daughter",
    "चिरंजीव": "son",
    "कन्या": "daughter",
    "सुकन्या": "daughter",
    "पती": "husband",
    "पत्नी": "wife",
    "भाऊ": "brother",
    "बंधू": "brother",
    "बहीण": "sister",
    "सख्खे बंधू": "brother_and_brother",
    "आजोबा": "grandfather",
    "आजी": "grandmother",
    "नातू": "grandson",
    "नात": "granddaughter",
    "वडील आणि मुलगा": "father_and_son",
    "वडील व मुलगा": "father_and_son",
    "वडील-मुलगा": "father_and_son",
    "वडील ते मुलगा": "father_and_son",
    "वडील आणि मुलगी": "father_and_daughter",
    "वडील व मुलगी": "father_and_daughter",
    "वडील-मुलगी": "father_and_daughter",
    "वडील ते मुलगी": "father_and_daughter",
    "आई आणि मुलगा": "mother_and_son",
    "आई व मुलगा": "mother_and_son",
    "आई-मुलगा": "mother_and_son",
    "आई ते मुलगा": "mother_and_son",
    "आई आणि मुलगी": "mother_and_daughter",
    "आई व मुलगी": "mother_and_daughter",
    "आई-मुलगी": "mother_and_daughter",
    "आई ते मुलगी": "mother_and_daughter",
    "पती आणि पत्नी": "husband_and_wife",
    "पती-पत्नी": "husband_and_wife",
    "भाऊ आणि बहीण": "brother_and_sister",
    "भाऊ-बहीण": "brother_and_sister",
    "आई-वडील आणि मुलगा": "parents_and_son",
    "आई-वडील व मुलगा": "parents_and_son",
    "आई-वडील ते मुलगा": "parents_and_son",
    "आई-वडील आणि मुलगी": "parents_and_daughter",
    "आई-वडील व मुलगी": "parents_and_daughter",
    "आई-वडील ते मुलगी": "parents_and_daughter",
    "मुलगा आणि आई": "son_to_mother",
    "मुलगा व आई": "son_to_mother",
    "मुलगा ते आई": "son_to_mother",
    "मुलगा आणि वडील": "son_to_father",
    "मुलगा व वडील": "son_to_father",
    "मुलगा ते वडील": "son_to_father",
    "मुलगी आणि आई": "daughter_to_mother",
    "मुलगी आणि वडील": "daughter_to_father",
    "मयत मुलाची विधवा": "wife_of_deceased_son",
    "दिवंगत पुत्राची पत्नी": "wife_of_deceased_son",
}

# Controlled tuple of all recognized eligible relationship keys and forms
ARTICLE_34_ELIGIBLE_RELATIONSHIPS: tuple[str, ...] = tuple(
    sorted(
        set(ARTICLE_34_NORMALIZED_ELIGIBLE_RELATIONSHIPS)
        | set(ARTICLE_34_RELATIONSHIP_LEXICAL_MAPPINGS.keys())
    )
)

# Explicit non-eligible relationships
ARTICLE_34_INELIGIBLE_RELATIONSHIPS: tuple[str, ...] = (
    "uncle",
    "aunt",
    "nephew",
    "niece",
    "cousin",
    "friend",
    "brother_in_law",
    "sister_in_law",
    "son_in_law",
    "daughter_in_law",
    "father_in_law",
    "mother_in_law",
    "paternal_uncle",
    "maternal_uncle",
    "paternal_aunt",
    "maternal_aunt",
    # Marathi / Hindi terms
    "काका",
    "मामा",
    "पुतण्या",
    "भाचा",
    "चुलत भाऊ",
    "मित्र",
    "मेहुणा",
    "चाचा",
    "मामी",
    "चाची",
    "मौसी",
    "भतीजा",
    "भतीजी",
    "भांजा",
    "भांजी",
    "ससुर",
    "सास",
    "दामाद",
    "बहू",
)

# Controlled list of Maharashtra metro cess jurisdictions
METRO_CESS_JURISDICTIONS: tuple[str, ...] = (
    "mumbai",
    "mumbai suburban",
    "brihanmumbai",
    "mcgm",
    "bmc",
    "thane",
    "navi mumbai",
    "pune",
    "pimpri-chinchwad",
    "pcmc",
    "pmc",
    "nagpur",
    "nmc",
    "nashik",
    "मुंबई",
    "ठाणे",
    "नवी मुंबई",
    "पुणे",
    "पिंपरी-चिंचवड",
    "नागपूर",
    "नाशिक",
)

# Explicit non-metro / rural indicators
NON_METRO_INDICATORS: tuple[str, ...] = (
    "gram panchayat",
    "grampanchayat",
    "village panchayat",
    "rural",
    "ग्रामपंचायत",
    "गाव",
    "ग्रामीण",
)


class Settings(BaseSettings):
    app_api_token: str = ""
    database_url: str = "sqlite:///./backend/data/case_linking.db"
    storage_dir: Path = Path("./backend/private_uploads")
    max_upload_mb: int = 50
    max_pdf_pages: int = 200
    ocr_languages: str = "eng+hin+mar"
    paddle_comparison_language: Literal["en", "hi", "mr"] = "en"
    # "auto" uses CUDA whenever the installed OCR runtime and hardware support it,
    # and otherwise keeps the ingestion service available on CPU.
    ocr_device: Literal["auto", "cpu", "gpu"] = "auto"
    ocr_dpi: int = 180
    ocr_secondary_verification: bool = False
    ocr_retry_rotated_pages: bool = False
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Article 34 Stamp Duty Assessment configuration
    stamp_duty_rule_version: str = STAMP_DUTY_RULE_VERSION
    stamp_duty_legal_reference: str = STAMP_DUTY_LEGAL_REFERENCE
    stamp_duty_effective_from: str = STAMP_DUTY_EFFECTIVE_FROM
    stamp_duty_effective_to: str | None = STAMP_DUTY_EFFECTIVE_TO
    stamp_duty_base_amount: Decimal = STAMP_DUTY_BASE_AMOUNT
    stamp_duty_metro_cess_rate: Decimal = STAMP_DUTY_METRO_CESS_RATE

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
