from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import BigInteger, Boolean, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, reconstructor, relationship as sa_relationship, relationship

from .db import Base


def new_id() -> str:
    return str(uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Ingestion(Base):
    __tablename__ = "ingestions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(16), default="pdf", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    pages: Mapped[list["Page"]] = relationship(back_populates="ingestion", cascade="all, delete-orphan")
    documents: Mapped[list["LogicalDocument"]] = relationship(back_populates="ingestion", cascade="all, delete-orphan")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    ingestion_id: Mapped[str] = mapped_column(ForeignKey("ingestions.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class CaseRecord(Base):
    __tablename__ = "cases"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255), default="Unreviewed case")
    reference: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="needs_review")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    documents: Mapped[list["LogicalDocument"]] = relationship(back_populates="case")
    people: Mapped[list["CasePerson"]] = relationship(back_populates="case", cascade="all, delete-orphan")
    properties: Mapped[list["CaseProperty"]] = relationship(back_populates="case", cascade="all, delete-orphan")


class Person(Base):
    __tablename__ = "persons"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    display_name: Mapped[str] = mapped_column(String(255), index=True)
    name_original: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    cases: Mapped[list["CasePerson"]] = relationship(back_populates="person", passive_deletes="all")


class CasePerson(Base):
    __tablename__ = "case_person_relationships"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    relationship_type: Mapped[str] = mapped_column(String(64), default="mentioned")
    case: Mapped[CaseRecord] = relationship(back_populates="people")
    person: Mapped[Person] = relationship(back_populates="cases")
    __table_args__ = (UniqueConstraint("case_id", "person_id", "relationship_type"),)


class PropertyRecord(Base):
    __tablename__ = "properties"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    identifier: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(Text)
    cases: Mapped[list["CaseProperty"]] = relationship(back_populates="property", passive_deletes="all")


class CaseProperty(Base):
    __tablename__ = "case_property_relationships"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    property_id: Mapped[str] = mapped_column(ForeignKey("properties.id", ondelete="CASCADE"), index=True)
    case: Mapped[CaseRecord] = relationship(back_populates="properties")
    property: Mapped[PropertyRecord] = relationship(back_populates="cases")
    __table_args__ = (UniqueConstraint("case_id", "property_id"),)


class DocumentPerson(Base):
    __tablename__ = "document_person_relationships"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    person_id: Mapped[str] = mapped_column(ForeignKey("persons.id", ondelete="CASCADE"), index=True)
    relationship_type: Mapped[str] = mapped_column(String(64), default="mentioned")
    __table_args__ = (UniqueConstraint("document_id", "person_id", "relationship_type"),)


class DocumentProperty(Base):
    __tablename__ = "document_property_relationships"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    property_id: Mapped[str] = mapped_column(ForeignKey("properties.id", ondelete="CASCADE"), index=True)
    relationship_type: Mapped[str] = mapped_column(String(64), default="references")
    __table_args__ = (UniqueConstraint("document_id", "property_id", "relationship_type"),)


class LogicalDocument(Base):
    __tablename__ = "logical_documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    ingestion_id: Mapped[str] = mapped_column(ForeignKey("ingestions.id", ondelete="CASCADE"), index=True)
    case_id: Mapped[str | None] = mapped_column(ForeignKey("cases.id", ondelete="SET NULL"), index=True)
    start_page: Mapped[int] = mapped_column(Integer)
    end_page: Mapped[int] = mapped_column(Integer)
    category: Mapped[str] = mapped_column(String(80), default="unknown")
    predicted_category: Mapped[str] = mapped_column(String(80), default="unknown")
    classification_method: Mapped[str] = mapped_column(String(32), default="rules")
    classification_evidence: Mapped[str | None] = mapped_column(Text)
    confidence_proxy: Mapped[float] = mapped_column(Float, default=0)
    review_status: Mapped[str] = mapped_column(String(32), default="needs_review")
    case_link_status: Mapped[str] = mapped_column(String(32), default="unlinked")
    case_link_evidence: Mapped[str | None] = mapped_column(Text)
    ingestion: Mapped[Ingestion] = relationship(back_populates="documents")
    case: Mapped[CaseRecord | None] = relationship(back_populates="documents")
    fields: Mapped[list["ExtractedField"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    relationships: Mapped[list["DocumentRelationship"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="DocumentRelationship.document_id",
    )
    stamp_duty_assessment: Mapped["StampDutyAssessment | None"] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        uselist=False,
    )

    @property
    def pages(self) -> list["Page"]:
        cached = getattr(self, "_pages_cache", None)
        if cached is not None:
            return cached
        ing = getattr(self, "ingestion", None)
        if ing and getattr(ing, "pages", None):
            return sorted(
                [p for p in ing.pages if self.start_page <= p.page_number <= self.end_page],
                key=lambda p: p.page_number,
            )
        return []

    @pages.setter
    def pages(self, value: list["Page"]) -> None:
        self._pages_cache = value


class DocumentRelationship(Base):
    __tablename__ = "document_relationships"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    related_document_id: Mapped[str] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    relation_type: Mapped[str] = mapped_column(String(64), default="possible_supporting_document")
    evidence: Mapped[str] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(32), default="proposed")
    document: Mapped[LogicalDocument] = relationship(foreign_keys=[document_id], back_populates="relationships")


class Page(Base):
    __tablename__ = "pages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    ingestion_id: Mapped[str] = mapped_column(ForeignKey("ingestions.id", ondelete="CASCADE"), index=True)
    ingestion: Mapped[Ingestion] = relationship(back_populates="pages")
    page_number: Mapped[int] = mapped_column(Integer)
    digital_text: Mapped[str | None] = mapped_column(Text)
    ocr_text_original: Mapped[str | None] = mapped_column(Text)
    ocr_text_corrected: Mapped[str | None] = mapped_column(Text)
    ocr_engine: Mapped[str] = mapped_column(String(64), default="pymupdf")
    ocr_languages: Mapped[str] = mapped_column(String(64), default="")
    processing_version: Mapped[str] = mapped_column(String(32), default="0.1.0")
    ocr_confidence_proxy: Mapped[float | None] = mapped_column(Float)
    ocr_word_boxes: Mapped[str | None] = mapped_column(Text)
    ocr_error: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(Text)
    processed_image_path: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(80), default="unknown")
    classification_evidence: Mapped[str | None] = mapped_column(Text)
    classification_confidence_proxy: Mapped[float] = mapped_column(Float, default=0)
    processing_status: Mapped[str] = mapped_column(String(32), default="queued")
    __table_args__ = (UniqueConstraint("ingestion_id", "page_number"),)


class ComparisonRun(Base):
    __tablename__ = "comparison_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    ingestion_id: Mapped[str] = mapped_column(ForeignKey("ingestions.id", ondelete="CASCADE"), index=True)
    mode: Mapped[str] = mapped_column(String(24), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    page_comparison_json_path: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    engines: Mapped[list["EngineProcessingResult"]] = relationship(
        back_populates="comparison",
        cascade="all, delete-orphan",
        order_by="EngineProcessingResult.engine",
    )


class EngineProcessingResult(Base):
    __tablename__ = "engine_processing_results"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    comparison_id: Mapped[str] = mapped_column(ForeignKey("comparison_runs.id", ondelete="CASCADE"), index=True)
    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    package_version: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[str | None] = mapped_column(Text)
    confidence_json: Mapped[str | None] = mapped_column(Text)
    output_json_path: Mapped[str | None] = mapped_column(Text)
    markdown_path: Mapped[str | None] = mapped_column(Text)
    normalized_json_path: Mapped[str | None] = mapped_column(Text)
    processing_seconds: Mapped[float | None] = mapped_column(Float)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    comparison: Mapped[ComparisonRun] = relationship(back_populates="engines")
    __table_args__ = (UniqueConstraint("comparison_id", "engine"),)


class ExtractedField(Base):
    __tablename__ = "extracted_fields"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    field_name: Mapped[str] = mapped_column(String(100), index=True)
    extracted_value: Mapped[str | None] = mapped_column(Text)
    normalized_value: Mapped[str | None] = mapped_column(Text)
    verified_value: Mapped[str | None] = mapped_column(Text)
    source_page: Mapped[int] = mapped_column(Integer)
    evidence: Mapped[str | None] = mapped_column(Text)
    confidence_proxy: Mapped[float | None] = mapped_column(Float)
    extraction_method: Mapped[str] = mapped_column(String(32), default="rule")
    verification_status: Mapped[str] = mapped_column(String(32), default="unreviewed")
    sensitive: Mapped[bool] = mapped_column(default=False)
    document: Mapped[LogicalDocument] = relationship(back_populates="fields")


class ValidationFinding(Base):
    __tablename__ = "validation_findings"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    case_id: Mapped[str | None] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("logical_documents.id", ondelete="CASCADE"), index=True)
    rule_id: Mapped[str] = mapped_column(String(100), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="medium")
    kind: Mapped[str] = mapped_column(String(32), default="missing_information")
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[str | None] = mapped_column(Text)
    source_pages: Mapped[str | None] = mapped_column(String(255))
    review_status: Mapped[str] = mapped_column(String(32), default="open")


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(36), index=True)
    decision: Mapped[str] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[str] = mapped_column(String(36), index=True)
    action: Mapped[str] = mapped_column(String(64))
    before_value: Mapped[str | None] = mapped_column(Text)
    after_value: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(128), default="local-reviewer")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class StampDutyAssessment(Base):
    __tablename__ = "stamp_duty_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("logical_documents.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    donor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    donee_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    donor_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    donee_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    donee_relation_to_donor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    relationship: Mapped[str | None] = mapped_column(String(100), nullable=True)
    relationship_normalized: Mapped[str | None] = mapped_column(String(100), nullable=True)
    relationship_eligibility: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unknown"
    )
    property_type: Mapped[str] = mapped_column(
        String(64), nullable=False, default="unknown"
    )
    property_type_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cess_uncertain: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    location_band: Mapped[str | None] = mapped_column(String(64), nullable=True)
    property_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    property_location: Mapped[str | None] = mapped_column(Text, nullable=True)
    survey_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    area: Mapped[str | None] = mapped_column(String(100), nullable=True)
    jurisdiction: Mapped[str] = mapped_column(
        String(64), nullable=False, default="unknown"
    )
    selected_rule: Mapped[str | None] = mapped_column(String(64), nullable=True, default="review_required")
    duty_rate: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cess_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rule_version: Mapped[str | None] = mapped_column(String(64), nullable=True, default="MH-ART34-2024.1")
    effective_date: Mapped[str | None] = mapped_column(String(32), nullable=True, default="2017-05-16")
    calculation_components: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    identification_documents: Mapped[str | None] = mapped_column(Text, nullable=True)
    market_value: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    base_duty: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metro_cess_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    metro_cess_amount: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    expected_total: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    stamp_duty_paid: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    metro_cess_paid: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_stamp_duty_paid: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    challan_number: Mapped[str | None] = mapped_column(String(100), nullable=True)
    difference: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="requires_review"
    )
    legal_rule_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="MH-ART34-2024.1"
    )
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_pages: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unreviewed"
    )
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    document: Mapped["LogicalDocument"] = sa_relationship(back_populates="stamp_duty_assessment")

    @property
    def extracted_id_documents_json(self) -> str | None:
        return self.identification_documents

    @extracted_id_documents_json.setter
    def extracted_id_documents_json(self, value: str | None) -> None:
        self.identification_documents = value

    @reconstructor
    def _init_on_load(self):
        if getattr(self, "relationship_eligibility", None):
            from .processing import RelationshipEligibility
            self.relationship_eligibility = RelationshipEligibility(self.relationship_eligibility)


Document = LogicalDocument
