# Casework — Local Document Intelligence Prototype

A local-first prototype for ingesting mixed PDF bundles, extracting digital text or English/Hindi/Marathi OCR, proposing document groups and conservative case references, and supporting reviewer corrections with JSON/CSV exports.

> **Prototype, not production software.** Heuristic classification and extraction are not calibrated and must be checked against source pages. This project is not legal advice, does not determine fraud or validity, and is not certified for government, identity-document, or regulated use. Do not process real confidential records until the security limitations below have been addressed and your organization has authorized the deployment.

## What works

- FastAPI backend and responsive React/TypeScript interface.
- Local PDF upload with signature, extension, size, and page-count checks; SHA-256 duplicate detection; private UUID-named file storage.
- Multi-image PNG/JPG/JPEG upload as an ordered page bundle, with browser previews, server-side format/dimension checks, duplicate detection, preserved originals, and the same OCR/review pipeline.
- Asynchronous per-page PDF processing through a local adapter: PyMuPDF page rendering when available, otherwise pypdf text and embedded-image extraction.
- Standard uploads use the Casework pipeline with PaddleOCR and EasyOCR; they do not run PP-StructureV3 or Docling. Those parsers remain available only through the separate, opt-in document-comparison API, which stores their native artifacts and normalized results without replacing Casework extraction.
- Downloadable formatted PDF reports place each original PDF page beside its full extracted text and field table; fields explicitly marked unavailable are highlighted in red, and long text continues on additional report pages.
- PaddleOCR is the primary scanned-page OCR engine; EasyOCR independently corroborates overlapping words, preserves disagreements as reviewable alternatives, and can contribute high-confidence words PaddleOCR missed. Both engines return word-level confidence and boxes; confidence is not calibrated accuracy. Selectable text continues to work without a renderer; unsupported scanned pages receive a page-level warning rather than fabricated OCR output.
- Configurable keyword classification for tax invoices, gift deeds, valuation records, receipts, identity documents, challans, condominium records, survey records, official letters, registration documents, property-tax records, and Unknown. Tax invoices use a dedicated GST field profile; expected but unreadable or absent values are recorded with an explicit `not_available` status.
- Contiguous page grouping with explicit `Document <id>` / `Letter <id>` boundaries; labelled extraction for references, property IDs/areas, values, identity numbers, dates, challans, and named parties.
- Possible missing-annexure references are surfaced with the source line for reviewer verification; the system makes no legal conclusion.
- SQLite persistence with relational case, person, property, document, page, field, finding, review-decision, and audit records. PostgreSQL URLs can be configured through `DATABASE_URL`.
- Exact registration-reference case candidates only; name similarity alone never merges records. Reviewer actions can approve/reject links, split documents, merge adjacent documents, correct OCR/fields/categories, and resolve findings.
- Dashboard counts reflect stored records. Case reports export to JSON and CSV; sensitive fields are masked in default exports.
- Alembic schema migrations and automated API/pipeline tests. Test PDFs are generated in temporary test directories; no demo documents are bundled with the application.

## Requirements

- Python 3.12 or 3.13 for the Windows PaddlePaddle wheels.
- Node.js 20 or newer and npm.
- PaddleOCR/PaddlePaddle and EasyOCR are installed from `requirements.txt`. Their model weights are downloaded on first use; the application does not send uploaded files or extracted text to external services.

### Windows setup

From the project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m alembic upgrade head
```

In another PowerShell window:

```powershell
Set-Location frontend
npm install
npm run build
```

PaddlePaddle currently publishes Windows wheels through Python 3.13; Python 3.12 is the recommended setup for this project. The first OCR request downloads the PaddleOCR and EasyOCR model weights into their local caches and may take longer than subsequent pages. Keep the computer online for that first model download. `OCR_LANGUAGES` accepts `eng`, `hin`, and `mar` (or `en`, `hi`, and `mr`); the Devanagari PaddleOCR model supports Hindi and Marathi, and EasyOCR is configured for the requested languages.

Start the backend from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in a second window:

```powershell
Set-Location frontend
npm run dev
```

Open the Vite URL printed in the terminal (normally `http://127.0.0.1:5173`). The backend health endpoint is `http://127.0.0.1:8000/api/health`.

### PDF support in this environment

| PDF content | Current behavior |
|---|---|
| Selectable / embedded text | Supported: pypdf extracts the text layer without page rendering. |
| Scanned or complex page needing composition/rendering | Supported when PyMuPDF is available: the page is rendered locally for OCR. If PyMuPDF cannot be loaded, the app falls back to selectable text and supported embedded images; scans requiring full-page rendering receive an OCR-skipped warning. |
| A scan exposed as an extractable embedded image | Supported by the pypdf fallback when the image can be decoded and PaddleOCR/EasyOCR are installed. OCR is run on extracted image(s); this is not equivalent to rendering the whole page. |
| Malformed or password-protected PDF | Rejected at upload with an error; it is not passed to OCR. |

An earlier interpreter environment on this machine recorded Windows Code Integrity Event 3077 while loading `pymupdf\_mupdf.pyd`. The project's Python 3.12 `.venv` now imports PyMuPDF and successfully renders PDF pages. Do not disable or work around endpoint controls; if PyMuPDF is blocked in another environment, ask the endpoint administrator to approve it or specify a supported renderer. The health endpoint reports OCR-engine availability separately from `pdf_renderer_available`; configured OCR alone does not imply full scanned-PDF rendering is available.

### Optional document-engine comparison

PP-StructureV3 and Docling are not used by normal document uploads or by the positioned-OCR/formatted-report workflow. They remain optional evaluation adapters reachable through `POST /api/ingestions/{id}/comparisons`; comparison runs are triggered explicitly through that API and shown by the comparison review routes. They store native engine artifacts and normalized page output separately from Casework extraction. PP-StructureV3 or Docling output does not replace or improve the default OCR result automatically. The comparison endpoints are intended for local evaluation, not the normal reviewer workflow.

Comparison text differences are not accuracy scores. Accuracy remains “Not Measured” unless verified ground-truth annotations are supplied. Engine confidence metadata is engine-specific and not a calibrated probability. The optional adapters require their respective heavyweight dependencies and may download model weights on first use; availability and runtime depend on the deployment environment. Missing-value indicators remain application-level validation based on the configured extraction profile, not a parser prediction.

### Formatted PDF extraction report

After an ingestion finishes, download its **Formatted PDF** from the dashboard or a document review page. This is available for both PDF and uploaded-image bundles. Each report sheet shows a generated positioned-OCR page image at left, with recognized words and detected table rules on a white page, and the saved Casework extraction, labeled fields, and OCR warnings at right. This generated view is not a copy of the original scan; compare it with the original page in document review when verifying OCR. Expected fields already marked `not_available` by the applicable extraction profile are highlighted in red with a reminder to verify the source; other missing values are not guessed or flagged. Long OCR text continues on additional sheets alongside the generated page image. Sensitive extracted identifiers are masked in the report, as in the default API exports. Reports are generated locally on request and are available at `GET /api/ingestions/{id}/report.pdf` once ingestion processing has finished.

The document review screen opens in **Positioned OCR** when page coordinates are available. It places words in black at their detected page positions and overlays detected horizontal/vertical rules from uploaded images or rendered PDF pages. It also lists extraction-profile fields marked unavailable in red; it does not guess additional requirements. This is still an approximate OCR preview—not a reconstruction of exact table cells. Use the original source page beside it as the reference. The **Formatted fields** table highlights unavailable expected fields in red; Word view and editable plain text remain available.

Tax-invoice values are extracted conservatively from recognized text and should be checked against the source. “Not available” means the expected field was not found in OCR text; it does not prove that the value is absent from the source image. GST table columns and identifiers are not yet fully validated or reliably normalized.

Mozilla [PDF.js](https://mozilla.github.io/pdf.js/getting_started/) was evaluated as a possible browser-side renderer but is **not integrated**. Source-page review uses the backend's local page-image endpoint, so it displays a single rasterized page rather than depending on the browser's built-in PDF viewer. Digital-text pages are rendered on demand; pages already rendered for OCR reuse their stored source image. A PDF.js OCR path would require new client-side page rendering and a protected page-image upload interface. Consider it only if the approved local renderer is unavailable and the endpoint/security administrator explicitly approves PDF.js and that data flow. No alternative renderer is installed by this prototype.

### Image upload workflow

When a PDF cannot be rendered by the approved local adapter, upload authorized PNG/JPG/JPEG page images directly from the dashboard's image-bundle panel. The files are sent only to the loopback backend, validated for actual image format and dimensions, stored privately as an ordered bundle, and processed by the same PaddleOCR/EasyOCR, classification, field extraction, findings, linking, and review stages. Each uploaded image is one page; the original is retained for review. The `POST /api/ingestions/images` endpoint accepts repeated `files` fields and an optional `bundle_name`. Uploading page images is an alternative input format, not a way to override endpoint security controls. Do not send sensitive pages unless their use and local storage have been approved.

The app binds to loopback in the local commands. By default `APP_API_TOKEN` is blank for local development; this is **not authentication**. For a shared or otherwise exposed environment, set a long random `APP_API_TOKEN` in `.env`, enter the same value in the UI's Access settings, and configure the database, private storage, backups, TLS, identity provider, authorization roles, and operational controls through your organization's approved deployment process. Do not expose this prototype directly to a network.

## Configuration

Copy `.env.example` to `.env`. Important settings:

| Variable | Purpose |
|---|---|
| `APP_API_TOKEN` | Optional bearer-token gate for API access. Blank is local-only development mode, not a secure default for shared use. |
| `DATABASE_URL` | SQLAlchemy URL; defaults to a local SQLite database. |
| `STORAGE_DIR` | Private uploaded PDF and rendered page storage, outside the frontend tree. |
| `MAX_UPLOAD_MB` / `MAX_PDF_PAGES` | Upload and processing limits. |
| `OCR_LANGUAGES` / `OCR_DPI` | OCR language codes (`eng`, `hin`, `mar`) and render resolution. |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins. |
| `VITE_API_URL` | Frontend API base URL. |

Set `DATABASE_URL` to an approved PostgreSQL SQLAlchemy URL (for example, `postgresql+psycopg://user:password@localhost/casework`) to use PostgreSQL. The psycopg driver is included; SQLite remains the tested prototype default.

## Tests

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest backend\tests -q
```

Frontend type-check and production bundle:

```powershell
Set-Location frontend
npm run build
```

Tests create temporary PDF fixtures with test-only names and references; they do not rely on bundled sample documents or upload confidential examples.

## Evaluation with authorized documents

1. Obtain written authorization and use a controlled machine/account with access limited to approved reviewers.
2. Verify local storage location, encryption at rest, backup policy, retention period, and deletion process with your security administrator.
3. Keep the system loopback-bound; configure an approved token or access-control layer before any shared use.
4. Use copies of authorized PDFs and first test a non-sensitive representative sample after both local OCR models are available.
5. Compare OCR against a manually transcribed sample and record character/word error rates. Label representative pages for classification accuracy and extracted fields for precision/recall. Label known links/non-links and measure case-link precision. Record processing time and per-page failures.
6. Do not treat rule scores as calibrated probabilities. Do not accept case links, OCR, classifications, or extracted values without source-page review.
7. Delete test ingestions when finished and verify the configured storage and backup-retention behavior.

The repository includes no confidential sample or claimed accuracy result. No evaluation metrics are reported until measured against a labelled, authorized dataset.

## Known limitations and security work before real records

- PaddleOCR is the primary engine and EasyOCR is an independent verifier, not a guarantee of higher accuracy. Pages with fewer than three PaddleOCR words and pages scoring below 80/100 run full-page EasyOCR. Other pages below 94/100, or with any word below 85/100, use EasyOCR recognition only on PaddleOCR's low-confidence text lines; pages meeting the high-confidence threshold skip EasyOCR. This reduces full-page CPU inference but cannot recover text PaddleOCR failed to detect on a page that skips full-page verification, so verify all source content. For matching disagreements, the higher-scoring reading is selected only when its score is at least 85/100 and either leads by 15 points or the other engine is at or below 55/100; otherwise PaddleOCR is retained and both readings are shown. Unmatched EasyOCR words are included at 80/100 or higher. These are conservative score-based rules, not calibrated correctness probabilities, and engine scores are not guaranteed to be directly comparable. Inference thread counts are capped to reduce CPU oversubscription. Initial OCR use downloads model weights into local caches. If PyMuPDF is unavailable or blocked in another environment, the app falls back to pypdf for embedded text and supported embedded images; some scans then cannot be OCRed without full-page rendering. Word boxes from pypdf-extracted images are image-relative rather than normalized page coordinates.
- OCR now uses the oriented source pixels directly instead of applying contrast and a median filter unconditionally; on the slow sample image, preprocessing coincided with lower PaddleOCR scores and caused expensive full-page verification. If the first pass finds no words, one likely orientation retry is attempted instead of three full-image rotations. Rotating a complete table page can still affect layout; vertical table headings need region processing. OCR confidence values are engine-reported scores, not calibrated accuracy probabilities, and boxes for words split from a detected text line are approximate.
- Segmentation joins contiguous pages with the same rule-based class and honors explicit differing document/letter numbers; it does not yet analyze page-number sequences, layout, or stamps. Annexure checks rely on explicit headings and require reviewer confirmation. Split/merge decisions are editable, but undo history is not a one-click restore.
- Extraction supports a limited set of labelled references, names, property identifiers/areas, values, dates, challans, and identity numbers. It does not infer Hindi fields comprehensively, parse complex tables, link payment records robustly, or calculate stamp duty/property value.
- Case association uses exact extracted registration references as candidates and keeps records without that evidence separate. Entity resolution, cross-ingestion review queues, similarity suggestions, and organization-specific identifiers are not implemented.
- Review notes are limited to decisions; the audit table is local and is not yet a tamper-evident audit service. There are no user roles, server-side authorization scopes, encryption management, anti-malware scanning, rate limits, or production secrets management.
- The optional bearer token is a single shared development gate, not identity-based access control. Without a configured token, the API is unauthenticated and must remain loopback-only.
- JSON/CSV exports contain structured records but no generated PDF report. PDF exports never include source documents.
- Automated tests cover PDF ingestion, the OCR ensemble parser and disagreement behavior, single/multiple direct image upload, an embedded-image OCR fixture when both engines are installed, explicit warnings for unsupported scan pages, digital text, rule extraction/classification, duplicate uploads, corrections, and export. Test fixtures are generated temporarily and do not establish real-world OCR accuracy or validate legal/financial interpretations.

Never use the prototype to make legal, financial, identity, or case-linking decisions without qualified human review and explicit authorization.
