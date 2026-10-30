import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Link, NavLink, Route, Routes, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle, ArrowDownToLine, ArrowRight, Bell, BriefcaseBusiness, Check, ChevronDown, CircleHelp,
  Clock3, FileCheck2, FileDown, FileText, Files, LayoutDashboard, LockKeyhole, Menu, Search, ShieldCheck,
  ImagePlus, RotateCcw, Trash2, Upload, X,
} from "lucide-react";
import { api, apiBase, ComparisonDetail, ComparisonRun, DashboardData, Document, download, Field, uploadImages, uploadPdf } from "./api";

type Ingestion = { id: string; filename: string; source_type: "pdf" | "image"; status: string; progress: number; stage?: string; error: string | null };
type CaseItem = { id: string; title: string; reference: string | null; status: string; documents: Document[]; people: { id: string; name: string; relationship: string }[]; properties: { id: string; identifier: string; description: string | null }[] };
type Finding = { id: string; case_id: string | null; document_id: string | null; rule_id: string; severity: string; kind: string; explanation: string; evidence: string | null; source_pages: string | null; review_status: string };
type AuditEntry = { id: string; entity_type: string; action: string; before_value: string | null; after_value: string | null; actor: string; created_at: string };

const navItems = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/documents", label: "Documents", icon: Files },
  { to: "/cases", label: "Cases", icon: BriefcaseBusiness },
  { to: "/findings", label: "Findings", icon: AlertCircle },
];

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [tokenPrompt, setTokenPrompt] = useState(false);
  const queryClient = useQueryClient();
  const health = useQuery({ queryKey: ["health"], queryFn: () => api<{ ocr_configured: boolean }>("/health") });
  useEffect(() => {
    if (health.error?.message.includes("Authentication")) setTokenPrompt(true);
  }, [health.error]);
  function saveToken(value: string) {
    sessionStorage.setItem("casework-token", value);
    setTokenPrompt(false);
    queryClient.invalidateQueries();
  }
  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <Link className="brand" to="/" onClick={() => setMenuOpen(false)}>
          <div className="brand-mark"><ShieldCheck size={21} /></div>
          <div><strong>Casework</strong><span>DOCUMENT INTELLIGENCE</span></div>
        </Link>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="primary-nav">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setMenuOpen(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>
              {label === "Findings" && <span className="nav-dot" />}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="secure-note"><LockKeyhole size={15} /><span>LOCAL WORKSPACE<br /><b>Your files stay on this machine</b></span></div>
          <button className="sidebar-help" onClick={() => setTokenPrompt(true)}><CircleHelp size={16} /> Access settings <ChevronDown size={14} /></button>
          <div className="profile"><div className="avatar">LR</div><div><b>Local reviewer</b><span>Case review desk</span></div><button title="Notifications"><Bell size={16} /></button></div>
        </div>
      </aside>
      {menuOpen && <button className="mobile-backdrop" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
      <main className="main-area">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setMenuOpen(!menuOpen)} aria-label="Toggle menu"><Menu size={20} /></button>
          <div className="breadcrumbs">Case review <span>/</span> <b>Document intelligence</b></div>
          <div className="top-actions"><div className="local-indicator"><span /> Local processing</div><button className="icon-button" aria-label="Help"><CircleHelp size={19} /></button></div>
        </header>
        <div className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/documents/:id" element={<Review />} />
            <Route path="/comparisons" element={<ComparisonRuns />} />
            <Route path="/comparisons/:id" element={<ComparisonReview />} />
            <Route path="/cases" element={<Cases />} />
            <Route path="/findings" element={<Findings />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </div>
      </main>
      {tokenPrompt && <TokenDialog onSave={saveToken} onClose={() => setTokenPrompt(false)} />}
    </div>
  );
}

function PageTitle({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-title"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

type OcrAlternative = { text: string; confidence?: number; engine?: string };
type OcrEngineScore = { confidence: number; engine?: string };
type OcrTableRule = { orientation: "horizontal" | "vertical"; x1: number; y1: number; x2: number; y2: number };
type OcrWordToken = {
  text: string;
  confidence?: number;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  line?: number;
  engine?: string;
  agreement?: boolean;
  alternatives?: OcrAlternative[];
  selection_reason?: string;
  engine_scores?: OcrEngineScore[];
};

function isOcrAlternative(value: unknown): value is OcrAlternative {
  return typeof value === "object"
    && value !== null
    && "text" in value
    && typeof value.text === "string";
}

function isOcrEngineScore(value: unknown): value is OcrEngineScore {
  return typeof value === "object"
    && value !== null
    && "confidence" in value
    && typeof value.confidence === "number";
}

function parseOcrWords(page: { ocr_text?: string; ocr_word_boxes?: string | null } | undefined): OcrWordToken[] {
  if (!page) return [];
  const rawBoxes = page.ocr_word_boxes?.trim();
  if (rawBoxes) {
    try {
      const parsed = JSON.parse(rawBoxes);
      const words = Array.isArray(parsed) ? parsed : Array.isArray(parsed?.words) ? parsed.words : [];
      const normalized: OcrWordToken[] = words
        .map((entry: Record<string, unknown>) => ({
          text: String(entry.text ?? "").trim(),
          confidence: typeof entry.confidence === "number" ? entry.confidence : undefined,
          x: typeof entry.x === "number" ? entry.x : undefined,
          y: typeof entry.y === "number" ? entry.y : undefined,
          width: typeof entry.width === "number" ? entry.width : undefined,
          height: typeof entry.height === "number" ? entry.height : undefined,
          line: typeof entry.line === "number" ? entry.line : undefined,
          engine: typeof entry.engine === "string" ? entry.engine : undefined,
          agreement: typeof entry.agreement === "boolean" ? entry.agreement : undefined,
          alternatives: Array.isArray(entry.alternatives) ? entry.alternatives.filter(isOcrAlternative) : undefined,
          selection_reason: typeof entry.selection_reason === "string" ? entry.selection_reason : undefined,
          engine_scores: Array.isArray(entry.engine_scores) ? entry.engine_scores.filter(isOcrEngineScore) : undefined,
        }))
        .filter((token: OcrWordToken) => token.text.length > 0);
      if (normalized.length) return normalized;
    } catch {
      // Fall back to plain-text splitting below.
    }
  }
  return (page.ocr_text ?? "")
    .split(/\s+/)
    .filter(Boolean)
    .map((text) => ({ text, confidence: undefined }));
}

function parseOcrTableRules(page: { ocr_word_boxes?: string | null } | undefined): OcrTableRule[] {
  const rawBoxes = page?.ocr_word_boxes?.trim();
  if (!rawBoxes) return [];
  try {
    const parsed = JSON.parse(rawBoxes);
    if (!Array.isArray(parsed?.table_lines)) return [];
    return parsed.table_lines.filter((line: Record<string, unknown>): line is OcrTableRule =>
      (line.orientation === "horizontal" || line.orientation === "vertical")
      && typeof line.x1 === "number"
      && typeof line.y1 === "number"
      && typeof line.x2 === "number"
      && typeof line.y2 === "number",
    );
  } catch {
    return [];
  }
}

function OcrWordView({ words }: { words: OcrWordToken[] }) {
  if (!words.length) return <div className="ocr-empty">No OCR words are available for this page.</div>;
  return (
    <div className="ocr-word-view" aria-label="OCR word view">
      {words.map((word, index) => {
        const confidence = typeof word.confidence === "number" ? word.confidence : null;
        const confidenceClass = confidence === null ? "medium" : confidence >= 70 ? "high" : confidence >= 40 ? "medium" : "low";
        const confidenceTitle = confidence == null
          ? "Word confidence not reported"
          : `${word.engine ?? "OCR engine"} confidence ${confidence.toFixed(1)} / 100`;
        const alternativesTitle = word.alternatives?.length
          ? ` Other reading${word.alternatives.length > 1 ? "s" : ""}: ${word.alternatives.map((alternative) => {
            const score = typeof alternative.confidence === "number" ? ` (${alternative.confidence.toFixed(1)} / 100)` : "";
            return `${alternative.engine ?? "Other engine"}: ${alternative.text}${score}`;
          }).join("; ")}`
          : "";
        const agreementTitle = word.agreement ? " PaddleOCR and EasyOCR agree." : "";
        const engineScoresTitle = word.engine_scores?.length
          ? ` Engine scores: ${word.engine_scores.map((score) => `${score.engine ?? "OCR engine"} ${score.confidence.toFixed(1)} / 100`).join("; ")}.`
          : "";
        const selectionTitle = word.selection_reason === "higher_engine_confidence"
          ? " Selected over the other engine because its confidence was substantially higher."
          : word.selection_reason === "confidence_not_decisive"
            ? " Engines disagreed; PaddleOCR was retained because the confidence difference was not decisive."
            : "";
        const conflictClass = word.alternatives?.length ? " conflict" : "";
        return (
          <span key={`${word.text}-${index}`} className={`ocr-word-token ${confidenceClass}${conflictClass}`} title={`${confidenceTitle}${engineScoresTitle}${agreementTitle}${alternativesTitle}${selectionTitle}`}>
            {word.text}
            {word.engine_scores && word.engine_scores.length > 1 && (
              <small className="ocr-word-alternative">
                {word.engine_scores.map((score) => `${score.engine ?? "OCR"} ${score.confidence.toFixed(1)}`).join(" · ")}
              </small>
            )}
            {word.alternatives?.map((alternative, alternativeIndex) => (
              <small className="ocr-word-alternative" key={`${alternative.engine}-${alternative.text}-${alternativeIndex}`}>
                {alternative.engine ?? "Other engine"}: {alternative.text}
                {typeof alternative.confidence === "number" ? ` (${alternative.confidence.toFixed(1)} / 100)` : ""}
              </small>
            ))}
          </span>
        );
      })}
    </div>
  );
}

function OcrLayoutView({
  words,
  rules,
  rulesLoading,
  missingFields,
  imageWidth,
  imageHeight,
}: {
  words: OcrWordToken[];
  rules: OcrTableRule[];
  rulesLoading: boolean;
  missingFields: string[];
  imageWidth: number | null;
  imageHeight: number | null;
}) {
  const positionedWords = words.filter((word) =>
    typeof word.x === "number"
    && typeof word.y === "number"
    && typeof word.width === "number"
    && typeof word.height === "number",
  );
  if ((!positionedWords.length && !rules.length) || !imageWidth || !imageHeight) {
    return <div className="ocr-empty">Positioned OCR layout is unavailable for this page. Use Word view or Plain text.</div>;
  }
  return (
    <>
      <p className="ocr-layout-note">
        OCR words follow their detected positions. {rulesLoading ? "Detecting page rules…" : rules.length ? `${rules.length} long page rule(s) detected and overlaid.` : "No long page rules were detected."}
        {" "}Compare with the source; exact table cells are not inferred.
      </p>
      <svg
        className="ocr-layout-page"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        role="img"
        aria-label="Positioned OCR text layout"
        preserveAspectRatio="xMidYMid meet"
      >
        <rect width={imageWidth} height={imageHeight} fill="white" />
        {rules.map((rule, index) => (
          <line
            key={`${rule.orientation}-${index}`}
            className="ocr-table-rule"
            x1={rule.x1}
            y1={rule.y1}
            x2={rule.x2}
            y2={rule.y2}
          />
        ))}
        {positionedWords.map((word, index) => {
          const confidence = typeof word.confidence === "number" ? word.confidence : null;
          return (
            <text
              key={`${word.text}-${index}`}
              className="ocr-page-text"
              x={word.x}
              y={(word.y ?? 0) + (word.height ?? 0) * 0.82}
              fontSize={Math.max(
                6,
                Math.min(
                  (word.height ?? 0) * 0.72,
                  (word.width ?? 0) / Math.max(1, word.text.length * 0.58),
                ),
              )}
            >
              <title>{confidence === null ? "Word confidence not reported" : `OCR confidence ${confidence.toFixed(1)} / 100`}</title>
              {word.text}
            </text>
          );
        })}
      </svg>
      {missingFields.length > 0 && (
        <div className="ocr-missing-fields">
          <strong>Expected fields not found in OCR</strong>
          <span>{missingFields.join(" · ")}</span>
        </div>
      )}
    </>
  );
}

function ExtractedFieldTable({ fields }: { fields: Field[] }) {
  if (!fields.length) {
    return <div className="ocr-empty">No labeled fields were extracted for this page. Use Positioned OCR or Plain text to review the available OCR.</div>;
  }
  return (
    <div className="table-scroll ocr-fields-scroll">
      <table className="ocr-fields-table">
        <thead><tr><th>FIELD</th><th>EXTRACTED VALUE</th><th>STATUS</th></tr></thead>
        <tbody>
          {fields.map((field) => {
            const missing = field.availability === "not_available";
            return (
              <tr className={missing ? "missing" : ""} key={field.id}>
                <td>{humanize(field.field_name)}</td>
                <td>{field.display_value ?? (missing ? "Not available" : field.value ?? "No value extracted")}</td>
                <td><span className={`ocr-field-status ${missing ? "missing" : ""}`}>{missing ? "NOT AVAILABLE" : humanize(field.verification_status)}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Dashboard() {
  const client = useQueryClient();
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [imageBatchError, setImageBatchError] = useState("");
  const [draggingImages, setDraggingImages] = useState(false);
  const [formattedReportError, setFormattedReportError] = useState("");
  const stats = useQuery({ queryKey: ["dashboard"], queryFn: () => api<DashboardData>("/dashboard"), refetchInterval: 3000 });
  const ingestions = useQuery({ queryKey: ["ingestions"], queryFn: () => api<Ingestion[]>("/ingestions"), refetchInterval: 3000 });
  const upload = useMutation({
    mutationFn: (file: File) => uploadPdf(file),
    onSuccess: () => { client.invalidateQueries(); },
  });
  const retry = useMutation({
    mutationFn: (id: string) => api(`/ingestions/${id}/retry`, { method: "POST" }),
    onSuccess: () => { client.invalidateQueries({ queryKey: ["ingestions"] }); client.invalidateQueries({ queryKey: ["dashboard"] }); },
  });
  const reprocessFields = useMutation({
    mutationFn: (id: string) => api(`/ingestions/${id}/reprocess-fields`, { method: "POST" }),
    onSuccess: () => { client.invalidateQueries(); },
  });
  const deleteIngestion = useMutation({
    mutationFn: (id: string) => api(`/ingestions/${id}`, { method: "DELETE" }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["ingestions"] });
      client.invalidateQueries({ queryKey: ["dashboard"] });
      client.invalidateQueries({ queryKey: ["documents"] });
      client.invalidateQueries({ queryKey: ["cases"] });
      client.invalidateQueries({ queryKey: ["findings"] });
    },
  });
  const imageUpload = useMutation({
    mutationFn: (files: File[]) => uploadImages(files),
    onSuccess: () => { setImageFiles([]); setImageBatchError(""); client.invalidateQueries(); },
  });
  const imagePreviews = useMemo(
    () => imageFiles.map((file) => ({ file, url: URL.createObjectURL(file) })),
    [imageFiles],
  );
  useEffect(() => () => imagePreviews.forEach((preview) => URL.revokeObjectURL(preview.url)), [imagePreviews]);
  function addImageFiles(selected: File[]) {
    setImageBatchError("");
    const combined = [...imageFiles, ...selected];
    if (combined.length > 200) {
      setImageBatchError("Select no more than 200 images in one batch.");
      return;
    }
    if (combined.some((file) => !/\.(png|jpe?g)$/i.test(file.name) || !["image/png", "image/jpeg"].includes(file.type))) {
      setImageBatchError("Only valid PNG, JPG, and JPEG images are supported.");
      return;
    }
    if (combined.reduce((size, file) => size + file.size, 0) > 50 * 1024 * 1024) {
      setImageBatchError("The complete image batch must be 50 MB or smaller.");
      return;
    }
    setImageFiles(combined);
  }
  function downloadFormattedReport(item: Ingestion) {
    setFormattedReportError("");
    const filename = `${item.filename.replace(/\.pdf$/i, "")}-formatted-report.pdf`;
    void download(`/ingestions/${item.id}/report.pdf`, filename).catch((error: unknown) => {
      setFormattedReportError(error instanceof Error ? error.message : "Could not download the formatted PDF report.");
    });
  }
  const metrics = stats.data ? [
    { title: "Ingestions", value: stats.data.ingestions, icon: Files, tint: "blue", note: "PDF bundles received" },
    { title: "Pages processed", value: stats.data.pages, icon: FileText, tint: "violet", note: "Source pages indexed" },
    { title: "Documents identified", value: stats.data.documents, icon: FileCheck2, tint: "green", note: "Logical document groups" },
    { title: "Cases", value: stats.data.cases, icon: BriefcaseBusiness, tint: "amber", note: "Conservative associations" },
  ] : [];
  return <>
    <PageTitle eyebrow="WORKSPACE OVERVIEW" title="Good morning, reviewer" description="Review local case files, resolve uncertain links, and keep every decision traceable." action={<div className="upload-actions"><label className={`button button-primary ${upload.isPending ? "disabled" : ""}`}><Upload size={16} /> Upload PDF<input type="file" accept="application/pdf,.pdf" hidden disabled={upload.isPending} onChange={(event) => { const file = event.target.files?.[0]; if (file) upload.mutate(file); event.currentTarget.value = ""; }} /></label><label className="button button-secondary"><ImagePlus size={16} /> Add images<input type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" multiple hidden onChange={(event) => { addImageFiles(Array.from(event.target.files ?? [])); event.currentTarget.value = ""; }} /></label></div>} />
    {upload.error && <div className="notice notice-error"><AlertCircle size={16} />{upload.error.message}</div>}
    {upload.isSuccess && <div className="notice notice-success"><Check size={16} />{upload.data.duplicate ? "This exact PDF is already in the workspace." : "PDF received. Local processing has started."}{upload.data.comparison_run_id && <> <Link to={`/comparisons/${upload.data.comparison_run_id}`} className="text-link">Open engine run <ArrowRight size={13} /></Link></>}</div>}
    {upload.data?.comparison_run_id && <div className="notice notice-info">Casework's existing OCR and review workflow still runs. Selected document engines run separately against the original PDF; comparison mode is sequential to limit resource usage.</div>}
    {deleteIngestion.error && <div className="notice notice-error"><AlertCircle size={16} />{(deleteIngestion.error as Error).message}</div>}
    {formattedReportError && <div className="notice notice-error"><AlertCircle size={16} />{formattedReportError}</div>}
    {reprocessFields.error && <div className="notice notice-error"><AlertCircle size={16} />{(reprocessFields.error as Error).message}</div>}
    {imageUpload.error && <div className="notice notice-error"><AlertCircle size={16} />{imageUpload.error.message}</div>}
    {imageUpload.isSuccess && <div className="notice notice-success"><Check size={16} />{imageUpload.data.duplicate ? "This exact image batch is already in the workspace." : `${imageUpload.data.page_count} image page(s) received as one bundle.`}</div>}
    {stats.isError && <ErrorBanner message={(stats.error as Error).message} />}
    <div className="metric-grid">
      {metrics.map(({ title, value, icon: Icon, tint, note }) => <div className="metric-card" key={title}><div className={`metric-icon ${tint}`}><Icon size={18} /></div><div className="metric-title">{title}</div><div className="metric-value">{value.toLocaleString()}</div><div className="metric-note">{note}</div></div>)}
      {!stats.data && !stats.isError && [0,1,2,3].map((n) => <div key={n} className="metric-card skeleton" />)}
    </div>
    <div className="overview-grid">
      <section className="panel ingestion-panel">
        <div className="panel-heading"><div><h2>Recent ingestions</h2><p>Processing status for uploaded PDF bundles</p></div><Link to="/documents" className="text-link">View documents <ArrowRight size={14} /></Link></div>
        <div className="table-scroll"><table><thead><tr><th>FILE</th><th>RECEIVED</th><th>STATUS</th><th>PROGRESS</th><th>ACTION</th></tr></thead><tbody>
          {ingestions.data?.length ? ingestions.data.slice(0, 6).map((item) => <tr key={item.id}>
            <td><div className="file-cell"><span className="file-icon"><FileText size={16} /></span><span className="filename" title={item.filename}>{item.filename}</span></div></td>
            <td className="muted-cell">{item.source_type === "image" ? "Image page bundle" : "PDF upload"}</td>
            <td><StatusPill value={item.status} />{item.status === "failed" && <button className="retry-button" disabled={retry.isPending} onClick={() => retry.mutate(item.id)}>Retry</button>}</td>
            <td><div className="progress-cell"><div className="progress-track"><span style={{ width: `${item.progress}%` }} /></div><small>{item.progress}%</small>{item.status === "processing" && item.stage && <small className="processing-stage">{item.stage.replace(/_/g, " ")}</small>}</div></td>
            <td>
              {["completed", "completed_with_warnings"].includes(item.status) && <button className="button button-secondary button-small reprocess-button" title="Download positioned OCR page images with extracted text and red missing-field flags" onClick={() => downloadFormattedReport(item)}><FileDown size={13} /><span>Formatted PDF</span></button>}
              {["completed", "completed_with_warnings"].includes(item.status) && <button className="button button-secondary button-small reprocess-button" title="Reclassify and re-extract from saved OCR text" disabled={reprocessFields.isPending} onClick={() => reprocessFields.mutate(item.id)}><RotateCcw size={13} /><span>Re-extract</span></button>}
              <button
                className="action-delete-button"
                title={`Delete ${item.filename}`}
                disabled={deleteIngestion.isPending}
                onClick={() => {
                  if (window.confirm(`Are you sure you want to delete "${item.filename}" and its associated pages/documents?`)) {
                    deleteIngestion.mutate(item.id);
                  }
                }}
              >
                <Trash2 size={14} />
                <span>Delete</span>
              </button>
            </td>
          </tr>) : <tr><td colSpan={5}><EmptyState title="No documents yet" detail="Upload a PDF bundle or a batch of page images to start a local review." /></td></tr>}
        </tbody></table></div>
      </section>
      <section className="panel review-queue">
        <div className="panel-heading"><div><h2>Review queue</h2><p>Items that may need your attention</p></div><span className="queue-count">{stats.data?.documents_requiring_review ?? 0}</span></div>
        <div className="queue-item"><div className="queue-icon review"><Clock3 size={17} /></div><div><b>Documents to review</b><span>Confirm classification and extracted fields</span></div><strong>{stats.data?.documents_requiring_review ?? 0}</strong></div>
        <div className="queue-item"><div className="queue-icon unknown"><CircleHelp size={17} /></div><div><b>Unclassified pages</b><span>Unknown category or no matching cues</span></div><strong>{stats.data?.unclassified_pages ?? 0}</strong></div>
        <div className="queue-item"><div className="queue-icon failed"><AlertCircle size={17} /></div><div><b>Failed processing jobs</b><span>Check details and retry when safe</span></div><strong>{stats.data?.failed_jobs ?? 0}</strong></div>
        <Link className="queue-footer" to="/findings">Open findings <ArrowRight size={14} /></Link>
      </section>
    </div>
    <section className="panel upload-panel">
      <div className="upload-illustration"><Upload size={22} /></div><div><b>Keep your case files connected</b><span>Upload a PDF bundle. Pages are processed locally and remain separate until evidence supports a link.</span></div>
      <label className="button button-secondary">Select a PDF<input type="file" accept="application/pdf,.pdf" hidden disabled={upload.isPending} onChange={(event) => { const file = event.target.files?.[0]; if (file) upload.mutate(file); event.currentTarget.value = ""; }} /></label>
    </section>
    <section className="panel image-upload-panel">
      <div className="panel-heading"><div><h2>Upload page images</h2><p>Choose PNG/JPG pages together as one bundle. Pages are still grouped by the normal review pipeline.</p></div><ImagePlus size={18} /></div>
      <div className={`image-drop-zone ${draggingImages ? "drag-active" : ""}`}
        onDragOver={(event) => { event.preventDefault(); setDraggingImages(true); }}
        onDragLeave={() => setDraggingImages(false)}
        onDrop={(event) => { event.preventDefault(); setDraggingImages(false); addImageFiles(Array.from(event.dataTransfer.files)); }}>
        <ImagePlus size={20} /><span>Drop PNG/JPG pages here, or</span>
        <label className="button button-secondary button-small">Choose images<input type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" multiple hidden onChange={(event) => { addImageFiles(Array.from(event.target.files ?? [])); event.currentTarget.value = ""; }} /></label>
        <small>Up to 200 pages, 50 MB total; images are uploaded to this local server.</small>
      </div>
      {imageBatchError && <div className="notice notice-error image-batch-notice"><AlertCircle size={15} />{imageBatchError}</div>}
      {imagePreviews.length > 0 && <div className="image-preview-list">
        {imagePreviews.map(({ file, url }, index) => <div className="image-preview-card" key={`${file.name}-${file.lastModified}-${index}`}>
          <img src={url} alt={`Preview of ${file.name}`} />
          <div className="image-preview-meta"><b title={file.name}>{file.name}</b><span>{formatBytes(file.size)}</span><small>{imageUpload.isPending ? "Uploading…" : "Ready to upload"}</small></div>
          <button className="icon-button" disabled={imageUpload.isPending} aria-label={`Remove ${file.name}`} onClick={() => setImageFiles((current) => current.filter((_, itemIndex) => itemIndex !== index))}><X size={15} /></button>
        </div>)}
      </div>}
      <div className="image-upload-footer">
        <span>{imageFiles.length} image page(s) selected · one bundle</span>
        <button className="button button-primary" disabled={!imageFiles.length || imageUpload.isPending} onClick={() => imageUpload.mutate(imageFiles)}>
          {imageUpload.isPending ? "Uploading…" : `Upload ${imageFiles.length || ""} image${imageFiles.length === 1 ? "" : "s"}`}
        </button>
      </div>
    </section>
  </>;
}

function ComparisonRuns() {
  const query = useQuery({
    queryKey: ["comparisons"],
    queryFn: () => api<ComparisonRun[]>("/comparisons"),
    refetchInterval: 3000,
  });
  return <>
    <PageTitle eyebrow="DOCUMENT ENGINE EVALUATION" title="Engine comparisons" description="Review persisted runs and compare original engine outputs without merging them into the Casework OCR result." />
    {query.isError && <ErrorBanner message={(query.error as Error).message} />}
    <section className="panel comparison-runs">
      <div className="panel-heading"><div><h2>Processing runs</h2><p>Each run records the uploaded PDF hash and independent engine status.</p></div></div>
      {query.data?.length ? query.data.map((run) => <Link className="comparison-run-row" to={`/comparisons/${run.id}`} key={run.id}>
        <div><b>{humanize(run.mode)}</b><small>PDF {run.ingestion_id.slice(0, 8)} · SHA-256 {run.source_sha256.slice(0, 16)}…</small></div>
        <div className="comparison-run-engines">{run.engines.map((engine) => <span key={engine.engine}>{humanize(engine.engine)} <StatusPill value={engine.status} /></span>)}</div>
        <StatusPill value={run.status} />
      </Link>) : !query.isLoading && <EmptyState title="No comparison runs" detail="No document-engine comparison runs have been recorded for this workspace." />}
      {query.isLoading && <div className="loading">Loading document-engine runs…</div>}
    </section>
  </>;
}

function ComparisonReview() {
  const { id = "" } = useParams();
  const [activePage, setActivePage] = useState(1);
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceError, setSourceError] = useState("");
  const [actionError, setActionError] = useState("");
  const query = useQuery({
    queryKey: ["comparison", id],
    queryFn: () => api<ComparisonDetail>(`/comparisons/${id}`),
    refetchInterval: (state) => ["queued", "processing"].includes(state.state.data?.status ?? "") ? 2500 : false,
  });
  const run = query.data;
  const pp = run?.engines.find((engine) => engine.engine === "pp_structure");
  const docling = run?.engines.find((engine) => engine.engine === "docling");
  const comparedPage = run?.page_comparison?.pages.find((page) => page.page_number === activePage);
  const ppPage = pp?.normalized?.pages.find((page) => page.page_number === activePage);
  const doclingPage = docling?.normalized?.pages.find((page) => page.page_number === activePage);
  const pageNumbers = [...new Set([
    ...(pp?.normalized?.pages.map((page) => page.page_number) ?? []),
    ...(docling?.normalized?.pages.map((page) => page.page_number) ?? []),
  ])].sort((a, b) => a - b);
  useEffect(() => {
    if (pageNumbers.length && !pageNumbers.includes(activePage)) setActivePage(pageNumbers[0]);
  }, [activePage, pageNumbers.join(",")]);
  useEffect(() => {
    if (!run) return;
    let active = true;
    let objectUrl = "";
    setSourceUrl("");
    setSourceError("");
    const token = sessionStorage.getItem("casework-token");
    void fetch(`${apiBase}/ingestions/${run.ingestion_id}/source`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    }).then((response) => {
      if (!response.ok) throw new Error(`Original PDF request failed (${response.status})`);
      return response.blob();
    }).then((blob) => {
      if (!active) return;
      objectUrl = URL.createObjectURL(blob);
      setSourceUrl(objectUrl);
    }).catch((error: unknown) => {
      if (active) setSourceError(error instanceof Error ? error.message : "Could not load the original PDF.");
    });
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [run?.id, run?.ingestion_id]);
  async function saveArtifact(path: string | null, filename: string) {
    if (!path) return;
    setActionError("");
    try { await download(path, filename); }
    catch (error) { setActionError(error instanceof Error ? error.message : "Artifact download failed."); }
  }
  if (query.isLoading) return <div className="loading">Loading comparison…</div>;
  if (query.isError || !run) return <><ErrorBanner message={(query.error as Error)?.message ?? "Comparison not found"} /><Link to="/comparisons" className="button button-secondary">Back to engine comparisons</Link></>;
  return <>
    <div className="review-top"><Link to="/comparisons" className="back-link">← Engine comparisons</Link><span className="review-id">Run {run.id.slice(0, 8)}</span><div className="review-actions"><StatusPill value={run.status} /></div></div>
    <PageTitle eyebrow="DOCUMENT ENGINE EVALUATION" title="Side-by-side results" description="Outputs are retained independently. Native confidence values are engine-specific; accuracy remains unmeasured without verified ground truth." />
    {run.error && <ErrorBanner message={run.error} />}
    {actionError && <ErrorBanner message={actionError} />}
    <section className="panel comparison-metadata">
      <div><span>Source hash</span><code>{run.source_sha256}</code></div>
      <div><span>Processing mode</span><b>{humanize(run.mode)}</b></div>
      <div><span>Accuracy</span><b>Not Measured — no verified reference data</b></div>
      <div className="comparison-downloads">
        {pp?.native_output_url && <button className="button button-secondary button-small" onClick={() => void saveArtifact(pp.native_output_url, `pp-structure-${run.id}.json`)}>PP native JSON</button>}
        {pp?.markdown_url && <button className="button button-secondary button-small" onClick={() => void saveArtifact(pp.markdown_url, `pp-structure-${run.id}.md`)}>PP Markdown</button>}
        {docling?.native_output_url && <button className="button button-secondary button-small" onClick={() => void saveArtifact(docling.native_output_url, `docling-${run.id}.json`)}>Docling JSON</button>}
        {docling?.markdown_url && <button className="button button-secondary button-small" onClick={() => void saveArtifact(docling.markdown_url, `docling-${run.id}.md`)}>Docling Markdown</button>}
        {run.page_comparison_url && <button className="button button-secondary button-small" onClick={() => void saveArtifact(run.page_comparison_url, `comparison-${run.id}.json`)}>Comparison JSON</button>}
      </div>
    </section>
    <div className="comparison-page-nav">
      <label>Source page<select value={activePage} onChange={(event) => setActivePage(Number(event.target.value))}>{pageNumbers.map((pageNumber) => <option key={pageNumber} value={pageNumber}>{pageNumber}</option>)}</select></label>
      <span>{run.page_comparison?.accuracy ?? "Single-engine run; cross-engine accuracy comparison not available."}</span>
    </div>
    <div className="comparison-columns">
      <section className="panel comparison-column"><div className="panel-heading"><div><h2>Original PDF</h2><p>Page {activePage} · same source SHA-256 recorded for both engines</p></div></div>
        <div className="comparison-source">{sourceUrl ? <iframe key={`${sourceUrl}-${activePage}`} src={`${sourceUrl}#page=${activePage}&toolbar=0&navpanes=0`} title={`Original PDF page ${activePage}`} /> : <span className={sourceError ? "form-error" : "muted-cell"}>{sourceError || "Loading source PDF…"}</span>}</div>
      </section>
      <ComparisonEnginePanel engine={pp} page={ppPage} title="PP-StructureV3" />
      <ComparisonEnginePanel engine={docling} page={doclingPage} title="Docling" />
    </div>
    <section className="panel comparison-differences">
      <div className="panel-heading"><div><h2>Page {activePage} text differences</h2><p>{comparedPage?.text_equal_after_whitespace_normalization ? "Texts match after whitespace normalization." : "Differences below are text changes, not accuracy percentages."}</p></div></div>
      {comparedPage ? <div className="difference-grid">
        {([
          ["Word", comparedPage.word_differences],
          ["Line", comparedPage.line_differences],
          ["Paragraph", comparedPage.paragraph_differences],
          ["Character", comparedPage.character_differences],
        ] as const).map(([label, differences]) => <details className="difference-section" key={label}>
          <summary>{label} differences ({differences.length})</summary>
          {differences.length ? differences.slice(0, 300).map((difference, index) => <div className="difference-item" key={`${label}-${index}`}><b>{humanize(difference.operation)}</b><span><strong>PP-StructureV3:</strong> {difference.left || "∅"}</span><span><strong>Docling:</strong> {difference.right || "∅"}</span></div>) : <p>No differences at this level.</p>}
        </details>)}
      </div> : <div className="empty-inline">Text comparison will appear when both engines have completed successfully.</div>}
    </section>
  </>;
}

function ComparisonEnginePanel({ engine, page, title }: {
  engine: ComparisonDetail["engines"][number] | undefined;
  page: { page_number: number; text: string; blocks: Array<Record<string, unknown>>; tables: Array<Record<string, unknown>> } | undefined;
  title: string;
}) {
  return <section className="panel comparison-column">
    <div className="panel-heading"><div><h2>{title}</h2><p>{engine?.package_version ? `Version ${engine.package_version}` : "Engine status"}</p></div><StatusPill value={engine?.status ?? "not_selected"} /></div>
    {engine?.error && <div className="comparison-error">{engine.error}</div>}
    {engine ? <div className="comparison-engine-body">
      <div className="comparison-engine-meta"><span>Pages: {engine.page_count || "Not available"}</span><span>Duration: {engine.processing_seconds === null ? "Not available" : `${engine.processing_seconds.toFixed(2)} s`}</span></div>
      <h3>Extracted page text</h3>
      <pre className="comparison-text">{page?.text || (engine.status === "completed" ? "No page text was emitted by this engine." : "Waiting for engine output…")}</pre>
      <details><summary>Layout blocks ({page?.blocks.length ?? 0})</summary><pre className="comparison-json">{JSON.stringify(page?.blocks ?? [], null, 2)}</pre></details>
      <details><summary>Tables ({page?.tables.length ?? 0})</summary><pre className="comparison-json">{JSON.stringify(page?.tables ?? [], null, 2)}</pre></details>
      <details><summary>Native confidence metadata</summary><pre className="comparison-json">{JSON.stringify(engine.confidence, null, 2)}</pre></details>
      <details><summary>Processing configuration</summary><pre className="comparison-json">{JSON.stringify(engine.config, null, 2)}</pre></details>
    </div> : <div className="empty-inline">This engine was not selected for this run.</div>}
  </section>;
}

function Documents() {
  const [search, setSearch] = useState("");
  const client = useQueryClient();
  const query = useQuery({ queryKey: ["documents", search], queryFn: () => api<{ items: Document[]; total: number }>(`/documents?q=${encodeURIComponent(search)}`) });
  const deleteIngestion = useMutation({
    mutationFn: (ingestionId: string) => api(`/ingestions/${ingestionId}`, { method: "DELETE" }),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["documents"] });
      client.invalidateQueries({ queryKey: ["dashboard"] });
      client.invalidateQueries({ queryKey: ["ingestions"] });
      client.invalidateQueries({ queryKey: ["cases"] });
      client.invalidateQueries({ queryKey: ["findings"] });
    },
  });
  return <>
    <PageTitle eyebrow="DOCUMENT INVENTORY" title="Documents" description="Browse proposed document groupings and inspect classification evidence." />
    {deleteIngestion.error && <div className="notice notice-error"><AlertCircle size={16} />{(deleteIngestion.error as Error).message}</div>}
    <section className="panel inventory-panel">
      <div className="inventory-toolbar"><div className="searchbox"><Search size={16} /><input placeholder="Search classification evidence" value={search} onChange={(event) => setSearch(event.target.value)} /></div><span className="result-count">{query.data?.total ?? 0} documents</span></div>
      {query.isError && <ErrorBanner message={(query.error as Error).message} />}
      <div className="table-scroll"><table><thead><tr><th>DOCUMENT</th><th>TYPE</th><th>PAGE RANGE</th><th>CASE LINK</th><th>REVIEW</th><th>ACTIONS</th></tr></thead><tbody>
        {query.data?.items.length ? query.data.items.map((doc) => <tr key={doc.id}>
          <td><Link to={`/documents/${doc.id}`} className="document-link"><span className="file-icon"><FileText size={16} /></span><span><b>{doc.id.slice(0, 8)}</b><small>{doc.classification_evidence || "No classification evidence"}</small></span></Link></td>
          <td><span className="category-tag">{humanize(doc.category)}</span></td>
          <td className="muted-cell">Pages {doc.start_page}{doc.end_page !== doc.start_page ? `–${doc.end_page}` : ""}</td>
          <td><StatusPill value={doc.case_link_status} /></td>
          <td><StatusPill value={doc.review_status} /></td>
          <td>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Link to={`/documents/${doc.id}`} className="row-arrow" title="Inspect document"><ArrowRight size={16} /></Link>
              <button
                className="action-delete-button"
                title="Delete document and its source bundle"
                disabled={deleteIngestion.isPending}
                onClick={() => {
                  if (window.confirm(`Are you sure you want to delete this document (${doc.id.slice(0, 8)}) and its upload?`)) {
                    deleteIngestion.mutate(doc.ingestion_id);
                  }
                }}
              >
                <Trash2 size={14} />
              </button>
            </div>
          </td>
        </tr>) : <tr><td colSpan={6}><EmptyState title="No documents identified" detail="Uploaded documents will appear here after processing." /></td></tr>}
      </tbody></table></div>
    </section>
  </>;
}

function Review() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [activePage, setActivePage] = useState(1);
  const [ocrDraft, setOcrDraft] = useState<string | null>(null);
  const [ocrView, setOcrView] = useState<"fields" | "text" | "words" | "layout">("layout");
  const [message, setMessage] = useState("");
  const [formattedReportError, setFormattedReportError] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceError, setSourceError] = useState("");
  const [sourceDimensions, setSourceDimensions] = useState<{ width: number; height: number } | null>(null);
  const query = useQuery({ queryKey: ["document", id], queryFn: () => api<Document>(`/documents/${id}`) });
  const audit = useQuery({ queryKey: ["document-audit", id], queryFn: () => api<AuditEntry[]>(`/documents/${id}/audit`) });
  const doc = query.data;
  useEffect(() => { if (doc?.pages?.length && !doc.pages.some((page) => page.page_number === activePage)) setActivePage(doc.pages[0].page_number); }, [doc, activePage]);
  const page = doc?.pages?.find((item) => item.page_number === activePage);
  const ocrWords = useMemo(() => parseOcrWords(page), [page]);
  const ocrTableRules = useMemo(() => parseOcrTableRules(page), [page]);
  const missingExpectedFields = useMemo(
    () => [...new Set(doc?.fields
      .filter((field) => field.availability === "not_available" || field.verification_status === "not_available")
      .map((field) => humanize(field.field_name)) ?? [])],
    [doc?.fields],
  );
  const tableRulesQuery = useQuery({
    queryKey: ["page-table-lines", doc?.id, activePage],
    queryFn: () => {
      if (!doc) throw new Error("Document data is unavailable.");
      return api<{ table_lines: OcrTableRule[] }>(`/documents/${doc.id}/pages/${activePage}/table-lines`);
    },
    enabled: Boolean(doc && page && ocrView === "layout" && !ocrTableRules.length),
    retry: false,
  });
  const positionedRules = ocrTableRules.length ? ocrTableRules : tableRulesQuery.data?.table_lines ?? [];
  const saveField = useMutation({
    mutationFn: ({ field, value }: { field: Field; value: string }) => api(`/fields/${field.id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ value, verification_status: "verified" }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["document-audit", id] }); setMessage("Field correction saved to the review history."); },
  });
  const saveOcr = useMutation({
    mutationFn: () => api(`/pages/${page?.id}/ocr`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ corrected_text: ocrDraft }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["document-audit", id] }); setOcrDraft(null); setMessage("OCR correction saved. Original OCR is preserved."); },
  });
  const saveCategory = useMutation({
    mutationFn: (category: string) => api(`/documents/${id}/category`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ category }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["document-audit", id] }); },
  });
  const formattedReport = useMutation({
    mutationFn: async () => {
      if (!doc) throw new Error("Document data is unavailable.");
      return download(
        `/ingestions/${doc.ingestion_id}/report.pdf`,
        `casework-${doc.ingestion_id.slice(0, 8)}-formatted-report.pdf`,
      );
    },
    onSuccess: () => setMessage("Formatted PDF report downloaded."),
    onError: (error: Error) => setFormattedReportError(error.message),
  });
  const cases = useQuery({ queryKey: ["cases"], queryFn: () => api<CaseItem[]>("/cases") });
  const linkedCases = cases.data?.filter((item) => item.id !== doc?.case_id) ?? [];
  const linkCase = useMutation({
    mutationFn: (body: { case_id?: string; decision: "approved" | "rejected" }) => api(`/documents/${id}/case-link`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["document-audit", id] }); qc.invalidateQueries({ queryKey: ["cases"] }); },
  });
  const documents = useQuery({ queryKey: ["documents", "all"], queryFn: () => api<{ items: Document[] }>("/documents?page_size=100") });
  const adjacent = documents.data?.items.find((item) => item.ingestion_id === doc?.ingestion_id && (item.end_page + 1 === doc?.start_page || doc?.end_page + 1 === item.start_page));
  const split = useMutation({
    mutationFn: () => api(`/documents/${id}/split`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ split_before_page: activePage }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["documents"] }); setMessage(`Document split before page ${activePage}.`); },
  });
  const merge = useMutation({
    mutationFn: () => api(`/documents/${id}/merge`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ other_document_id: adjacent?.id }) }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["document", id] }); qc.invalidateQueries({ queryKey: ["documents"] }); setMessage("Adjacent documents merged."); },
  });
  useEffect(() => {
    let active = true;
    let objectUrl = "";
    if (doc) {
      setSourceUrl("");
      setSourceError("");
      setSourceDimensions(null);
      const token = sessionStorage.getItem("casework-token");
      const sourcePath = `/documents/${doc.id}/pages/${activePage}/image`;
      void fetch(`${apiBase}${sourcePath}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      }).then((response) => {
        if (!response.ok) throw new Error(`Source page request failed (${response.status})`);
        return response.blob();
      }).then((blob) => {
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setSourceUrl(objectUrl);
      }).catch((error: unknown) => {
        if (!active) return;
        setSourceError(error instanceof Error ? error.message : "Could not load the source page.");
      });
    }
    return () => { active = false; if (objectUrl) URL.revokeObjectURL(objectUrl); };
  }, [doc?.id, doc?.ingestion_id, doc?.source_type, activePage]);
  if (query.isLoading) return <div className="loading">Loading document review…</div>;
  if (query.isError || !doc) return <><ErrorBanner message={(query.error as Error)?.message ?? "Document not found"} /><button className="button button-secondary" onClick={() => navigate("/documents")}>Back to documents</button></>;
  return <>
    <div className="review-top"><Link to="/documents" className="back-link">← Documents</Link><span className="review-id">Document {doc.id.slice(0, 8)}</span><div className="review-actions"><button className="button button-secondary" disabled={formattedReport.isPending} onClick={() => formattedReport.mutate()}><FileDown size={15} /> Download formatted PDF</button><StatusPill value={doc.review_status} /><button className="button button-primary" onClick={() => { saveCategory.mutate(doc.category); setMessage("Review state updated."); }}> <Check size={15} /> Mark reviewed</button></div></div>
    <PageTitle eyebrow="DOCUMENT REVIEW" title={`Pages ${doc.start_page}${doc.end_page !== doc.start_page ? `–${doc.end_page}` : ""}`} description="Compare the source page with its extracted text and evidence." />
    {message && <div className="notice notice-success"><Check size={16} />{message}<button className="close-notice" onClick={() => setMessage("")}><X size={15} /></button></div>}
    {formattedReportError && <div className="notice notice-error"><AlertCircle size={16} />{formattedReportError}</div>}
    <div className="review-layout">
      <section className="panel source-panel"><div className="panel-heading"><div><h2>Source page</h2><p>{doc.source_type === "image" ? "Original uploaded image" : "Original PDF page"} · page {activePage}</p></div><div className="page-stepper"><button disabled={activePage <= doc.start_page} onClick={() => setActivePage(activePage - 1)}>‹</button><span>{activePage} / {doc.end_page}</span><button disabled={activePage >= doc.end_page} onClick={() => setActivePage(activePage + 1)}>›</button></div></div>
        <div className="review-boundary-actions">
          {activePage > doc.start_page && <button className="button button-secondary button-small" disabled={split.isPending} onClick={() => split.mutate()}>Split before page {activePage}</button>}
          {adjacent && <button className="button button-secondary button-small" disabled={merge.isPending} onClick={() => merge.mutate()}>Merge adjacent pages</button>}
          {(split.error || merge.error) && <span className="form-error">{(split.error ?? merge.error)?.message}</span>}
        </div>
        <div className="source-image-wrap">{page && (sourceUrl ? <img src={sourceUrl} alt={`Original source page ${activePage}`} onLoad={(event) => setSourceDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} /> : <span className={sourceError ? "form-error" : "muted-cell"}>{sourceError || "Loading source page…"}</span>)}</div>
      </section>
      <div className="review-right">
        <section className="panel extracted-panel"><div className="panel-heading"><div><h2>Document details</h2><p>Classification and field review</p></div></div>
          <label className="form-label">Document type<select value={doc.category} onChange={(event) => saveCategory.mutate(event.target.value)}><option value="unknown">Unknown / unclassified</option>{["gift_deed","valuation","receipt","identity_document","challan","condominium","survey_record","official_letter","registration","property_tax","tax_invoice"].map((item) => <option key={item} value={item}>{humanize(item)}</option>)}</select></label>
          <div className="classification-evidence"><b>Classification evidence</b><span>{doc.classification_evidence || "No configured category rule matched."}</span><small>Rule-match score {doc.confidence_proxy.toFixed(2)}; heuristic only, not a calibrated probability or accuracy score.</small></div>
          <div className="link-evidence"><b>Case association</b><StatusPill value={doc.case_link_status} /><p>{doc.case_link_evidence ?? "No case association evidence."}</p>{doc.case_id && <Link to="/cases" className="text-link">Open case <ArrowRight size={13} /></Link>}
            {doc.case_link_status === "high_confidence_candidate" && doc.case_id && <div className="link-actions"><button className="button button-primary button-small" onClick={() => linkCase.mutate({ case_id: doc.case_id ?? undefined, decision: "approved" })}><Check size={13} /> Approve link</button><button className="button button-secondary button-small" onClick={() => linkCase.mutate({ decision: "rejected" })}>Keep separate</button></div>}
            {doc.case_link_status === "insufficient_evidence" && linkedCases.length > 0 && <div className="link-actions"><select aria-label="Possible case" id="link-case-select"><option value="">Select a possible case</option>{linkedCases.map((item) => <option value={item.id} key={item.id}>{item.title}</option>)}</select><button className="button button-secondary button-small" onClick={() => { const selectedCase = document.getElementById("link-case-select") as HTMLSelectElement | null; if (selectedCase?.value) linkCase.mutate({ case_id: selectedCase.value, decision: "approved" }); }}>Approve case link</button></div>}
          </div>
          <div className="section-rule" /><h3 className="subheading">Extracted fields <span>{doc.fields.length}</span></h3>
          {doc.fields.length ? doc.fields.map((field) => <FieldEditor field={field} key={field.id} onSave={(value) => saveField.mutate({ field, value })} />) : <p className="empty-inline">No fields were extracted. Review the source and OCR text.</p>}
        </section>
        <section className="panel ocr-panel"><div className="panel-heading"><div><h2>OCR text</h2><p>Original text is retained separately</p></div></div>
          {page?.ocr_error && <div className="ocr-warning"><AlertCircle size={14} />{page.ocr_error}</div>}
          <div className="ocr-confidence"><span>Mean OCR word confidence</span><b>{page?.ocr_confidence_proxy == null ? "Unavailable" : `${page.ocr_confidence_proxy.toFixed(1)} / 100`}</b><small>{page?.confidence_description ?? "No OCR engine confidence was produced for this page."}</small></div>
          <div className="ocr-toggle-row">
            <button type="button" className={ocrView === "fields" ? "button button-primary button-small" : "button button-secondary button-small"} onClick={() => setOcrView("fields")}>Formatted fields</button>
            <button type="button" className={ocrView === "layout" ? "button button-primary button-small" : "button button-secondary button-small"} onClick={() => setOcrView("layout")}>Positioned OCR</button>
            <button type="button" className={ocrView === "words" ? "button button-primary button-small" : "button button-secondary button-small"} onClick={() => setOcrView("words")}>Word view</button>
            <button type="button" className={ocrView === "text" ? "button button-primary button-small" : "button button-secondary button-small"} onClick={() => setOcrView("text")}>Plain text</button>
          </div>
          {ocrView === "fields" ? (
            <ExtractedFieldTable fields={doc.fields} />
          ) : ocrView === "layout" ? (
            <>
              {tableRulesQuery.isError && <div className="ocr-warning"><AlertCircle size={14} />Table-line detection failed: {(tableRulesQuery.error as Error).message}</div>}
              <OcrLayoutView words={ocrWords} rules={positionedRules} rulesLoading={tableRulesQuery.isLoading} missingFields={missingExpectedFields} imageWidth={sourceDimensions?.width ?? null} imageHeight={sourceDimensions?.height ?? null} />
            </>
          ) : ocrView === "words" ? (
            <OcrWordView words={ocrWords} />
          ) : (
            <textarea value={ocrDraft ?? page?.ocr_text ?? ""} onChange={(event) => setOcrDraft(event.target.value)} rows={8} placeholder="No text was extracted from this page." />
          )}
          {page?.ocr_text_original && page.ocr_text !== page.ocr_text_original && <details className="original-ocr"><summary>Show original OCR output</summary><pre>{page.ocr_text_original}</pre></details>}
          {ocrDraft !== null && <button className="button button-secondary button-small" disabled={saveOcr.isPending} onClick={() => saveOcr.mutate()}>Save OCR correction</button>}
        </section>
        <section className="panel audit-panel"><div className="panel-heading"><div><h2>Review history</h2><p>Recorded changes for this document and its pages</p></div></div>
          {audit.data?.length ? audit.data.map((entry) => <div className="audit-entry" key={entry.id}><b>{humanize(entry.action)}</b><span>{entry.actor} · {new Date(entry.created_at).toLocaleString()}</span>{(entry.before_value || entry.after_value) && <small>{entry.before_value ?? "—"} → {entry.after_value ?? "—"}</small>}</div>) : <p className="empty-inline">No review changes recorded yet.</p>}
        </section>
      </div>
    </div>
  </>;
}

function FieldEditor({ field, onSave }: { field: Field; onSave: (value: string) => void }) {
  const [value, setValue] = useState(field.value ?? "");
  const [editing, setEditing] = useState(false);
  useEffect(() => { setValue(field.value ?? ""); }, [field.value]);
  const missing = field.verification_status === "not_available";
  return <div className={`field-card ${missing ? "missing" : ""}`}><div className="field-heading"><b>{humanize(field.field_name)}</b><span className={`verification ${field.verification_status}`}>{missing ? "NOT AVAILABLE" : field.verification_status}</span></div>
    {editing ? <div className="field-edit"><input value={value} onChange={(event) => setValue(event.target.value)} autoFocus /><button className="button button-primary button-small" onClick={() => { onSave(value); setEditing(false); }}>Save</button><button className="icon-button" onClick={() => { setValue(field.value ?? ""); setEditing(false); }} aria-label="Cancel"><X size={15} /></button></div> : <button className="field-value" onClick={() => setEditing(true)}>{field.display_value ?? (value || (field.verification_status === "not_available" ? "Not available" : "Not extracted"))} <span>Edit</span></button>}
    <small>Source page {field.source_page}{field.evidence ? ` · “${field.evidence}”` : ""}</small>
  </div>;
}

function Cases() {
  const query = useQuery({ queryKey: ["cases"], queryFn: () => api<CaseItem[]>("/cases") });
  const [selected, setSelected] = useState<string | null>(null);
  if (query.isLoading) return <div className="loading">Loading cases…</div>;
  return <>
    <PageTitle eyebrow="CASE EXPLORER" title="Cases" description="Cases are associated only when an exact reference is extracted; unlinked records remain separate." />
    {query.isError && <ErrorBanner message={(query.error as Error).message} />}
    <div className="case-layout"><section className="panel case-list-panel"><div className="panel-heading"><div><h2>Case records</h2><p>{query.data?.length ?? 0} cases in this workspace</p></div></div>
      {query.data?.length ? query.data.map((item) => <button key={item.id} className={`case-list-item ${selected === item.id ? "selected" : ""}`} onClick={() => setSelected(item.id)}><span className="case-symbol"><BriefcaseBusiness size={17} /></span><span><b>{item.title}</b><small>{item.reference ? `Reference ${item.reference}` : "No verified reference"} · {item.documents.length} documents</small></span><ArrowRight size={15} /></button>) : <EmptyState title="No cases yet" detail="Upload a PDF to create reviewable case records." />}
    </section>
    <section className="panel case-detail-panel">{(() => {
      const item = query.data?.find((row) => row.id === selected) ?? query.data?.[0];
      if (!item) return <EmptyState title="Select a case" detail="Case relationships will appear here after ingestion." />;
      return <><div className="panel-heading"><div><div className="eyebrow">CASE SUMMARY</div><h2>{item.title}</h2><p>{item.reference ? `Reference ${item.reference}` : "Association requires reviewer confirmation"}</p></div><div className="export-menu"><button className="button button-secondary button-small" onClick={() => void download(`/cases/${item.id}/report?format=json`, `case-${item.id}.json`)}><ArrowDownToLine size={14} /> JSON</button><button className="button button-secondary button-small" onClick={() => void download(`/cases/${item.id}/report?format=csv`, `case-${item.id}.csv`)}>CSV</button></div></div>
        <div className="relationship-summary"><div className="relationship-stat"><span>Documents</span><b>{item.documents.length}</b></div><div className="relationship-stat"><span>People mentioned</span><b>{item.people.length}</b></div><div className="relationship-stat"><span>Properties referenced</span><b>{item.properties.length}</b></div></div>
        <h3 className="subheading">People & properties</h3>
        <div className="entity-grid">
          <div className="entity-list"><b>People mentioned</b>{item.people.length ? item.people.map((person) => <span key={person.id}>{person.name}<small>{humanize(person.relationship)}</small></span>) : <small>No people extracted</small>}</div>
          <div className="entity-list"><b>Properties referenced</b>{item.properties.length ? item.properties.map((property) => <span key={property.id}>{property.identifier}<small>{property.description ?? "Description not extracted"}</small></span>) : <small>No property references extracted</small>}</div>
        </div>
        <h3 className="subheading">Linked documents</h3>{item.documents.length ? item.documents.map((doc) => <Link key={doc.id} to={`/documents/${doc.id}`} className="case-document"><span className="file-icon"><FileText size={16} /></span><span><b>{humanize(doc.category)}</b><small>Pages {doc.start_page}–{doc.end_page} · {doc.case_link_evidence}</small></span><ArrowRight size={15} /></Link>) : <p className="empty-inline">No documents currently associated.</p>}
      </>;
    })()}</section></div>
  </>;
}

function Findings() {
  const query = useQuery({ queryKey: ["findings"], queryFn: () => api<Finding[]>("/findings") });
  const client = useQueryClient();
  const decision = useMutation({ mutationFn: ({ id, status }: { id: string; status: string }) => api(`/findings/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }) }), onSuccess: () => client.invalidateQueries({ queryKey: ["findings"] }) });
  return <>
    <PageTitle eyebrow="VALIDATION & REVIEW" title="Findings" description="Transparent extraction and missing-information flags. These are review prompts, not legal conclusions." />
    {query.isError && <ErrorBanner message={(query.error as Error).message} />}
    <section className="panel findings-panel"><div className="panel-heading"><div><h2>Review findings</h2><p>{query.data?.filter((item) => item.review_status === "open").length ?? 0} open items</p></div></div>
      {query.data?.length ? query.data.map((finding) => <div key={finding.id} className="finding-row"><span className={`finding-icon ${finding.severity}`}><AlertCircle size={17} /></span><div className="finding-main"><div className="finding-title"><b>{humanize(finding.rule_id)}</b><StatusPill value={finding.review_status} /></div><p>{finding.explanation}</p><small>{finding.kind.replaceAll("_", " ")}{finding.source_pages ? ` · source page ${finding.source_pages}` : ""}</small>{finding.document_id && <Link className="text-link" to={`/documents/${finding.document_id}`}>Open source evidence <ArrowRight size={13} /></Link>}</div>
        {finding.review_status === "open" && <div className="finding-controls"><button title="Resolve" onClick={() => decision.mutate({ id: finding.id, status: "resolved" })}><Check size={16} /></button><button title="Dismiss" onClick={() => decision.mutate({ id: finding.id, status: "dismissed" })}><X size={16} /></button></div>}
      </div>) : <EmptyState title="No findings" detail="Processing flags for missing information or OCR failures will show up here." />}
    </section>
  </>;
}

function TokenDialog({ onSave, onClose }: { onSave: (token: string) => void; onClose: () => void }) {
  const [value, setValue] = useState("");
  return <div className="modal-backdrop"><form className="token-dialog" onSubmit={(event) => { event.preventDefault(); onSave(value.trim()); }}><div className="token-icon"><LockKeyhole size={20} /></div><h2>API access</h2><p>Enter the local API bearer token if one is configured in the backend environment.</p><input type="password" value={value} onChange={(event) => setValue(event.target.value)} placeholder="Local API token" autoFocus /><div className="dialog-actions"><button type="button" className="button button-secondary" onClick={onClose}>Cancel</button><button className="button button-primary" type="submit">Save token</button></div></form></div>;
}

function StatusPill({ value }: { value: string }) {
  const tone = ["completed", "reviewed", "verified", "resolved", "approved"].includes(value) ? "success" : ["failed", "incorrect"].includes(value) ? "danger" : ["queued", "processing", "high_confidence_candidate"].includes(value) ? "info" : "warning";
  return <span className={`status-pill ${tone}`}><span />{humanize(value)}</span>;
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><div><Files size={20} /></div><b>{title}</b><span>{detail}</span></div>;
}

function ErrorBanner({ message }: { message: string }) {
  return <div className="notice notice-error"><AlertCircle size={16} />{message}</div>;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
