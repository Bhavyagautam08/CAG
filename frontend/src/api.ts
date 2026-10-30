export const apiBase = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000/api";

function headers(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const token = sessionStorage.getItem("casework-token");
  if (token) headers.set("Authorization", `Bearer ${token}`);
  return headers;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBase}${path}`, { ...init, headers: headers(init?.headers) });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error ?? body.detail ?? `Request failed (${response.status})`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function download(path: string, filename: string): Promise<void> {
  const response = await fetch(`${apiBase}${path}`, { headers: headers() });
  if (!response.ok) throw new Error(`Export failed (${response.status})`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function uploadPdf(file: File): Promise<{ id: string; status: string; duplicate?: boolean; comparison_run_id?: string | null }> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${apiBase}/ingestions`, { method: "POST", headers: headers(), body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? body.error ?? `Upload failed (${response.status})`);
  }
  return response.json();
}

export interface EngineResult {
  engine: "pp_structure" | "docling";
  status: string;
  package_version: string | null;
  config: Record<string, unknown>;
  confidence: Record<string, unknown>;
  processing_seconds: number | null;
  page_count: number;
  error: string | null;
  normalized: {
    pages: Array<{
      page_number: number;
      text: string;
      blocks: Array<Record<string, unknown>>;
      tables: Array<Record<string, unknown>>;
    }>;
  } | null;
  native_output_url: string | null;
  markdown_url: string | null;
}

export interface ComparisonRun {
  id: string;
  ingestion_id: string;
  mode: string;
  source_sha256: string;
  status: string;
  error: string | null;
  created_at: string;
  engines: Array<{
    engine: "pp_structure" | "docling";
    status: string;
    package_version: string | null;
    page_count: number;
    processing_seconds: number | null;
    error: string | null;
  }>;
}

export interface ComparisonDetail extends ComparisonRun {
  engines: EngineResult[];
  page_comparison: {
    accuracy: string;
    confidence_note: string;
    pages: Array<{
      page_number: number;
      pp_structure_text: string;
      docling_text: string;
      text_equal_after_whitespace_normalization: boolean;
      character_differences: Array<{ operation: string; left: string; right: string }>;
      word_differences: Array<{ operation: string; left: string; right: string }>;
      line_differences: Array<{ operation: string; left: string; right: string }>;
      paragraph_differences: Array<{ operation: string; left: string; right: string }>;
      pp_structure_blocks: Array<Record<string, unknown>>;
      docling_blocks: Array<Record<string, unknown>>;
      pp_structure_tables: Array<Record<string, unknown>>;
      docling_tables: Array<Record<string, unknown>>;
    }>;
  } | null;
  page_comparison_url: string | null;
}

export async function uploadImages(files: File[], bundleName?: string): Promise<{ id: string; status: string; duplicate?: boolean; page_count: number; source_type: string }> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  if (bundleName) form.append("bundle_name", bundleName);
  const response = await fetch(`${apiBase}/ingestions/images`, { method: "POST", headers: headers(), body: form });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? body.error ?? `Image upload failed (${response.status})`);
  }
  return response.json();
}

export interface DashboardData {
  ingestions: number;
  pages: number;
  documents: number;
  cases: number;
  documents_requiring_review: number;
  unclassified_pages: number;
  failed_jobs: number;
}

export interface Document {
  id: string;
  ingestion_id: string;
  source_type: "pdf" | "image";
  case_id: string | null;
  start_page: number;
  end_page: number;
  category: string;
  predicted_category: string;
  confidence_proxy: number;
  classification_evidence: string | null;
  review_status: string;
  case_link_status: string;
  case_link_evidence: string | null;
  fields: Field[];
  pages?: Page[];
}

export interface Field {
  id: string;
  field_name: string;
  value: string | null;
  display_value?: string | null;
  availability?: "available" | "not_available";
  extracted_value: string | null;
  verified_value: string | null;
  source_page: number;
  evidence: string | null;
  verification_status: string;
  sensitive: boolean;
}

export interface Page {
  id: string;
  page_number: number;
  ocr_text: string;
  ocr_text_original: string;
  ocr_error: string | null;
  ocr_engine?: string;
  ocr_confidence_proxy?: number | null;
  ocr_word_boxes?: string | null;
  confidence_description?: string | null;
  category: string;
}
