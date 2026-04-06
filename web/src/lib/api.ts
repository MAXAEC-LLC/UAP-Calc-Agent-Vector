const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Helper: fetch with a generous timeout and clear error messages
async function apiFetch(
  url: string,
  init?: RequestInit,
  timeoutMs = 120_000,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out — is the backend running?");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  reply: string;
  sources: { filename: string; distance: number; source_type?: "property" | "document" }[];
}

export interface DocumentInfo {
  filename: string;
  chunks: number;
}

export interface DocumentsResponse {
  documents: DocumentInfo[];
  total_chunks: number;
}

export interface UploadResponse {
  filename: string;
  chunks: number;
}

export async function sendChat(
  messages: ChatMessage[],
  useRag: boolean = true,
): Promise<ChatResponse> {
  const res = await apiFetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, use_rag: useRag }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Chat failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch(
    `${API_BASE}/api/upload`,
    { method: "POST", body: form },
    600_000, // 10 min — large docs need embedding per chunk
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentsResponse> {
  const res = await apiFetch(`${API_BASE}/api/documents`);
  if (!res.ok) throw new Error(`Failed to list documents (${res.status})`);
  return res.json();
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await apiFetch(
    `${API_BASE}/api/documents/${encodeURIComponent(filename)}`,
    { method: "DELETE" },
  );
  if (!res.ok) throw new Error(`Failed to delete document (${res.status})`);
}

export async function checkHealth(): Promise<{ status: string; documents: number }> {
  const res = await apiFetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("Backend unavailable");
  return res.json();
}

// ── Agent Settings ─────────────────────────────────────────────────────

export interface AgentSettingsMap {
  [agentName: string]: {
    [key: string]: number | string;
  };
}

export async function getAgentSettings(): Promise<{ settings: AgentSettingsMap }> {
  const res = await apiFetch(`${API_BASE}/api/settings`);
  if (!res.ok) throw new Error(`Failed to get settings (${res.status})`);
  return res.json();
}

export async function updateAgentSettings(
  settings: AgentSettingsMap,
): Promise<{ settings: AgentSettingsMap }> {
  const res = await apiFetch(`${API_BASE}/api/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ settings }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Update settings failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── Pinecone Index Management ──────────────────────────────────────────

export interface PineconeIndex {
  name: string;
  dimension: number;
  metric: string;
  host: string;
  ready: boolean;
  state: string;
}

export interface IndexListResponse {
  indexes: PineconeIndex[];
  active: string;
}

export interface ActiveIndexResponse {
  name: string;
  total_vectors: number;
  dimension: number;
  namespaces: Record<string, { vector_count: number }>;
}

export interface PropertySearchResult {
  bbl: string;
  address: string;
  borough: string;
  zone: string;
  overlay: string;
  lotArea: number;
  builtFar: number;
  numFloors: number;
  yearBuilt: number;
  bldgClass: string;
  lat: number;
  lng: number;
}

export interface ValidatedLotInfo {
  bbl: string;
  address: string;
  lotArea: number;
  zone: string;
}

export interface BlockLotInfo {
  lot: number;
  address: string;
  lotArea: number;
  zone: string;
}

export interface PropertyScenario {
  code: string;
  label: string;
  max_res_floor_area: number;
  max_number_of_units: number;
  affordable_floor_area: number;
  affordable_floor_area_uap: number;
  affordable_floor_area_485x: number;
  affordable_units_percentage: number;
  affordable_units_total: number;
  market_rate_units: number;
  ami_breakdown: { ami: number; units: number }[];
  triggers_prevailing_wages: boolean;
  triggers_40_ami: boolean;
  is_uap_eligible: boolean;
  available: boolean;
  notes: string[];
}

export interface AcrisDocument {
  document_id: string;
  doc_type: string;
  doc_date?: string | null;
  recorded_filed?: string | null;
  doc_amount?: number | null;
  party1: string;
  party2: string;
}

export interface AcrisSummary {
  documents: AcrisDocument[];
  last_deed_date?: string | null;
  last_deed_amount?: number | null;
  last_deed_buyer: string;
  last_deed_seller: string;
  total_mortgage_amount?: number | null;
  open_liens: number;
}

export interface HpdViolationSummary {
  open_class_a: number;
  open_class_b: number;
  open_class_c: number;
  total_open: number;
  rent_impairing: number;
  most_recent_date?: string | null;
}

export interface DobJobRecord {
  job_number: string;
  job_type: string;
  job_status: string;
  initial_cost?: number | null;
  proposed_dwelling_units?: number | null;
  existing_dwelling_units?: number | null;
  proposed_zoning_sqft?: number | null;
}

export interface DobJobSummary {
  active_jobs: DobJobRecord[];
  has_active_new_building: boolean;
  has_active_alteration: boolean;
  total_active: number;
}

export interface EcbViolationSummary {
  open_violations: number;
  total_penalties: number;
  total_balance_due: number;
  most_recent_date?: string | null;
}

export interface DofSaleRecord {
  sale_price?: number | null;
  sale_date?: string | null;
  building_class: string;
  residential_units: number;
  commercial_units: number;
  total_units: number;
  gross_square_feet?: number | null;
}

export interface ComparableSalesSummary {
  subject_sale?: DofSaleRecord | null;
  comparable_sales: DofSaleRecord[];
  total_found: number;
}

export interface HpdLitigationSummary {
  open_cases: number;
  case_types: string[];
  most_recent_date?: string | null;
}

export interface FdnyVacateSummary {
  total_vacate_orders: number;
  active_vacate_orders: number;
  vacated_units: number;
}

export interface PropertyLotRecord {
  bbl: string;
  borough: string;
  block: string;
  lot: string;
  address: string;
  zoning: string;
  overlay1?: string | null;
  overlay2?: string | null;
  lot_area: number;
  building_area: number;
  res_far: number;
  units_total: number;
  year_built?: number | null;
  assessed_value?: number | null;
  market_value?: number | null;
  dof_taxable?: number | null;
  has_pluto: boolean;
  has_dof: boolean;
  has_acris: boolean;
  has_hpd: boolean;
  has_dob: boolean;
  has_ecb: boolean;
  has_sales: boolean;
  has_litigation: boolean;
  has_fdny: boolean;
  lot_type_code?: number | null;
  lot_type: string;
}

export interface PropertyContext {
  primary_bbl: string;
  adjacent_bbls: string[];
  selected_bbls: string[];
  address: string;
  borough: string;
  block: string;
  lots: string[];
  zoning_district: string;
  overlay: string;
  overlay_far?: number | null;
  community_facility_far?: number | null;
  standard_far?: number | null;
  qah_far?: number | null;
  standard_height_limit?: number | null;
  qah_height_limit?: number | null;
  lot_coverage_corner?: number | null;
  lot_coverage_interior?: number | null;
  street_type_assumption: string;
  has_narrow_wide: boolean;
  lot_type: string;
  lot_area: number;
  building_area: number;
  units_total: number;
  assessed_value?: number | null;
  market_value?: number | null;
  dof_taxable?: number | null;
  scenarios: PropertyScenario[];
  lots_detail: PropertyLotRecord[];
  acris_summary?: AcrisSummary | null;
  hpd_violations?: HpdViolationSummary | null;
  dob_jobs?: DobJobSummary | null;
  ecb_violations?: EcbViolationSummary | null;
  comparable_sales?: ComparableSalesSummary | null;
  hpd_litigations?: HpdLitigationSummary | null;
  fdny_vacates?: FdnyVacateSummary | null;
  sources: Record<string, unknown>;
  property_brief: string;
}

export async function listIndexes(): Promise<IndexListResponse> {
  const res = await apiFetch(`${API_BASE}/api/indexes`);
  if (!res.ok) throw new Error(`Failed to list indexes (${res.status})`);
  return res.json();
}

export async function getActiveIndex(): Promise<ActiveIndexResponse> {
  const res = await apiFetch(`${API_BASE}/api/indexes/active`);
  if (!res.ok) throw new Error(`Failed to get active index (${res.status})`);
  return res.json();
}

export async function createIndex(
  name: string,
  dimension: number = 3072,
  metric: string = "cosine",
  cloud: string = "aws",
  region: string = "us-east-1",
): Promise<{ created: string }> {
  const res = await apiFetch(`${API_BASE}/api/indexes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, dimension, metric, cloud, region }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Create index failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function deleteIndex(name: string): Promise<{ deleted: string }> {
  const res = await apiFetch(`${API_BASE}/api/indexes/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Delete index failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function switchIndex(name: string): Promise<{ active: string }> {
  const res = await apiFetch(`${API_BASE}/api/indexes/switch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Switch index failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function searchPropertyAddress(query: string): Promise<PropertySearchResult[]> {
  const res = await apiFetch(`${API_BASE}/api/property/search-address?q=${encodeURIComponent(query)}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Property search failed (${res.status}): ${detail}`);
  }
  const payload = await res.json();
  return Array.isArray(payload.results) ? payload.results : [];
}

export async function validatePropertyLot(bbl: string): Promise<ValidatedLotInfo> {
  const res = await apiFetch(`${API_BASE}/api/property/validate-lot?bbl=${encodeURIComponent(bbl)}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Lot validation failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getBlockLots(borough: number, block: number): Promise<BlockLotInfo[]> {
  const qp = new URLSearchParams({ borough: String(borough), block: String(block) });
  const res = await apiFetch(`${API_BASE}/api/property/block-lots?${qp.toString()}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Block lot lookup failed (${res.status}): ${detail}`);
  }
  const payload = await res.json();
  return Array.isArray(payload.lots) ? payload.lots : [];
}

export async function setPropertyContext(primaryBbl: string, adjacentBbls: string[] = []): Promise<PropertyContext> {
  const res = await apiFetch(`${API_BASE}/api/property/context`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ primary_bbl: primaryBbl, adjacent_bbls: adjacentBbls }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Set property context failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getPropertyContext(): Promise<PropertyContext | null> {
  const res = await apiFetch(`${API_BASE}/api/property/context`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Get property context failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function clearPropertyContext(): Promise<{ cleared: boolean }> {
  const res = await apiFetch(`${API_BASE}/api/property/context`, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Clear property context failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── Blueprint Management (ContextLibrary) ──────────────────────────────

export interface Blueprint {
  id: string;
  subject: string;
  instructions: string;
}

export async function listBlueprints(): Promise<{ blueprints: Blueprint[] }> {
  const res = await apiFetch(`${API_BASE}/api/blueprints`);
  if (!res.ok) throw new Error(`Failed to list blueprints (${res.status})`);
  return res.json();
}

export async function createBlueprint(
  subject: string,
  instructions: string,
): Promise<{ id: string; subject: string }> {
  const res = await apiFetch(`${API_BASE}/api/blueprints`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject, instructions }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Create blueprint failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function generateBlueprint(
  subject: string,
): Promise<{ id: string; subject: string; instructions: string }> {
  const res = await apiFetch(`${API_BASE}/api/blueprints/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ subject }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Generate blueprint failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function deleteBlueprint(id: string): Promise<{ deleted: string }> {
  const res = await apiFetch(`${API_BASE}/api/blueprints/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Delete blueprint failed (${res.status}): ${detail}`);
  }
  return res.json();
}

// ── Underwriting Template ──────────────────────────────────────────────

export interface TemplateCell {
  v: string | number | boolean | null;
  r: number;
  c: number;
  f?: boolean;
  z?: string | null;
}

export interface TemplateSheet {
  name: string;
  data: (TemplateCell | null)[][];
  maxRow: number;
  maxCol: number;
}

export interface ParsedTemplate {
  filename: string;
  sheets: TemplateSheet[];
}

export interface ExtractionResult {
  updates: Record<string, Record<string, string | number>>;
  sources?: Record<string, Record<string, string>>;
  confidence?: Record<string, Record<string, string>>;
  message?: string;
}

export interface UnderwritingRecalculationWarning {
  sheet: string;
  message: string;
  refs?: string[];
}

export interface UnderwritingRecalculationResult {
  formulaValues: Record<
    string,
    Record<string, string | number | boolean | null>
  >;
  warnings?: UnderwritingRecalculationWarning[];
}

export async function parseUnderwritingTemplate(file: File): Promise<ParsedTemplate> {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch(
    `${API_BASE}/api/underwriting/parse-template`,
    { method: "POST", body: form },
    300_000,
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Template parse failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function extractUnderwritingValues(): Promise<ExtractionResult> {
  const res = await apiFetch(
    `${API_BASE}/api/underwriting/extract`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
    600_000,
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Extraction failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function recalculateUnderwritingFormulaValues(
  updates: Record<string, Record<string, string | number>>,
): Promise<UnderwritingRecalculationResult> {
  const res = await apiFetch(
    `${API_BASE}/api/underwriting/recalculate`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    },
    300_000,
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Formula recalculation failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function downloadFilledTemplate(
  updates: Record<string, Record<string, string | number>>,
): Promise<Blob> {
  const res = await apiFetch(
    `${API_BASE}/api/underwriting/download`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ updates }),
    },
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Download failed (${res.status}): ${detail}`);
  }
  return res.blob();
}

// ── TC201 – Income & Expense Schedule ──────────────────────────────────

export interface ResidentialOccupancy {
  occupancy_type: string;
  number_of_units: number | null;
  monthly_rent: number | null;
}

export interface NonresidentialFloor {
  floor: string;
  applicant_related_sqft: number | null;
  rented_sqft: number | null;
  vacant_sqft: number | null;
  gross_sqft: number | null;
}

export interface MiscExpenseItem {
  item: string;
  amount: number | null;
}

export interface TC201Data {
  filename: string;
  assessment_year: string;
  // Part 1
  borough: string;
  block: string;
  lot: string;
  bbl: string;
  tax_commission_group_no: string;
  is_condo: string;
  covers_multiple_lots: string;
  total_lots: number | null;
  related_lots: string;
  covers_entire_lot: string;
  // Part 2
  reporting_period_from: string;
  reporting_period_to: string;
  accounting_basis: string;
  accounting_basis_changed: string;
  // Part 3
  residential_occupancy: ResidentialOccupancy[];
  rent_includes_recurring_charges: string;
  // Part 4
  nonresidential_floors: NonresidentialFloor[];
  // Part 5
  entire_lot_leased: string;
  lease_type: string;
  applicant_receives_rental_income: string;
  lessor: string;
  lessee: string;
  lease_from: string;
  lease_to: string;
  annual_rent: number | null;
  additional_sums: number | null;
  lessor_pays_expenses: string;
  lessor_pays_details: string;
  land_only_lease: string;
  // Part 6 - Income
  income_residential_regulated_prior: number | null;
  income_residential_regulated_current: number | null;
  income_residential_unregulated_prior: number | null;
  income_residential_unregulated_current: number | null;
  income_residential_subtotal_prior: number | null;
  income_residential_subtotal_current: number | null;
  income_office_prior: number | null;
  income_office_current: number | null;
  income_retail_prior: number | null;
  income_retail_current: number | null;
  income_loft_prior: number | null;
  income_loft_current: number | null;
  income_factory_prior: number | null;
  income_factory_current: number | null;
  income_warehouse_prior: number | null;
  income_warehouse_current: number | null;
  income_storage_prior: number | null;
  income_storage_current: number | null;
  income_parking_prior: number | null;
  income_parking_current: number | null;
  income_subtotal_prior: number | null;
  income_subtotal_current: number | null;
  income_owner_occupied_prior: number | null;
  income_owner_occupied_current: number | null;
  income_operating_escalation_prior: number | null;
  income_operating_escalation_current: number | null;
  income_re_tax_escalation_prior: number | null;
  income_re_tax_escalation_current: number | null;
  income_utility_services_prior: number | null;
  income_utility_services_current: number | null;
  income_other_services_prior: number | null;
  income_other_services_current: number | null;
  income_govt_subsidies_prior: number | null;
  income_govt_subsidies_current: number | null;
  income_signage_prior: number | null;
  income_signage_current: number | null;
  income_cell_towers_prior: number | null;
  income_cell_towers_current: number | null;
  income_other_prior: number | null;
  income_other_current: number | null;
  income_other_description: string;
  income_total_gross_prior: number | null;
  income_total_gross_current: number | null;
  // Part 7 - Expenses
  expense_fuel_prior: number | null;
  expense_fuel_current: number | null;
  expense_light_power_prior: number | null;
  expense_light_power_current: number | null;
  expense_cleaning_prior: number | null;
  expense_cleaning_current: number | null;
  expense_wages_prior: number | null;
  expense_wages_current: number | null;
  expense_repairs_prior: number | null;
  expense_repairs_current: number | null;
  expense_management_prior: number | null;
  expense_management_current: number | null;
  expense_insurance_prior: number | null;
  expense_insurance_current: number | null;
  expense_water_sewer_prior: number | null;
  expense_water_sewer_current: number | null;
  expense_advertising_prior: number | null;
  expense_advertising_current: number | null;
  expense_painting_prior: number | null;
  expense_painting_current: number | null;
  expense_leasing_ti_prior: number | null;
  expense_leasing_ti_current: number | null;
  expense_misc_prior: number | null;
  expense_misc_current: number | null;
  expense_before_taxes_prior: number | null;
  expense_before_taxes_current: number | null;
  expense_real_estate_taxes_prior: number | null;
  expense_real_estate_taxes_current: number | null;
  expense_total_prior: number | null;
  expense_total_current: number | null;
  // Part 8 - Net
  net_before_re_taxes_prior: number | null;
  net_before_re_taxes_current: number | null;
  net_after_re_taxes_prior: number | null;
  net_after_re_taxes_current: number | null;
  // Part 9
  misc_expenses: MiscExpenseItem[];
  // Part 10
  tenants_electricity_from_applicant: string;
  tenants_electricity_separate_charge: string;
  notes: string;
}

export async function getTC201(): Promise<TC201Data | null> {
  const res = await apiFetch(`${API_BASE}/api/tc201`);
  if (!res.ok) throw new Error(`Failed to get TC201 (${res.status})`);
  return res.json();
}

export async function updateTC201(data: Partial<TC201Data>): Promise<TC201Data> {
  const res = await apiFetch(`${API_BASE}/api/tc201`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`TC201 update failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function clearTC201(): Promise<{ cleared: boolean }> {
  const res = await apiFetch(`${API_BASE}/api/tc201`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to clear TC201 (${res.status})`);
  return res.json();
}

export async function fillTC201FromProperty(): Promise<TC201Data> {
  const res = await apiFetch(
    `${API_BASE}/api/tc201/fill-from-property`,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" },
    300_000,
  );
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`TC201 fill failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function downloadTC201PDF(): Promise<Blob> {
  const res = await apiFetch(`${API_BASE}/api/tc201/download`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`TC201 PDF download failed (${res.status}): ${detail}`);
  }
  return res.blob();
}
