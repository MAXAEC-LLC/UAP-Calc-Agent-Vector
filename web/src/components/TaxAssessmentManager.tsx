"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  clearTC201,
  downloadTC201PDF,
  fillTC201FromProperty,
  getTC201,
  updateTC201,
  type TC201Data,
  type ResidentialOccupancy,
  type NonresidentialFloor,
  type MiscExpenseItem,
} from "@/lib/api";

/* ── Formatting helpers ─────────────────────────────────────────────── */

function fmtDollar(val: number | null | undefined): string {
  if (val == null) return "";
  return "$" + val.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

/* ── Inline editable field ──────────────────────────────────────────── */

function EditField({
  label,
  value,
  fieldKey,
  type = "text",
  width,
  onSave,
}: {
  label: string;
  value: string | number | null | undefined;
  fieldKey: string;
  type?: "text" | "number";
  width?: number;
  onSave: (key: string, val: string | number | null) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.focus();
  }, [editing]);

  function startEdit() {
    setDraft(value != null ? String(value) : "");
    setEditing(true);
  }

  function commit() {
    setEditing(false);
    const trimmed = draft.trim();
    if (trimmed === "") {
      onSave(fieldKey, null);
    } else if (type === "number") {
      const n = Number(trimmed.replace(/[,$]/g, ""));
      onSave(fieldKey, Number.isNaN(n) ? trimmed : n);
    } else {
      onSave(fieldKey, trimmed);
    }
  }

  const displayValue = value != null && value !== "" ? String(value) : "—";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, minHeight: 26 }}>
      <span style={{ fontSize: 11, color: "var(--brand-granite-gray)", minWidth: width ?? 170, flexShrink: 0 }}>
        {label}
      </span>
      {editing ? (
        <input
          ref={inputRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => { if (e.key === "Enter") commit(); if (e.key === "Escape") setEditing(false); }}
          style={{
            flex: 1, padding: "2px 6px", fontSize: 12,
            background: "var(--bg-elevated)", border: "1px solid var(--blue-accent)",
            color: "var(--foreground)", outline: "none",
          }}
        />
      ) : (
        <span
          onClick={startEdit}
          title="Click to edit"
          style={{
            flex: 1, fontSize: 12, color: "var(--foreground)",
            cursor: "pointer", padding: "2px 6px",
            borderBottom: "1px dashed var(--border-color)",
          }}
        >
          {displayValue}
        </span>
      )}
    </div>
  );
}

/* ── Section wrapper ────────────────────────────────────────────────── */

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <h3 style={{
        margin: "0 0 6px", fontSize: 12, fontWeight: 700,
        color: "var(--blue-accent)", textTransform: "uppercase",
        letterSpacing: "0.05em", borderBottom: "1px solid var(--border-color)", paddingBottom: 3,
      }}>
        {title}
      </h3>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>{children}</div>
    </div>
  );
}

/* ── Two-column income/expense row ──────────────────────────────────── */

function DualRow({
  label,
  priorKey,
  currentKey,
  priorVal,
  currentVal,
  onSave,
}: {
  label: string;
  priorKey: string;
  currentKey: string;
  priorVal: number | null | undefined;
  currentVal: number | null | undefined;
  onSave: (key: string, val: string | number | null) => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, minHeight: 24 }}>
      <span style={{ fontSize: 11, color: "var(--brand-granite-gray)", minWidth: 220, flexShrink: 0 }}>{label}</span>
      <div style={{ width: 110, flexShrink: 0 }}>
        <EditField label="" value={priorVal} fieldKey={priorKey} type="number" width={0} onSave={onSave} />
      </div>
      <div style={{ width: 110, flexShrink: 0 }}>
        <EditField label="" value={currentVal} fieldKey={currentKey} type="number" width={0} onSave={onSave} />
      </div>
    </div>
  );
}

function DualHeader() {
  return (
    <div style={{ display: "flex", gap: 4, fontSize: 10, color: "var(--brand-granite-gray)", fontWeight: 600, marginBottom: 2 }}>
      <span style={{ minWidth: 220 }} />
      <span style={{ width: 110, textAlign: "center" }}>Prior Year</span>
      <span style={{ width: 110, textAlign: "center" }}>Current Year</span>
    </div>
  );
}

/* ── Blank template ─────────────────────────────────────────────────── */

const BLANK_TC201: TC201Data = {
  filename: "", assessment_year: "2026/27",
  borough: "", block: "", lot: "", bbl: "",
  tax_commission_group_no: "", is_condo: "", covers_multiple_lots: "",
  total_lots: null, related_lots: "", covers_entire_lot: "",
  reporting_period_from: "", reporting_period_to: "",
  accounting_basis: "", accounting_basis_changed: "",
  residential_occupancy: [], rent_includes_recurring_charges: "",
  nonresidential_floors: [],
  entire_lot_leased: "", lease_type: "", applicant_receives_rental_income: "",
  lessor: "", lessee: "", lease_from: "", lease_to: "",
  annual_rent: null, additional_sums: null,
  lessor_pays_expenses: "", lessor_pays_details: "", land_only_lease: "",
  income_residential_regulated_prior: null, income_residential_regulated_current: null,
  income_residential_unregulated_prior: null, income_residential_unregulated_current: null,
  income_residential_subtotal_prior: null, income_residential_subtotal_current: null,
  income_office_prior: null, income_office_current: null,
  income_retail_prior: null, income_retail_current: null,
  income_loft_prior: null, income_loft_current: null,
  income_factory_prior: null, income_factory_current: null,
  income_warehouse_prior: null, income_warehouse_current: null,
  income_storage_prior: null, income_storage_current: null,
  income_parking_prior: null, income_parking_current: null,
  income_subtotal_prior: null, income_subtotal_current: null,
  income_owner_occupied_prior: null, income_owner_occupied_current: null,
  income_operating_escalation_prior: null, income_operating_escalation_current: null,
  income_re_tax_escalation_prior: null, income_re_tax_escalation_current: null,
  income_utility_services_prior: null, income_utility_services_current: null,
  income_other_services_prior: null, income_other_services_current: null,
  income_govt_subsidies_prior: null, income_govt_subsidies_current: null,
  income_signage_prior: null, income_signage_current: null,
  income_cell_towers_prior: null, income_cell_towers_current: null,
  income_other_prior: null, income_other_current: null, income_other_description: "",
  income_total_gross_prior: null, income_total_gross_current: null,
  expense_fuel_prior: null, expense_fuel_current: null,
  expense_light_power_prior: null, expense_light_power_current: null,
  expense_cleaning_prior: null, expense_cleaning_current: null,
  expense_wages_prior: null, expense_wages_current: null,
  expense_repairs_prior: null, expense_repairs_current: null,
  expense_management_prior: null, expense_management_current: null,
  expense_insurance_prior: null, expense_insurance_current: null,
  expense_water_sewer_prior: null, expense_water_sewer_current: null,
  expense_advertising_prior: null, expense_advertising_current: null,
  expense_painting_prior: null, expense_painting_current: null,
  expense_leasing_ti_prior: null, expense_leasing_ti_current: null,
  expense_misc_prior: null, expense_misc_current: null,
  expense_before_taxes_prior: null, expense_before_taxes_current: null,
  expense_real_estate_taxes_prior: null, expense_real_estate_taxes_current: null,
  expense_total_prior: null, expense_total_current: null,
  net_before_re_taxes_prior: null, net_before_re_taxes_current: null,
  net_after_re_taxes_prior: null, net_after_re_taxes_current: null,
  misc_expenses: [],
  tenants_electricity_from_applicant: "", tenants_electricity_separate_charge: "",
  notes: "",
};

/* ── Main component ─────────────────────────────────────────────────── */

export default function TaxAssessmentManager() {
  const [data, setData] = useState<TC201Data>(BLANK_TC201);
  const [loading, setLoading] = useState(false);
  const [filling, setFilling] = useState(false);
  const [error, setError] = useState("");
  const [saveStatus, setSaveStatus] = useState("");

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const tc = await getTC201();
      setData(tc ?? BLANK_TC201);
      setError("");
    } catch {
      setData(BLANK_TC201);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  async function handleClear() {
    try { await clearTC201(); setData(BLANK_TC201); setError(""); }
    catch (err) { setError(err instanceof Error ? err.message : "Clear failed"); }
  }

  async function handleFill() {
    setFilling(true); setError("");
    try { const result = await fillTC201FromProperty(); setData(result); }
    catch (err) { setError(err instanceof Error ? err.message : "Auto-fill failed"); }
    finally { setFilling(false); }
  }

  async function handleDownloadPDF() {
    setError("");
    try {
      const blob = await downloadTC201PDF();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `TC201_${data.bbl || "blank"}_${data.assessment_year.replace("/", "-")}.pdf`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) { setError(err instanceof Error ? err.message : "PDF download failed"); }
  }

  async function handleFieldSave(key: string, val: string | number | null) {
    setData((prev) => ({ ...prev, [key]: val }));
    setSaveStatus("Saving...");
    try {
      const updated = await updateTC201({ [key]: val });
      setData(updated);
      setSaveStatus("Saved");
      setTimeout(() => setSaveStatus(""), 1500);
    } catch {
      setSaveStatus("Save failed");
      setTimeout(() => setSaveStatus(""), 2000);
    }
  }

  if (loading) {
    return (
      <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
        <span style={{ color: "var(--brand-granite-gray)", fontSize: 14 }}>Loading TC201...</span>
      </div>
    );
  }

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
      {/* Header bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "var(--foreground)" }}>
          Form TC201 — Income &amp; Expense Schedule {data.assessment_year}
        </h2>
        {data.bbl && (
          <span style={{ fontSize: 11, color: "var(--blue-accent)", padding: "2px 8px", border: "1px solid rgba(59,130,246,0.25)", background: "rgba(59,130,246,0.08)" }}>
            {data.borough} / {data.block} / {data.lot}
          </span>
        )}
        {saveStatus && (
          <span style={{ fontSize: 11, color: saveStatus === "Saved" ? "#22c55e" : saveStatus === "Save failed" ? "#e55" : "var(--blue-accent)" }}>
            {saveStatus}
          </span>
        )}
        <div style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <button onClick={handleFill} disabled={filling} style={{
            padding: "4px 10px", background: filling ? "var(--bg-elevated)" : "var(--blue)",
            border: "1px solid var(--border-color)", color: "var(--foreground)",
            cursor: filling ? "not-allowed" : "pointer", fontSize: 11, fontWeight: 600,
          }}>
            {filling ? "Populating..." : "⬇ Fill from Property & Docs"}
          </button>
          <button onClick={handleDownloadPDF} style={{
            padding: "4px 10px", background: "transparent", border: "1px solid var(--border-color)",
            color: "var(--foreground)", cursor: "pointer", fontSize: 11,
          }}>↓ PDF</button>
          <button onClick={handleClear} style={{
            padding: "4px 10px", background: "transparent", border: "1px solid rgba(238,85,85,0.3)",
            color: "#e55", cursor: "pointer", fontSize: 11,
          }}>Reset</button>
        </div>
      </div>

      {error && (
        <p style={{ fontSize: 11, color: "#e55", padding: "4px 10px", marginBottom: 12, background: "rgba(238,85,85,0.1)", border: "1px solid rgba(238,85,85,0.2)" }}>
          {error}
        </p>
      )}

      {filling && (
        <div style={{ padding: "8px 12px", marginBottom: 12, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)", fontSize: 12, color: "var(--blue-accent)" }}>
          Extracting data from property context and uploaded documents...
        </div>
      )}

      {/* Part 1: Property Identification */}
      <Section title="Part 1 — Property Identification">
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="Borough" value={data.borough} fieldKey="borough" onSave={handleFieldSave} width={60} />
          <EditField label="Block" value={data.block} fieldKey="block" onSave={handleFieldSave} width={40} />
          <EditField label="Lot" value={data.lot} fieldKey="lot" onSave={handleFieldSave} width={30} />
          <EditField label="Group #" value={data.tax_commission_group_no} fieldKey="tax_commission_group_no" onSave={handleFieldSave} width={50} />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="Condo?" value={data.is_condo} fieldKey="is_condo" onSave={handleFieldSave} width={60} />
          <EditField label="Multiple lots?" value={data.covers_multiple_lots} fieldKey="covers_multiple_lots" onSave={handleFieldSave} width={90} />
          <EditField label="Total lots" value={data.total_lots} fieldKey="total_lots" type="number" onSave={handleFieldSave} width={70} />
        </div>
        <EditField label="Covers entire lot?" value={data.covers_entire_lot} fieldKey="covers_entire_lot" onSave={handleFieldSave} />
      </Section>

      {/* Part 2: Reporting Period */}
      <Section title="Part 2 — Reporting Period & Accounting Basis">
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="From" value={data.reporting_period_from} fieldKey="reporting_period_from" onSave={handleFieldSave} width={40} />
          <EditField label="To" value={data.reporting_period_to} fieldKey="reporting_period_to" onSave={handleFieldSave} width={30} />
          <EditField label="Basis" value={data.accounting_basis} fieldKey="accounting_basis" onSave={handleFieldSave} width={40} />
          <EditField label="Changed?" value={data.accounting_basis_changed} fieldKey="accounting_basis_changed" onSave={handleFieldSave} width={60} />
        </div>
      </Section>

      {/* Part 3: Residential Occupancy */}
      <Section title="Part 3 — Residential Occupancy (as of Jan 5, 2026)">
        {data.residential_occupancy.length > 0 ? (
          <>
            <div style={{ display: "flex", gap: 4, fontSize: 10, fontWeight: 600, color: "var(--brand-granite-gray)", marginBottom: 2 }}>
              <span style={{ minWidth: 170 }}>Type</span>
              <span style={{ width: 80, textAlign: "center" }}>Units</span>
              <span style={{ width: 100, textAlign: "center" }}>Monthly Rent</span>
            </div>
            {data.residential_occupancy.map((r, i) => (
              <div key={i} style={{ display: "flex", gap: 4, fontSize: 12, color: "var(--foreground)", minHeight: 22 }}>
                <span style={{ minWidth: 170, fontSize: 11 }}>{r.occupancy_type || "—"}</span>
                <span style={{ width: 80, textAlign: "center" }}>{r.number_of_units ?? "—"}</span>
                <span style={{ width: 100, textAlign: "center" }}>{r.monthly_rent != null ? fmtDollar(r.monthly_rent) : "—"}</span>
              </div>
            ))}
          </>
        ) : (
          <span style={{ fontSize: 11, color: "var(--brand-granite-gray)" }}>No residential occupancy data</span>
        )}
        <EditField label="Rent includes recurring charges?" value={data.rent_includes_recurring_charges} fieldKey="rent_includes_recurring_charges" onSave={handleFieldSave} />
      </Section>

      {/* Part 4: Nonresidential Occupancy */}
      <Section title="Part 4 — Nonresidential Occupancy (as of Jan 5, 2026)">
        {data.nonresidential_floors.length > 0 ? (
          <>
            <div style={{ display: "flex", gap: 4, fontSize: 10, fontWeight: 600, color: "var(--brand-granite-gray)", marginBottom: 2 }}>
              <span style={{ minWidth: 100 }}>Floor</span>
              <span style={{ width: 90, textAlign: "center" }}>Applicant SqFt</span>
              <span style={{ width: 90, textAlign: "center" }}>Rented SqFt</span>
              <span style={{ width: 90, textAlign: "center" }}>Vacant SqFt</span>
              <span style={{ width: 90, textAlign: "center" }}>Gross SqFt</span>
            </div>
            {data.nonresidential_floors.map((f, i) => (
              <div key={i} style={{ display: "flex", gap: 4, fontSize: 12, color: "var(--foreground)", minHeight: 22 }}>
                <span style={{ minWidth: 100, fontSize: 11 }}>{f.floor || "—"}</span>
                <span style={{ width: 90, textAlign: "center" }}>{f.applicant_related_sqft ?? "—"}</span>
                <span style={{ width: 90, textAlign: "center" }}>{f.rented_sqft ?? "—"}</span>
                <span style={{ width: 90, textAlign: "center" }}>{f.vacant_sqft ?? "—"}</span>
                <span style={{ width: 90, textAlign: "center" }}>{f.gross_sqft ?? "—"}</span>
              </div>
            ))}
          </>
        ) : (
          <span style={{ fontSize: 11, color: "var(--brand-granite-gray)" }}>No nonresidential occupancy data</span>
        )}
      </Section>

      {/* Part 5: Lease Information */}
      <Section title="Part 5 — Lease Information (as of Jan 5, 2026)">
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="Entire lot leased?" value={data.entire_lot_leased} fieldKey="entire_lot_leased" onSave={handleFieldSave} width={120} />
          <EditField label="Lease type" value={data.lease_type} fieldKey="lease_type" onSave={handleFieldSave} width={80} />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="Lessor (Landlord)" value={data.lessor} fieldKey="lessor" onSave={handleFieldSave} />
          <EditField label="Lessee (Tenant)" value={data.lessee} fieldKey="lessee" onSave={handleFieldSave} />
        </div>
        <div style={{ display: "flex", gap: 12 }}>
          <EditField label="Lease from" value={data.lease_from} fieldKey="lease_from" onSave={handleFieldSave} width={80} />
          <EditField label="Lease to" value={data.lease_to} fieldKey="lease_to" onSave={handleFieldSave} width={80} />
          <EditField label="Annual rent" value={data.annual_rent} fieldKey="annual_rent" type="number" onSave={handleFieldSave} width={80} />
        </div>
      </Section>

      {/* Part 6: Income */}
      <Section title="Part 6 — Income Information">
        <DualHeader />
        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--blue-accent)", margin: "4px 0 2px" }}>Residential</div>
        <DualRow label="a. Regulated" priorKey="income_residential_regulated_prior" currentKey="income_residential_regulated_current" priorVal={data.income_residential_regulated_prior} currentVal={data.income_residential_regulated_current} onSave={handleFieldSave} />
        <DualRow label="   Unregulated" priorKey="income_residential_unregulated_prior" currentKey="income_residential_unregulated_current" priorVal={data.income_residential_unregulated_prior} currentVal={data.income_residential_unregulated_current} onSave={handleFieldSave} />
        <DualRow label="   Subtotal residential" priorKey="income_residential_subtotal_prior" currentKey="income_residential_subtotal_current" priorVal={data.income_residential_subtotal_prior} currentVal={data.income_residential_subtotal_current} onSave={handleFieldSave} />

        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--blue-accent)", margin: "4px 0 2px" }}>Commercial</div>
        <DualRow label="b. Office" priorKey="income_office_prior" currentKey="income_office_current" priorVal={data.income_office_prior} currentVal={data.income_office_current} onSave={handleFieldSave} />
        <DualRow label="c. Retail" priorKey="income_retail_prior" currentKey="income_retail_current" priorVal={data.income_retail_prior} currentVal={data.income_retail_current} onSave={handleFieldSave} />
        <DualRow label="d. Loft" priorKey="income_loft_prior" currentKey="income_loft_current" priorVal={data.income_loft_prior} currentVal={data.income_loft_current} onSave={handleFieldSave} />
        <DualRow label="e. Factory" priorKey="income_factory_prior" currentKey="income_factory_current" priorVal={data.income_factory_prior} currentVal={data.income_factory_current} onSave={handleFieldSave} />
        <DualRow label="f. Warehouse" priorKey="income_warehouse_prior" currentKey="income_warehouse_current" priorVal={data.income_warehouse_prior} currentVal={data.income_warehouse_current} onSave={handleFieldSave} />
        <DualRow label="g. Storage" priorKey="income_storage_prior" currentKey="income_storage_current" priorVal={data.income_storage_prior} currentVal={data.income_storage_current} onSave={handleFieldSave} />
        <DualRow label="h. Garage/Parking" priorKey="income_parking_prior" currentKey="income_parking_current" priorVal={data.income_parking_prior} currentVal={data.income_parking_current} onSave={handleFieldSave} />
        <DualRow label="SUBTOTAL" priorKey="income_subtotal_prior" currentKey="income_subtotal_current" priorVal={data.income_subtotal_prior} currentVal={data.income_subtotal_current} onSave={handleFieldSave} />

        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--blue-accent)", margin: "4px 0 2px" }}>Other</div>
        <DualRow label="i. Owner-occupied/related" priorKey="income_owner_occupied_prior" currentKey="income_owner_occupied_current" priorVal={data.income_owner_occupied_prior} currentVal={data.income_owner_occupied_current} onSave={handleFieldSave} />

        <div style={{ fontSize: 10, fontWeight: 700, color: "var(--blue-accent)", margin: "4px 0 2px" }}>j. Ancillary Income</div>
        <DualRow label="   1. Operating escalation" priorKey="income_operating_escalation_prior" currentKey="income_operating_escalation_current" priorVal={data.income_operating_escalation_prior} currentVal={data.income_operating_escalation_current} onSave={handleFieldSave} />
        <DualRow label="   2. RE tax escalation" priorKey="income_re_tax_escalation_prior" currentKey="income_re_tax_escalation_current" priorVal={data.income_re_tax_escalation_prior} currentVal={data.income_re_tax_escalation_current} onSave={handleFieldSave} />
        <DualRow label="   3. Sale of utility services" priorKey="income_utility_services_prior" currentKey="income_utility_services_current" priorVal={data.income_utility_services_prior} currentVal={data.income_utility_services_current} onSave={handleFieldSave} />
        <DualRow label="   4. Sale of other services" priorKey="income_other_services_prior" currentKey="income_other_services_current" priorVal={data.income_other_services_prior} currentVal={data.income_other_services_current} onSave={handleFieldSave} />
        <DualRow label="   5. Government rent subsidies" priorKey="income_govt_subsidies_prior" currentKey="income_govt_subsidies_current" priorVal={data.income_govt_subsidies_prior} currentVal={data.income_govt_subsidies_current} onSave={handleFieldSave} />
        <DualRow label="   6. Signage/billboard" priorKey="income_signage_prior" currentKey="income_signage_current" priorVal={data.income_signage_prior} currentVal={data.income_signage_current} onSave={handleFieldSave} />
        <DualRow label="   7. Cell towers/telecom" priorKey="income_cell_towers_prior" currentKey="income_cell_towers_current" priorVal={data.income_cell_towers_prior} currentVal={data.income_cell_towers_current} onSave={handleFieldSave} />
        <DualRow label="k. Other (specify)" priorKey="income_other_prior" currentKey="income_other_current" priorVal={data.income_other_prior} currentVal={data.income_other_current} onSave={handleFieldSave} />
        <EditField label="   Other description" value={data.income_other_description} fieldKey="income_other_description" onSave={handleFieldSave} width={220} />

        <div style={{ borderTop: "1px solid var(--border-color)", marginTop: 4, paddingTop: 4 }}>
          <DualRow label="l. TOTAL GROSS INCOME" priorKey="income_total_gross_prior" currentKey="income_total_gross_current" priorVal={data.income_total_gross_prior} currentVal={data.income_total_gross_current} onSave={handleFieldSave} />
        </div>
      </Section>

      {/* Part 7: Expenses */}
      <Section title="Part 7 — Expense Information">
        <DualHeader />
        <DualRow label="a. Fuel" priorKey="expense_fuel_prior" currentKey="expense_fuel_current" priorVal={data.expense_fuel_prior} currentVal={data.expense_fuel_current} onSave={handleFieldSave} />
        <DualRow label="b. Light and power" priorKey="expense_light_power_prior" currentKey="expense_light_power_current" priorVal={data.expense_light_power_prior} currentVal={data.expense_light_power_current} onSave={handleFieldSave} />
        <DualRow label="c. Cleaning contracts" priorKey="expense_cleaning_prior" currentKey="expense_cleaning_current" priorVal={data.expense_cleaning_prior} currentVal={data.expense_cleaning_current} onSave={handleFieldSave} />
        <DualRow label="d. Wages and payroll" priorKey="expense_wages_prior" currentKey="expense_wages_current" priorVal={data.expense_wages_prior} currentVal={data.expense_wages_current} onSave={handleFieldSave} />
        <DualRow label="e. Repairs and maintenance" priorKey="expense_repairs_prior" currentKey="expense_repairs_current" priorVal={data.expense_repairs_prior} currentVal={data.expense_repairs_current} onSave={handleFieldSave} />
        <DualRow label="f. Management and admin" priorKey="expense_management_prior" currentKey="expense_management_current" priorVal={data.expense_management_prior} currentVal={data.expense_management_current} onSave={handleFieldSave} />
        <DualRow label="g. Insurance (annual)" priorKey="expense_insurance_prior" currentKey="expense_insurance_current" priorVal={data.expense_insurance_prior} currentVal={data.expense_insurance_current} onSave={handleFieldSave} />
        <DualRow label="h. Water and sewer" priorKey="expense_water_sewer_prior" currentKey="expense_water_sewer_current" priorVal={data.expense_water_sewer_prior} currentVal={data.expense_water_sewer_current} onSave={handleFieldSave} />
        <DualRow label="i. Advertising" priorKey="expense_advertising_prior" currentKey="expense_advertising_current" priorVal={data.expense_advertising_prior} currentVal={data.expense_advertising_current} onSave={handleFieldSave} />
        <DualRow label="j. Interior painting/decorating" priorKey="expense_painting_prior" currentKey="expense_painting_current" priorVal={data.expense_painting_prior} currentVal={data.expense_painting_current} onSave={handleFieldSave} />
        <DualRow label="k. Leasing & tenant improvement" priorKey="expense_leasing_ti_prior" currentKey="expense_leasing_ti_current" priorVal={data.expense_leasing_ti_prior} currentVal={data.expense_leasing_ti_current} onSave={handleFieldSave} />
        <DualRow label="l. Miscellaneous (Part 9)" priorKey="expense_misc_prior" currentKey="expense_misc_current" priorVal={data.expense_misc_prior} currentVal={data.expense_misc_current} onSave={handleFieldSave} />

        <div style={{ borderTop: "1px solid var(--border-color)", marginTop: 4, paddingTop: 4 }}>
          <DualRow label="m. EXPENSES BEFORE RE TAXES" priorKey="expense_before_taxes_prior" currentKey="expense_before_taxes_current" priorVal={data.expense_before_taxes_prior} currentVal={data.expense_before_taxes_current} onSave={handleFieldSave} />
          <DualRow label="n. Real estate taxes" priorKey="expense_real_estate_taxes_prior" currentKey="expense_real_estate_taxes_current" priorVal={data.expense_real_estate_taxes_prior} currentVal={data.expense_real_estate_taxes_current} onSave={handleFieldSave} />
          <DualRow label="o. TOTAL EXPENSES" priorKey="expense_total_prior" currentKey="expense_total_current" priorVal={data.expense_total_prior} currentVal={data.expense_total_current} onSave={handleFieldSave} />
        </div>
      </Section>

      {/* Part 8: Net Profit (or Loss) */}
      <Section title="Part 8 — Net Profit (or Loss)">
        <DualHeader />
        <DualRow label="a. Net before RE taxes" priorKey="net_before_re_taxes_prior" currentKey="net_before_re_taxes_current" priorVal={data.net_before_re_taxes_prior} currentVal={data.net_before_re_taxes_current} onSave={handleFieldSave} />
        <DualRow label="b. Net after RE taxes" priorKey="net_after_re_taxes_prior" currentKey="net_after_re_taxes_current" priorVal={data.net_after_re_taxes_prior} currentVal={data.net_after_re_taxes_current} onSave={handleFieldSave} />
      </Section>

      {/* Part 9: Misc Expenses */}
      <Section title="Part 9 — Itemization of Miscellaneous Expenses">
        {data.misc_expenses.length > 0 ? (
          <>
            <div style={{ display: "flex", gap: 4, fontSize: 10, fontWeight: 600, color: "var(--brand-granite-gray)", marginBottom: 2 }}>
              <span style={{ flex: 1 }}>Item</span>
              <span style={{ width: 100, textAlign: "right" }}>Amount</span>
            </div>
            {data.misc_expenses.map((m, i) => (
              <div key={i} style={{ display: "flex", gap: 4, fontSize: 12, color: "var(--foreground)", minHeight: 22 }}>
                <span style={{ flex: 1 }}>{m.item || "—"}</span>
                <span style={{ width: 100, textAlign: "right" }}>{m.amount != null ? fmtDollar(m.amount) : "—"}</span>
              </div>
            ))}
          </>
        ) : (
          <span style={{ fontSize: 11, color: "var(--brand-granite-gray)" }}>No miscellaneous expenses itemized</span>
        )}
      </Section>

      {/* Part 10: Tenants' Electricity */}
      <Section title="Part 10 — Tenants' Electricity">
        <EditField label="Electricity from applicant?" value={data.tenants_electricity_from_applicant} fieldKey="tenants_electricity_from_applicant" onSave={handleFieldSave} />
        <EditField label="Separate charge?" value={data.tenants_electricity_separate_charge} fieldKey="tenants_electricity_separate_charge" onSave={handleFieldSave} />
      </Section>

      {/* Notes */}
      {data.notes && (
        <Section title="Notes">
          <EditField label="Notes" value={data.notes} fieldKey="notes" onSave={handleFieldSave} />
        </Section>
      )}
    </div>
  );
}
