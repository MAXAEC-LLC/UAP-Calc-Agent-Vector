/**
 * Client-side TC201 PDF form filler using pdf-lib.
 * Fetches the blank template from the backend, fills AcroForm fields,
 * and returns a downloadable Blob.
 */
import { PDFDocument, StandardFonts } from "pdf-lib";
import type { TC201Data } from "./api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

/** Format number as comma-separated integer string (no $). */
function cur(val: number | null | undefined): string {
  if (val == null) return "";
  return Math.round(val).toLocaleString("en-US");
}

/** Stringify a value for a form field. */
function v(val: string | number | null | undefined): string {
  if (val == null) return "";
  return String(val);
}

/**
 * Build the AcroForm field-name → value map from TC201Data.
 * This mirrors the Python _generate_tc201_pdf() logic exactly.
 */
function buildFieldMap(data: TC201Data): Record<string, string> {
  const fields: Record<string, string> = {};

  // ── Part 1: Property Identification ──
  fields["Borough"] = v(data.borough);
  fields["Block"] = v(data.block);
  fields["Lot"] = v(data.lot);
  fields["GROUP"] = v(data.tax_commission_group_no);
  fields["REP_TC_GROUP_NUMBER"] = v(data.tax_commission_group_no);
  fields["is_condo_cover_all_tc109"] = v(data.is_condo);
  fields["is_cover_more_one_lot"] = v(data.covers_multiple_lots);
  if (data.total_lots != null) fields["total_lot_number"] = v(data.total_lots);
  fields["is_schedule_for_intire_lots"] = v(data.covers_entire_lot);
  if (data.related_lots) fields["lots_1"] = v(data.related_lots);

  // ── Part 2: Reporting Period ──
  if (data.reporting_period_from) {
    const parts = data.reporting_period_from.replace(/-/g, "/").split("/");
    if (parts.length >= 3) {
      fields["rp_from_month"] = parts[0];
      fields["rp_from_day"] = parts[1];
      fields["rp_from_year"] = parts[2];
    }
  }
  if (data.reporting_period_to) {
    const parts = data.reporting_period_to.replace(/-/g, "/").split("/");
    if (parts.length >= 3) {
      fields["rp_to_month"] = parts[0];
      fields["rp_to_day"] = parts[1];
      fields["rp_to_year"] = parts[2];
    }
  }
  if (data.accounting_basis?.toLowerCase().startsWith("cash")) {
    fields["cb_Accbasis_cash"] = "/On";
    fields["cb_Accbasis_Accrual"] = "/Off";
  } else if (data.accounting_basis?.toLowerCase().startsWith("accrual")) {
    fields["cb_Accbasis_Accrual"] = "/On";
    fields["cb_Accbasis_cash"] = "/Off";
  }
  if (data.accounting_basis_changed?.toUpperCase() === "Y") {
    fields["cb_Accbasis_changed_yes"] = "/On";
    fields["cb_Accbasis_changed_no"] = "/Off";
  } else if (data.accounting_basis_changed?.toUpperCase() === "N") {
    fields["cb_Accbasis_changed_no"] = "/On";
    fields["cb_Accbasis_changed_yes"] = "/Off";
  }

  // ── Part 3: Residential Occupancy ──
  const occMap: Record<string, [string, string]> = {
    "RENTED, REGULATED": ["regulated_units_number", "regulated_monthly_rent"],
    "RENTED, UNREGULATED": ["unregulated_units_number", "unregulated_monthly_rent"],
    "OWNER OCCUPIED": ["owner_occupied_units_number", "owner_occupied_monthly_rent"],
    "VACANT": ["vacant_units_number", "vacant_monthly_rent"],
  };
  let totalUnits = 0;
  let totalRent = 0;
  for (const r of data.residential_occupancy ?? []) {
    const key = r.occupancy_type.toUpperCase().trim();
    for (const [k, [unitsField, rentField]] of Object.entries(occMap)) {
      if (key.startsWith(k.substring(0, 6))) {
        if (r.number_of_units != null) {
          fields[unitsField] = v(r.number_of_units);
          totalUnits += r.number_of_units;
        }
        if (r.monthly_rent != null) {
          fields[rentField] = cur(r.monthly_rent);
          totalRent += r.monthly_rent;
        }
        break;
      }
    }
  }
  if (totalUnits) fields["total_units_number"] = v(totalUnits);
  if (totalRent) fields["total_monthly_rent"] = cur(totalRent);
  if (data.rent_includes_recurring_charges) {
    fields["is_include_all_charges"] = v(data.rent_includes_recurring_charges);
  }

  // ── Part 4: Nonresidential Occupancy ──
  const floorPrefixMap: Record<string, string> = {
    "FLOOR 3": "fl3", "3": "fl3",
    "2ND": "fl2", "2": "fl2",
    "1ST": "fl1", "1": "fl1",
    "BASEMENT": "base", "B": "base",
  };
  for (const nf of data.nonresidential_floors ?? []) {
    const key = nf.floor.toUpperCase().trim();
    let prefix: string | null = null;
    for (const [k, p] of Object.entries(floorPrefixMap)) {
      if (key.startsWith(k)) { prefix = p; break; }
    }
    if (!prefix) continue;
    if (nf.applicant_related_sqft != null) fields[`${prefix}_applicant`] = cur(nf.applicant_related_sqft);
    if (nf.rented_sqft != null) fields[`${prefix}_rented`] = cur(nf.rented_sqft);
    if (nf.vacant_sqft != null) fields[`${prefix}_vacant`] = cur(nf.vacant_sqft);
    if (nf.gross_sqft != null) fields[`${prefix}_total`] = cur(nf.gross_sqft);
  }

  // ── Part 5: Lease info ──
  fields["is_ground_lease_y"] = v(data.entire_lot_leased);
  if (data.lease_type) {
    const lt = data.lease_type.toLowerCase();
    fields["cb_lease_type_gross"] = lt.includes("gross") ? "/On" : "/Off";
    fields["cb_lease_type_net"] = lt === "net" ? "/On" : "/Off";
    fields["cb_lease_type_ground"] = lt.includes("ground") ? "/On" : "/Off";
  }
  if (data.applicant_receives_rental_income) {
    fields["is_lessee_receives_rental_income"] = v(data.applicant_receives_rental_income);
  }
  if (data.lessor) fields["LESSOR"] = v(data.lessor);
  if (data.lessee) fields["LESSEE"] = v(data.lessee);
  if (data.lease_from) {
    const parts = data.lease_from.replace(/-/g, "/").split("/");
    if (parts.length >= 2) {
      fields["term_of_lease_from_month"] = parts[0];
      fields["term_of_lease_from_year"] = parts[parts.length - 1];
    }
  }
  if (data.lease_to) {
    const parts = data.lease_to.replace(/-/g, "/").split("/");
    if (parts.length >= 2) {
      fields["term_of_lease_to_month"] = parts[0];
      fields["term_of_lease_to_year"] = parts[parts.length - 1];
    }
  }
  if (data.annual_rent != null) fields["annual_rent"] = cur(data.annual_rent);
  if (data.additional_sums != null) fields["additional_sums"] = cur(data.additional_sums);

  // ── Part 6: Income (current year) ──
  fields["Regulated"] = cur(data.income_residential_regulated_current);
  fields["Unregulated"] = cur(data.income_residential_unregulated_current);
  fields["SubtotalResidentialIncome"] = cur(data.income_residential_subtotal_current);
  fields["Office"] = cur(data.income_office_current);
  fields["Retail_Tenants"] = cur(data.income_retail_current);
  fields["Loft"] = cur(data.income_loft_current);
  fields["Factory"] = cur(data.income_factory_current);
  fields["Warehouse"] = cur(data.income_warehouse_current);
  fields["Storage"] = cur(data.income_storage_current);
  fields["Garages"] = cur(data.income_parking_current);
  fields["Subtotal"] = cur(data.income_subtotal_current);
  fields["Owner_Related"] = cur(data.income_owner_occupied_current);
  fields["Operating"] = cur(data.income_operating_escalation_current);
  fields["Tax_Escalation"] = cur(data.income_re_tax_escalation_current);
  fields["Utility_Services"] = cur(data.income_utility_services_current);
  fields["Other_Services"] = cur(data.income_other_services_current);
  fields["Rent_Sub"] = cur(data.income_govt_subsidies_current);
  fields["Signage"] = cur(data.income_signage_current);
  fields["Cell_Towers"] = cur(data.income_cell_towers_current);
  fields["Other"] = cur(data.income_other_current);
  if (data.income_other_description) fields["INCOME_Other_NAME"] = v(data.income_other_description);
  fields["Total_inc_Est"] = cur(data.income_total_gross_current);

  // ── Part 6: Income (prior year) ──
  fields["pRegulated"] = cur(data.income_residential_regulated_prior);
  fields["pUnregulated"] = cur(data.income_residential_unregulated_prior);
  fields["pSubtotalResidentialIncome"] = cur(data.income_residential_subtotal_prior);
  fields["pOffice"] = cur(data.income_office_prior);
  fields["pRetail_Tenants"] = cur(data.income_retail_prior);
  fields["pLoft"] = cur(data.income_loft_prior);
  fields["pFactory"] = cur(data.income_factory_prior);
  fields["pWarehouse"] = cur(data.income_warehouse_prior);
  fields["pStorage"] = cur(data.income_storage_prior);
  fields["pGarages"] = cur(data.income_parking_prior);
  fields["Psubtotal"] = cur(data.income_subtotal_prior);
  fields["pOwner_Related"] = cur(data.income_owner_occupied_prior);
  fields["pOperating"] = cur(data.income_operating_escalation_prior);
  fields["pTax_Escalation"] = cur(data.income_re_tax_escalation_prior);
  fields["pUtility_Services"] = cur(data.income_utility_services_prior);
  fields["pOther_Services"] = cur(data.income_other_services_prior);
  fields["pRent_Sub"] = cur(data.income_govt_subsidies_prior);
  fields["pSignage"] = cur(data.income_signage_prior);
  fields["pCell_Towers"] = cur(data.income_cell_towers_prior);
  fields["pOther"] = cur(data.income_other_prior);
  fields["pTotal_inc_Est"] = cur(data.income_total_gross_prior);

  // ── Part 7: Expenses (current year) ──
  fields["Fuel"] = cur(data.expense_fuel_current);
  fields["Light"] = cur(data.expense_light_power_current);
  fields["Cleaning"] = cur(data.expense_cleaning_current);
  fields["Wages"] = cur(data.expense_wages_current);
  fields["Repairs"] = cur(data.expense_repairs_current);
  fields["Management"] = cur(data.expense_management_current);
  fields["Insurance"] = cur(data.expense_insurance_current);
  fields["Water"] = cur(data.expense_water_sewer_current);
  fields["Advertising"] = cur(data.expense_advertising_current);
  fields["Interior"] = cur(data.expense_painting_current);
  fields["Amortized"] = cur(data.expense_leasing_ti_current);
  fields["Misc"] = cur(data.expense_misc_current);
  fields["SubTot_Expense"] = cur(data.expense_before_taxes_current);
  fields["Real_Taxes"] = cur(data.expense_real_estate_taxes_current);
  fields["Total_Expense"] = cur(data.expense_total_current);

  // ── Part 7: Expenses (prior year) ──
  fields["pFuel"] = cur(data.expense_fuel_prior);
  fields["pLight"] = cur(data.expense_light_power_prior);
  fields["pCleaning"] = cur(data.expense_cleaning_prior);
  fields["pWages"] = cur(data.expense_wages_prior);
  fields["pRepairs"] = cur(data.expense_repairs_prior);
  fields["pManagement"] = cur(data.expense_management_prior);
  fields["pInsurance"] = cur(data.expense_insurance_prior);
  fields["pWater"] = cur(data.expense_water_sewer_prior);
  fields["pAdvertising"] = cur(data.expense_advertising_prior);
  fields["pInterior"] = cur(data.expense_painting_prior);
  fields["pAmortized"] = cur(data.expense_leasing_ti_prior);
  fields["pMisc"] = cur(data.expense_misc_prior);
  fields["pSubTot_Expense"] = cur(data.expense_before_taxes_prior);
  fields["pReal_Taxes"] = cur(data.expense_real_estate_taxes_prior);
  fields["pTotal_Expense"] = cur(data.expense_total_prior);

  // ── Part 8: Net ──
  fields["before_taxes"] = cur(data.net_before_re_taxes_current);
  fields["After_taxes"] = cur(data.net_after_re_taxes_current);
  fields["pbefore_taxes"] = cur(data.net_before_re_taxes_prior);
  fields["pAfter_taxes"] = cur(data.net_after_re_taxes_prior);

  // ── Part 9: Misc expense itemization (up to 8 rows) ──
  for (let i = 0; i < Math.min((data.misc_expenses ?? []).length, 8); i++) {
    const m = data.misc_expenses[i];
    if (m.item) fields[`item${i + 1}`] = v(m.item);
    if (m.amount != null) fields[`amount${i + 1}`] = cur(m.amount);
  }

  // ── Part 10: Tenants' electricity ──
  if (data.tenants_electricity_from_applicant) {
    fields["is_tenants_from_applicant"] = v(data.tenants_electricity_from_applicant);
  }
  if (data.tenants_electricity_separate_charge) {
    fields["is_separate_charge"] = v(data.tenants_electricity_separate_charge);
  }

  // Strip empty values
  for (const key of Object.keys(fields)) {
    if (!fields[key]) delete fields[key];
  }

  return fields;
}

/**
 * Generate a filled TC201 PDF entirely on the client side.
 * 1. Fetches the blank template from GET /api/tc201/template
 * 2. Fills AcroForm fields using pdf-lib
 * 3. Returns a Blob ready for download
 */
export async function generateTC201PDF(data: TC201Data): Promise<Blob> {
  // Fetch the blank template
  const templateRes = await fetch(`${API_BASE}/api/tc201/template`);
  if (!templateRes.ok) {
    throw new Error(`Failed to fetch TC201 template (${templateRes.status})`);
  }
  const templateBytes = await templateRes.arrayBuffer();

  // Load and fill the PDF
  const pdfDoc = await PDFDocument.load(templateBytes);
  const form = pdfDoc.getForm();
  const fieldMap = buildFieldMap(data);

  // Embed the same font the template uses (/TiBo = Times-Bold)
  const timesBold = await pdfDoc.embedFont(StandardFonts.TimesRomanBold);

  // Get all field names from the form to blank residual data
  const allFields = form.getFields();
  const allFieldNames = new Set(allFields.map((f) => f.getName()));

  // Clear all fields first (remove residual template data)
  for (const field of allFields) {
    const name = field.getName();
    if (name in fieldMap) continue; // will be set below
    try {
      const textField = form.getTextField(name);
      textField.setText("");
    } catch {
      // Not a text field (checkbox, etc.) — skip
    }
  }

  // Fill fields with our data
  for (const [fieldName, value] of Object.entries(fieldMap)) {
    if (!allFieldNames.has(fieldName)) continue;
    try {
      // Handle checkbox fields
      if (value === "/On" || value === "/Off") {
        const cb = form.getCheckBox(fieldName);
        if (value === "/On") cb.check();
        else cb.uncheck();
      } else {
        const textField = form.getTextField(fieldName);
        textField.setText(value);
      }
    } catch {
      // Field type mismatch — try as text anyway
      try {
        const textField = form.getTextField(fieldName);
        textField.setText(value);
      } catch {
        // Skip fields that can't be set
      }
    }
  }

  // Re-generate appearance streams using the correct font (Times-Bold)
  // so that filled values render with the same typeface as the template.
  form.updateFieldAppearances(timesBold);

  // Flatten so the form appears as printed text
  form.flatten();

  const pdfBytes = await pdfDoc.save();
  return new Blob([pdfBytes as unknown as BlobPart], { type: "application/pdf" });
}
