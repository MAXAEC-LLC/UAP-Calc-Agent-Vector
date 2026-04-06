"""
NYC Form TC201 – Income and Expense Schedule for Rent-Producing Properties.
Tax Commission of the City of New York, assessment year 2026/27.
"""

from pydantic import BaseModel, Field


# ── Part 3: Residential Occupancy row ────────────────────────────────
class ResidentialOccupancy(BaseModel):
    occupancy_type: str = ""            # RENTED REGULATED / RENTED UNREGULATED / OWNER OCCUPIED / VACANT
    number_of_units: int | None = None
    monthly_rent: float | None = None


# ── Part 4: Nonresidential Occupancy row ─────────────────────────────
class NonresidentialFloor(BaseModel):
    floor: str = ""                     # FLOOR 3-, 2ND FLOOR, 1ST FLOOR, BASEMENT
    applicant_related_sqft: float | None = None
    rented_sqft: float | None = None    # unrelated tenants
    vacant_sqft: float | None = None
    gross_sqft: float | None = None


# ── Part 9: Miscellaneous expense item ───────────────────────────────
class MiscExpenseItem(BaseModel):
    item: str = ""
    amount: float | None = None


class TC201Data(BaseModel):
    """Structured representation of NYC Form TC201 (Income & Expense Schedule)."""

    # ── Meta ─────────────────────────────────────────────────────────
    filename: str = ""
    assessment_year: str = "2026/27"

    # ── Part 1: Property Identification ──────────────────────────────
    borough: str = ""
    block: str = ""
    lot: str = ""
    bbl: str = ""                       # derived convenience
    tax_commission_group_no: str = ""
    is_condo: str = ""                  # Y/N
    covers_multiple_lots: str = ""      # Y/N
    total_lots: int | None = None
    related_lots: str = ""              # "Block xxxx Lots xxxx"
    covers_entire_lot: str = ""         # Y/N

    # ── Part 2: Reporting Period & Accounting Basis ──────────────────
    reporting_period_from: str = ""     # MM/DD/YYYY
    reporting_period_to: str = ""
    accounting_basis: str = ""          # Cash / Accrual
    accounting_basis_changed: str = ""  # Y/N

    # ── Part 3: Residential Occupancy (as of Jan 5 2026) ─────────────
    residential_occupancy: list[ResidentialOccupancy] = Field(default_factory=list)
    rent_includes_recurring_charges: str = ""  # Y/N

    # ── Part 4: Nonresidential Occupancy (as of Jan 5 2026) ──────────
    nonresidential_floors: list[NonresidentialFloor] = Field(default_factory=list)

    # ── Part 5: Lease Information ────────────────────────────────────
    entire_lot_leased: str = ""         # Y/N
    lease_type: str = ""                # Gross / Net / Ground
    applicant_receives_rental_income: str = ""  # Y/N
    lessor: str = ""
    lessee: str = ""
    lease_from: str = ""
    lease_to: str = ""
    annual_rent: float | None = None
    additional_sums: float | None = None
    lessor_pays_expenses: str = ""      # Y/N
    lessor_pays_details: str = ""
    land_only_lease: str = ""           # Y/N

    # ── Part 6: Income Information (prior_year / current_year) ───────
    income_residential_regulated_prior: float | None = None
    income_residential_regulated_current: float | None = None
    income_residential_unregulated_prior: float | None = None
    income_residential_unregulated_current: float | None = None
    income_residential_subtotal_prior: float | None = None
    income_residential_subtotal_current: float | None = None
    income_office_prior: float | None = None
    income_office_current: float | None = None
    income_retail_prior: float | None = None
    income_retail_current: float | None = None
    income_loft_prior: float | None = None
    income_loft_current: float | None = None
    income_factory_prior: float | None = None
    income_factory_current: float | None = None
    income_warehouse_prior: float | None = None
    income_warehouse_current: float | None = None
    income_storage_prior: float | None = None
    income_storage_current: float | None = None
    income_parking_prior: float | None = None
    income_parking_current: float | None = None
    income_subtotal_prior: float | None = None
    income_subtotal_current: float | None = None
    income_owner_occupied_prior: float | None = None
    income_owner_occupied_current: float | None = None
    # Ancillary
    income_operating_escalation_prior: float | None = None
    income_operating_escalation_current: float | None = None
    income_re_tax_escalation_prior: float | None = None
    income_re_tax_escalation_current: float | None = None
    income_utility_services_prior: float | None = None
    income_utility_services_current: float | None = None
    income_other_services_prior: float | None = None
    income_other_services_current: float | None = None
    income_govt_subsidies_prior: float | None = None
    income_govt_subsidies_current: float | None = None
    income_signage_prior: float | None = None
    income_signage_current: float | None = None
    income_cell_towers_prior: float | None = None
    income_cell_towers_current: float | None = None
    income_other_prior: float | None = None
    income_other_current: float | None = None
    income_other_description: str = ""
    income_total_gross_prior: float | None = None
    income_total_gross_current: float | None = None

    # ── Part 7: Expense Information (prior_year / current_year) ──────
    expense_fuel_prior: float | None = None
    expense_fuel_current: float | None = None
    expense_light_power_prior: float | None = None
    expense_light_power_current: float | None = None
    expense_cleaning_prior: float | None = None
    expense_cleaning_current: float | None = None
    expense_wages_prior: float | None = None
    expense_wages_current: float | None = None
    expense_repairs_prior: float | None = None
    expense_repairs_current: float | None = None
    expense_management_prior: float | None = None
    expense_management_current: float | None = None
    expense_insurance_prior: float | None = None
    expense_insurance_current: float | None = None
    expense_water_sewer_prior: float | None = None
    expense_water_sewer_current: float | None = None
    expense_advertising_prior: float | None = None
    expense_advertising_current: float | None = None
    expense_painting_prior: float | None = None
    expense_painting_current: float | None = None
    expense_leasing_ti_prior: float | None = None
    expense_leasing_ti_current: float | None = None
    expense_misc_prior: float | None = None
    expense_misc_current: float | None = None
    expense_before_taxes_prior: float | None = None
    expense_before_taxes_current: float | None = None
    expense_real_estate_taxes_prior: float | None = None
    expense_real_estate_taxes_current: float | None = None
    expense_total_prior: float | None = None
    expense_total_current: float | None = None

    # ── Part 8: Net Profit (or Loss) ─────────────────────────────────
    net_before_re_taxes_prior: float | None = None
    net_before_re_taxes_current: float | None = None
    net_after_re_taxes_prior: float | None = None
    net_after_re_taxes_current: float | None = None

    # ── Part 9: Misc Expenses Itemization ────────────────────────────
    misc_expenses: list[MiscExpenseItem] = Field(default_factory=list)

    # ── Part 10: Tenants' Electricity ────────────────────────────────
    tenants_electricity_from_applicant: str = ""  # Y/N
    tenants_electricity_separate_charge: str = ""  # Y/N

    # ── Notes ────────────────────────────────────────────────────────
    notes: str = ""
