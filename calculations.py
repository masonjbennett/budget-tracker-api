"""
Pure calculation engine — all financial math, zero framework dependencies.
Extracted from budget_app.py (Streamlit version).
All tax data is official IRS 2026 per Rev. Proc. 2025-32 + OBBBA.
"""

import numpy as np
from typing import Optional

# ──────────────────────────────────────────────
# FEDERAL TAX DATA (2026, IRS Rev. Proc. 2025-32)
# ──────────────────────────────────────────────

FEDERAL_BRACKETS_2026 = {
    "Single": [
        (12_400, 0.10), (50_400, 0.12), (105_700, 0.22),
        (201_775, 0.24), (256_225, 0.32), (640_600, 0.35),
        (float("inf"), 0.37),
    ],
    "Married Filing Jointly": [
        (24_800, 0.10), (100_800, 0.12), (211_400, 0.22),
        (403_550, 0.24), (512_450, 0.32), (768_700, 0.35),
        (float("inf"), 0.37),
    ],
    "Married Filing Separately": [
        (12_400, 0.10), (50_400, 0.12), (105_700, 0.22),
        (201_775, 0.24), (256_225, 0.32), (384_350, 0.35),
        (float("inf"), 0.37),
    ],
    "Head of Household": [
        (17_700, 0.10), (67_450, 0.12), (105_700, 0.22),
        (201_775, 0.24), (256_200, 0.32), (640_600, 0.35),
        (float("inf"), 0.37),
    ],
}

STANDARD_DEDUCTION_2026 = {
    "Single": 16_100,
    "Married Filing Jointly": 32_200,
    "Married Filing Separately": 16_100,
    "Head of Household": 24_150,
}

FILING_STATUSES = list(FEDERAL_BRACKETS_2026.keys())

# ──────────────────────────────────────────────
# STATE TAX DATA
# ──────────────────────────────────────────────

STATE_TAX_DATA = {
    "Alabama": {"brackets": [(500, 0.02), (3000, 0.04), (float("inf"), 0.05)], "deduction": 3000},
    "Alaska": {"brackets": [], "deduction": 0},
    "Arizona": {"brackets": [(float("inf"), 0.025)], "deduction": 14600},
    "Arkansas": {"brackets": [(4400, 0.02), (8800, 0.04), (float("inf"), 0.044)], "deduction": 2340},
    "California": {"brackets": [(10412, 0.01), (24684, 0.02), (38959, 0.04), (54081, 0.06), (68350, 0.08), (349137, 0.093), (418961, 0.103), (698271, 0.113), (float("inf"), 0.123)], "deduction": 5540, "deduction_mfj": 11080},
    "Colorado": {"brackets": [(float("inf"), 0.044)], "deduction": 15700},
    "Connecticut": {"brackets": [(10000, 0.03), (50000, 0.05), (100000, 0.055), (200000, 0.06), (250000, 0.065), (500000, 0.069), (float("inf"), 0.0699)],
                     "brackets_mfj": [(20000, 0.03), (100000, 0.05), (200000, 0.055), (400000, 0.06), (500000, 0.065), (1000000, 0.069), (float("inf"), 0.0699)],
                     "deduction": 0, "deduction_mfj": 0},
    "Delaware": {"brackets": [(2000, 0.0), (5000, 0.022), (10000, 0.039), (20000, 0.048), (25000, 0.052), (60000, 0.0555), (float("inf"), 0.066)], "deduction": 3250},
    "Florida": {"brackets": [], "deduction": 0},
    "Georgia": {"brackets": [(float("inf"), 0.0519)], "deduction": 12000},
    "Hawaii": {"brackets": [(2400, 0.014), (4800, 0.032), (9600, 0.055), (14400, 0.064), (19200, 0.068), (24000, 0.072), (36000, 0.076), (48000, 0.079), (150000, 0.0825), (175000, 0.09), (200000, 0.10), (float("inf"), 0.11)], "deduction": 2200},
    "Idaho": {"brackets": [(float("inf"), 0.053)], "deduction": 14700},
    "Illinois": {"brackets": [(float("inf"), 0.0495)], "deduction": 0},
    "Indiana": {"brackets": [(float("inf"), 0.0295)], "deduction": 0},
    "Iowa": {"brackets": [(float("inf"), 0.0295)], "deduction": 2210},
    "Kansas": {"brackets": [(15000, 0.031), (30000, 0.0525), (float("inf"), 0.057)], "deduction": 3500},
    "Kentucky": {"brackets": [(float("inf"), 0.035)], "deduction": 3160},
    "Louisiana": {"brackets": [(float("inf"), 0.03)], "deduction": 0},
    "Maine": {"brackets": [(24500, 0.058), (58050, 0.0675), (float("inf"), 0.0715)], "deduction": 14600},
    "Maryland": {"brackets": [(1000, 0.02), (2000, 0.03), (3000, 0.04), (100000, 0.0475), (125000, 0.05), (150000, 0.0525), (250000, 0.055), (float("inf"), 0.0575)],
                  "brackets_mfj": [(1000, 0.02), (2000, 0.03), (3000, 0.04), (150000, 0.0475), (175000, 0.05), (225000, 0.0525), (300000, 0.055), (float("inf"), 0.0575)],
                  "deduction": 2550, "deduction_mfj": 5100},
    "Massachusetts": {"brackets": [(float("inf"), 0.05)], "deduction": 0},
    "Michigan": {"brackets": [(float("inf"), 0.0405)], "deduction": 5400},
    "Minnesota": {"brackets": [(33310, 0.0535), (109430, 0.068), (203150, 0.0785), (float("inf"), 0.0985)],
                   "brackets_mfj": [(48700, 0.0535), (193480, 0.068), (337930, 0.0785), (float("inf"), 0.0985)],
                   "deduction": 14575, "deduction_mfj": 29150},
    "Mississippi": {"brackets": [(10000, 0.04), (float("inf"), 0.04)], "deduction": 2300},
    "Missouri": {"brackets": [(1207, 0.02), (2414, 0.025), (3621, 0.03), (4828, 0.035), (6035, 0.04), (7242, 0.045), (8449, 0.05), (float("inf"), 0.048)], "deduction": 14600},
    "Montana": {"brackets": [(20500, 0.047), (float("inf"), 0.0565)], "deduction": 14600},
    "Nebraska": {"brackets": [(3700, 0.0246), (22170, 0.0351), (35730, 0.0455), (float("inf"), 0.0455)], "deduction": 8200},
    "Nevada": {"brackets": [], "deduction": 0},
    "New Hampshire": {"brackets": [], "deduction": 0},
    "New Jersey": {"brackets": [(20000, 0.014), (35000, 0.0175), (40000, 0.035), (75000, 0.05525), (500000, 0.0637), (1000000, 0.0897), (float("inf"), 0.1075)],
                    "brackets_mfj": [(20000, 0.014), (50000, 0.0175), (70000, 0.035), (80000, 0.05525), (150000, 0.0637), (500000, 0.0897), (1000000, 0.1075), (float("inf"), 0.1075)],
                    "deduction": 0, "deduction_mfj": 0},
    "New Mexico": {"brackets": [(5500, 0.015), (16500, 0.032), (33500, 0.043), (66500, 0.047), (210000, 0.049), (float("inf"), 0.059)],
                    "brackets_mfj": [(8000, 0.015), (25000, 0.032), (50000, 0.043), (100000, 0.047), (315000, 0.049), (float("inf"), 0.059)],
                    "deduction": 14600, "deduction_mfj": 29200},
    "New York": {"brackets": [(8500, 0.04), (11700, 0.045), (13900, 0.0525), (80650, 0.055), (215400, 0.06), (1077550, 0.0685), (5000000, 0.0965), (25000000, 0.103), (float("inf"), 0.109)],
                  "brackets_mfj": [(17150, 0.04), (23600, 0.045), (27900, 0.0525), (161550, 0.055), (323200, 0.06), (2155350, 0.0685), (5000000, 0.0965), (25000000, 0.103), (float("inf"), 0.109)],
                  "deduction": 8000, "deduction_mfj": 16050},
    "North Carolina": {"brackets": [(float("inf"), 0.0399)], "deduction": 14600},
    "North Dakota": {"brackets": [(44725, 0.0195), (float("inf"), 0.025)], "deduction": 14600},
    "Ohio": {"brackets": [(26050, 0.0), (float("inf"), 0.0275)], "deduction": 0},
    "Oklahoma": {"brackets": [(7200, 0.025), (12200, 0.035), (float("inf"), 0.045)],
                  "brackets_mfj": [(12200, 0.025), (24400, 0.035), (float("inf"), 0.045)],
                  "deduction": 7350, "deduction_mfj": 14700},
    "Oregon": {"brackets": [(4050, 0.0475), (10200, 0.0675), (125000, 0.0875), (float("inf"), 0.099)], "deduction": 2745},
    "Pennsylvania": {"brackets": [(float("inf"), 0.0307)], "deduction": 0},
    "Rhode Island": {"brackets": [(73450, 0.0375), (166950, 0.0475), (float("inf"), 0.0599)], "deduction": 10550},
    "South Carolina": {"brackets": [(3460, 0.0), (17340, 0.03), (float("inf"), 0.064)], "deduction": 14600},
    "South Dakota": {"brackets": [], "deduction": 0},
    "Tennessee": {"brackets": [], "deduction": 0},
    "Texas": {"brackets": [], "deduction": 0},
    "Utah": {"brackets": [(float("inf"), 0.0465)], "deduction": 0},
    "Vermont": {"brackets": [(45400, 0.0335), (110450, 0.066), (229550, 0.076), (float("inf"), 0.0875)], "deduction": 7050},
    "Virginia": {"brackets": [(3000, 0.02), (5000, 0.03), (17000, 0.05), (float("inf"), 0.0575)], "deduction": 4500},
    "Washington": {"brackets": [], "deduction": 0},
    "West Virginia": {"brackets": [(10000, 0.0236), (25000, 0.0315), (40000, 0.0354), (60000, 0.0472), (float("inf"), 0.0512)], "deduction": 0},
    "Wisconsin": {"brackets": [(14680, 0.035), (29370, 0.044), (323290, 0.053), (float("inf"), 0.0765)],
                   "brackets_mfj": [(19580, 0.035), (39150, 0.044), (431060, 0.053), (float("inf"), 0.0765)],
                   "deduction": 13230, "deduction_mfj": 24440},
    "Wyoming": {"brackets": [], "deduction": 0},
    "District of Columbia": {"brackets": [(10000, 0.04), (40000, 0.06), (60000, 0.065), (250000, 0.085), (500000, 0.0925), (1000000, 0.0975), (float("inf"), 0.1075)], "deduction": 14600},
}

# ──────────────────────────────────────────────
# FICA / SALT CONSTANTS
# ──────────────────────────────────────────────

FICA_SS_RATE = 0.062
FICA_SS_CAP = 184_500
FICA_MEDICARE_RATE = 0.0145
FICA_MEDICARE_SURTAX = 0.009
FICA_MEDICARE_SURTAX_THRESHOLDS = {
    "Single": 200_000, "Head of Household": 200_000,
    "Married Filing Jointly": 250_000, "Married Filing Separately": 125_000,
}

SALT_CAP_BASE = 40_400
SALT_CAP_FLOOR = 10_000
SALT_PHASEOUT_THRESHOLD = {
    "Single": 505_000, "Head of Household": 505_000,
    "Married Filing Jointly": 505_000, "Married Filing Separately": 252_500,
}
SALT_PHASEOUT_RATE = 0.30

K401_LIMIT = 24_500
HSA_INDIVIDUAL_LIMIT = 4_400

# ──────────────────────────────────────────────
# COST OF LIVING INDEX
# ──────────────────────────────────────────────

COL_INDEX = {
    "National Average": 100.0,
    "New York, NY": 187.2, "San Francisco, CA": 179.6, "Los Angeles, CA": 166.2,
    "Washington, DC": 152.1, "Boston, MA": 148.4, "San Diego, CA": 146.5,
    "Seattle, WA": 143.3, "Miami, FL": 133.8, "Denver, CO": 128.9,
    "Chicago, IL": 117.3, "Portland, OR": 120.4, "Austin, TX": 103.4,
    "Philadelphia, PA": 114.8, "Minneapolis, MN": 108.2, "Nashville, TN": 103.1,
    "Atlanta, GA": 105.7, "Dallas, TX": 104.8, "Houston, TX": 96.5,
    "Charlotte, NC": 98.4, "Phoenix, AZ": 100.7, "Las Vegas, NV": 102.3,
    "Tampa, FL": 99.5, "Raleigh, NC": 102.9, "Columbus, OH": 93.8,
    "Salt Lake City, UT": 104.2, "Detroit, MI": 89.4, "St. Louis, MO": 87.1,
    "Kansas City, MO": 91.2, "Indianapolis, IN": 90.3, "Cincinnati, OH": 90.9,
    "Pittsburgh, PA": 92.4, "Cleveland, OH": 88.7, "San Antonio, TX": 89.9,
    "Jacksonville, FL": 96.1, "Oklahoma City, OK": 87.3, "Louisville, KY": 91.8,
    "Memphis, TN": 84.2, "Birmingham, AL": 88.1, "Buffalo, NY": 93.5,
    "Richmond, VA": 101.3, "Honolulu, HI": 192.9, "Anchorage, AK": 127.4,
    "Fayetteville, AR": 84.5,
}

# ──────────────────────────────────────────────
# TAX CALCULATIONS
# ──────────────────────────────────────────────

def calc_bracket_tax(taxable_income: float, brackets: list) -> float:
    tax = 0.0
    prev = 0
    for ceiling, rate in brackets:
        if taxable_income <= 0:
            break
        span = min(taxable_income, ceiling - prev)
        tax += span * rate
        taxable_income -= span
        prev = ceiling
    return tax


def calc_student_loan_deduction(interest_paid: float, magi: float, filing: str = "Single") -> float:
    if filing == "Married Filing Separately":
        return 0
    max_ded = 2_500
    if filing == "Married Filing Jointly":
        lower, upper = 175_000, 205_000
    else:
        lower, upper = 85_000, 100_000
    if magi <= lower:
        return min(interest_paid, max_ded)
    if magi >= upper:
        return 0
    reduction = (magi - lower) / (upper - lower)
    return min(interest_paid, max_ded) * (1 - reduction)


def calc_federal_tax(gross: float, deductions_401k: float = 0, other_pretax: float = 0,
                     filing: str = "Single", student_loan_interest: float = 0,
                     charitable_cash: float = 0) -> dict:
    brackets = FEDERAL_BRACKETS_2026.get(filing, FEDERAL_BRACKETS_2026["Single"])
    standard = STANDARD_DEDUCTION_2026.get(filing, 16_100)
    agi = gross - deductions_401k - other_pretax
    sl_deduction = calc_student_loan_deduction(student_loan_interest, agi, filing)
    agi -= sl_deduction
    non_itemizer_limit = 2000 if filing == "Married Filing Jointly" else 1000
    charitable_atl = min(charitable_cash, non_itemizer_limit)
    agi -= charitable_atl
    taxable = max(0, agi - standard)
    tax = calc_bracket_tax(taxable, brackets)
    return {"tax": tax, "agi": agi, "taxable": taxable, "standard_deduction": standard}


def _get_state_brackets_for_filing(sdata: dict, filing: str):
    is_joint = filing == "Married Filing Jointly"
    if not is_joint:
        return sdata["brackets"], sdata["deduction"]
    if "brackets_mfj" in sdata:
        return sdata["brackets_mfj"], sdata.get("deduction_mfj", sdata["deduction"] * 2)
    mfj_brackets = [(c * 2 if c != float("inf") else float("inf"), r) for c, r in sdata["brackets"]]
    return mfj_brackets, sdata.get("deduction_mfj", sdata["deduction"] * 2)


def calc_state_tax(gross: float, state: str, deductions_401k: float = 0,
                   other_pretax: float = 0, filing: str = "Single") -> float:
    sdata = STATE_TAX_DATA.get(state)
    if not sdata or not sdata["brackets"]:
        return 0.0
    agi = gross - deductions_401k - other_pretax
    brackets, deduction = _get_state_brackets_for_filing(sdata, filing)
    taxable = max(0, agi - deduction)
    return calc_bracket_tax(taxable, brackets)


def calc_fica(gross: float, filing: str = "Single") -> float:
    ss = min(gross, FICA_SS_CAP) * FICA_SS_RATE
    medicare = gross * FICA_MEDICARE_RATE
    surtax_threshold = FICA_MEDICARE_SURTAX_THRESHOLDS.get(filing, 200_000)
    if gross > surtax_threshold:
        medicare += (gross - surtax_threshold) * FICA_MEDICARE_SURTAX
    return ss + medicare


def calc_salt_cap(magi: float, filing: str = "Single") -> float:
    threshold = SALT_PHASEOUT_THRESHOLD.get(filing, 505_000)
    mfs_cap = SALT_CAP_BASE // 2 if filing == "Married Filing Separately" else SALT_CAP_BASE
    if magi <= threshold:
        return mfs_cap
    reduction = SALT_PHASEOUT_RATE * (magi - threshold)
    return max(SALT_CAP_FLOOR, mfs_cap - reduction)


def compute_take_home(income: dict, charitable: float = 0) -> dict:
    gross = income["gross_salary"]
    bonus = income.get("bonus_amount", 0)
    bonus_type = income.get("bonus_type", "None")
    filing = income.get("filing_status", "Single")
    annual_gross = gross + (bonus if bonus_type != "None" else 0)

    contrib_401k = min(gross * income["contribution_401k"] / 100, K401_LIMIT)
    health = income["health_insurance"] * 12
    hsa = income["hsa"] * 12
    sl_interest = income.get("student_loan_interest", 0)
    pretax = contrib_401k + health + hsa

    fed = calc_federal_tax(annual_gross, contrib_401k, health + hsa, filing, sl_interest, charitable)
    state = calc_state_tax(annual_gross, income["state"], contrib_401k, health + hsa, filing)
    fica = calc_fica(annual_gross, filing)

    total_tax = fed["tax"] + state + fica
    annual_take_home = annual_gross - pretax - total_tax
    brackets = FEDERAL_BRACKETS_2026.get(filing, FEDERAL_BRACKETS_2026["Single"])

    return {
        "annual_gross": annual_gross,
        "contrib_401k": contrib_401k,
        "health": health,
        "hsa": hsa,
        "pretax": pretax,
        "fed_tax": fed["tax"],
        "state_tax": state,
        "fica": fica,
        "total_tax": total_tax,
        "annual_take_home": annual_take_home,
        "monthly_take_home": annual_take_home / 12,
        "agi": fed["agi"],
        "taxable": fed["taxable"],
        "std_ded": fed["standard_deduction"],
        "effective_rate": (total_tax / annual_gross * 100) if annual_gross else 0,
        "marginal_fed": _get_marginal_rate(fed["taxable"], brackets),
        "filing": filing,
    }


def _get_marginal_rate(taxable: float, brackets: list) -> float:
    for ceiling, rate in brackets:
        if taxable <= ceiling:
            return rate * 100
    return brackets[-1][1] * 100


# ──────────────────────────────────────────────
# SOCIAL SECURITY
# ──────────────────────────────────────────────

def calc_social_security(annual_salary: float, claiming_age: int = 67) -> float:
    aime = min(annual_salary, FICA_SS_CAP) / 12
    if aime <= 1286:
        pia = aime * 0.90
    elif aime <= 7749:
        pia = 1286 * 0.90 + (aime - 1286) * 0.32
    else:
        pia = 1286 * 0.90 + (7749 - 1286) * 0.32 + (aime - 7749) * 0.15
    fra = 67
    if claiming_age < fra:
        months_early = (fra - claiming_age) * 12
        if months_early <= 36:
            reduction = months_early * (5 / 9 / 100)
        else:
            reduction = 36 * (5 / 9 / 100) + (months_early - 36) * (5 / 12 / 100)
        pia *= (1 - reduction)
    elif claiming_age > fra:
        months_late = min((claiming_age - fra) * 12, 36)
        pia *= (1 + months_late * (8 / 12 / 100))
    return max(0, pia)


# ──────────────────────────────────────────────
# DEBT PAYOFF
# ──────────────────────────────────────────────

def simulate_payoff(debts: list, extra: float, strategy: str) -> dict:
    balances = {d["name"]: float(d["balance"]) for d in debts}
    rates = {d["name"]: d["rate"] / 100 / 12 for d in debts}
    mins = {d["name"]: float(d["min_payment"]) for d in debts}
    total_interest = 0
    months = 0
    schedule = []
    payoff_months = {}

    total_min = sum(mins.values()) + extra
    total_monthly_interest = sum(b * r for b, r in zip(balances.values(), rates.values()) if b > 0)
    if total_min <= 0 or (total_min <= total_monthly_interest and total_monthly_interest > 0):
        return {"months": -1, "total_interest": 0, "schedule": [], "payoff_months": {}}

    while any(b > 0.01 for b in balances.values()) and months < 600:
        months += 1
        month_interest = 0
        for name in balances:
            if balances[name] > 0:
                interest = balances[name] * rates[name]
                balances[name] += interest
                month_interest += interest
                total_interest += interest

        for name in balances:
            if balances[name] > 0:
                payment = min(mins[name], balances[name])
                balances[name] -= payment

        if strategy == "avalanche":
            order = sorted([n for n in balances if balances[n] > 0], key=lambda n: rates[n], reverse=True)
        else:
            order = sorted([n for n in balances if balances[n] > 0], key=lambda n: balances[n])

        remaining_extra = extra
        for name in order:
            if remaining_extra <= 0:
                break
            if balances[name] > 0:
                payment = min(remaining_extra, balances[name])
                balances[name] -= payment
                remaining_extra -= payment

        for name in balances:
            if balances[name] <= 0.01 and name not in payoff_months:
                payoff_months[name] = months

        schedule.append({
            "month": months,
            "total_balance": sum(max(0, b) for b in balances.values()),
            "interest": month_interest,
        })

    return {
        "months": months,
        "total_interest": total_interest,
        "schedule": schedule,
        "payoff_months": payoff_months,
    }


# ──────────────────────────────────────────────
# INVESTMENT PROJECTION
# ──────────────────────────────────────────────

def project_investment(start: float, monthly: float, rate: float,
                       years: int, contribution_growth: float = 0) -> dict:
    values = [start]
    contributions = [start]
    r = rate / 100 / 12
    current_monthly = monthly
    for m in range(1, years * 12 + 1):
        if m > 1 and m % 12 == 1:
            current_monthly = monthly * (1 + contribution_growth / 100) ** ((m - 1) // 12)
        values.append(values[-1] * (1 + r) + current_monthly)
        contributions.append(contributions[-1] + current_monthly)
    return {"values": values, "contributions": contributions}


# ──────────────────────────────────────────────
# MONTE CARLO SIMULATION
# ──────────────────────────────────────────────

def run_monte_carlo(
    current_age: int,
    retire_age: int,
    end_age: int,
    portfolio: float,
    annual_savings: float,
    annual_expenses: float,
    stock_pct: float = 80,
    inflation: float = 3.0,
    n_sims: int = 1000,
) -> dict:
    stock_alloc = stock_pct / 100
    bond_alloc = 1.0 - stock_alloc
    total_years = end_age - current_age
    inflation_mean = inflation / 100

    corr_matrix = np.array([[1.0, 0.05], [0.05, 1.0]])
    cholesky = np.linalg.cholesky(corr_matrix)

    paths = np.zeros((n_sims, total_years + 1))
    paths[:, 0] = portfolio
    failure_ages = []

    for sim in range(n_sims):
        port = float(portfolio)
        failed = False
        for year in range(1, total_years + 1):
            if failed:
                paths[sim, year] = 0
                continue
            z = np.random.normal(size=2)
            correlated = cholesky @ z
            stock_ret = 0.10 + 0.18 * correlated[0]
            bond_ret = 0.05 + 0.06 * correlated[1]
            port_ret = stock_alloc * stock_ret + bond_alloc * bond_ret
            yr_inflation = max(0, np.random.normal(inflation_mean, 0.015))

            age = current_age + year
            if age <= retire_age:
                inf_factor = (1 + inflation_mean) ** year
                port = port * (1 + port_ret) + annual_savings * inf_factor
            else:
                yrs_retired = age - retire_age
                inf_factor = (1 + yr_inflation) ** yrs_retired
                port = port * (1 + port_ret) - annual_expenses * inf_factor

            if port <= 0:
                port = 0
                failed = True
                failure_ages.append(age)
            paths[sim, year] = port

    ending = paths[:, -1]
    success_count = int(np.sum(ending > 0))

    return {
        "success_rate": success_count / n_sims * 100,
        "success_count": success_count,
        "n_sims": n_sims,
        "ages": list(range(current_age, end_age + 1)),
        "p5": paths[..., :].tolist() if n_sims <= 50 else [],  # only return paths for small sims
        "percentiles": {
            "p5": np.percentile(paths, 5, axis=0).tolist(),
            "p10": np.percentile(paths, 10, axis=0).tolist(),
            "p25": np.percentile(paths, 25, axis=0).tolist(),
            "p50": np.percentile(paths, 50, axis=0).tolist(),
            "p75": np.percentile(paths, 75, axis=0).tolist(),
            "p90": np.percentile(paths, 90, axis=0).tolist(),
            "p95": np.percentile(paths, 95, axis=0).tolist(),
        },
        "ending_values": ending.tolist(),
        "median_ending": float(np.median(ending)),
        "p10_ending": float(max(0, np.percentile(ending, 10))),
        "p90_ending": float(np.percentile(ending, 90)),
        "failure_ages": failure_ages,
        "retire_age": retire_age,
        "stock_pct": stock_pct,
    }
