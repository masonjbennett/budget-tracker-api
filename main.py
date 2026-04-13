"""FastAPI backend for Budget Tracker v2."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import os
from dotenv import load_dotenv
from supabase import create_client

from calculations import (
    compute_take_home, calc_federal_tax, calc_state_tax, calc_fica,
    calc_salt_cap, calc_social_security, calc_student_loan_deduction,
    simulate_payoff, project_investment, run_monte_carlo,
    FEDERAL_BRACKETS_2026, STANDARD_DEDUCTION_2026, STATE_TAX_DATA,
    FILING_STATUSES, COL_INDEX, K401_LIMIT, HSA_INDIVIDUAL_LIMIT,
)

load_dotenv()

app = FastAPI(title="Budget Tracker API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app", "https://budget-tracker-v2.vercel.app", "https://budget-tracker-v2-masonjbennett.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


# ──────────────────────────────────────────────
# REQUEST MODELS
# ──────────────────────────────────────────────

class IncomeInput(BaseModel):
    gross_salary: float = 100000
    state: str = "New York"
    filing_status: str = "Single"
    contribution_401k: float = 6
    health_insurance: float = 200
    hsa: float = 0
    bonus_amount: float = 0
    bonus_type: str = "None"
    student_loan_interest: float = 0
    charitable: float = 0


class DebtInput(BaseModel):
    name: str
    balance: float
    rate: float
    min_payment: float


class DebtPayoffInput(BaseModel):
    debts: list[DebtInput]
    extra: float = 200
    strategy: str = "avalanche"


class InvestmentInput(BaseModel):
    starting_amount: float = 5000
    monthly_contribution: float = 500
    annual_return: float = 7.0
    time_horizon: int = 30
    contribution_growth: float = 0


class MonteCarloInput(BaseModel):
    current_age: int = 24
    retire_age: int = 55
    end_age: int = 95
    portfolio: float = 25000
    annual_savings: float = 30000
    annual_expenses: float = 50000
    stock_pct: float = 80
    inflation: float = 3.0
    n_sims: int = 1000


class SocialSecurityInput(BaseModel):
    annual_salary: float = 100000
    claiming_age: int = 67


class SaltCapInput(BaseModel):
    magi: float
    filing: str = "Single"


class RothTraditionalInput(BaseModel):
    annual_contribution: float = 24500
    current_marginal_rate: float = 0.22
    future_tax_rate: float = 0.15
    annual_return: float = 7.0
    years: int = 30


class UserDataInput(BaseModel):
    access_token: str
    app_data: dict


class AuthInput(BaseModel):
    email: str
    password: str


# ──────────────────────────────────────────────
# TAX ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/api/take-home")
def api_take_home(input: IncomeInput):
    income = input.model_dump()
    return compute_take_home(income, input.charitable)


@app.post("/api/state-tax")
def api_state_tax(input: IncomeInput):
    return {"state_tax": calc_state_tax(
        input.gross_salary, input.state,
        min(input.gross_salary * input.contribution_401k / 100, K401_LIMIT),
        input.health_insurance * 12 + input.hsa * 12,
        input.filing_status,
    )}


@app.post("/api/salt-cap")
def api_salt_cap(input: SaltCapInput):
    return {"effective_cap": calc_salt_cap(input.magi, input.filing)}


@app.get("/api/tax-data")
def api_tax_data():
    def serialize_brackets(brackets):
        return [(c if c != float("inf") else "inf", r) for c, r in brackets]
    return {
        "federal_brackets": {k: serialize_brackets(v) for k, v in FEDERAL_BRACKETS_2026.items()},
        "standard_deductions": STANDARD_DEDUCTION_2026,
        "filing_statuses": FILING_STATUSES,
        "states": sorted(STATE_TAX_DATA.keys()),
        "k401_limit": K401_LIMIT,
        "hsa_limit": HSA_INDIVIDUAL_LIMIT,
    }


# ──────────────────────────────────────────────
# FINANCIAL PLANNING ENDPOINTS
# ──────────────────────────────────────────────

@app.post("/api/debt-payoff")
def api_debt_payoff(input: DebtPayoffInput):
    debts = [d.model_dump() for d in input.debts]
    return simulate_payoff(debts, input.extra, input.strategy)


@app.post("/api/investment-project")
def api_investment(input: InvestmentInput):
    return project_investment(
        input.starting_amount, input.monthly_contribution,
        input.annual_return, input.time_horizon, input.contribution_growth,
    )


@app.post("/api/monte-carlo")
def api_monte_carlo(input: MonteCarloInput):
    return run_monte_carlo(
        input.current_age, input.retire_age, input.end_age,
        input.portfolio, input.annual_savings, input.annual_expenses,
        input.stock_pct, input.inflation, input.n_sims,
    )


@app.post("/api/social-security")
def api_social_security(input: SocialSecurityInput):
    monthly = calc_social_security(input.annual_salary, input.claiming_age)
    return {"monthly": monthly, "annual": monthly * 12}


@app.post("/api/roth-vs-traditional")
def api_roth_traditional(input: RothTraditionalInput):
    r = input.annual_return / 100
    fv_factor = ((1 + r) ** input.years - 1) / r * (1 + r) if r > 0 else input.years

    trad_future = input.annual_contribution * fv_factor
    trad_after_tax = trad_future * (1 - input.future_tax_rate)

    roth_actual = input.annual_contribution * (1 - input.current_marginal_rate)
    roth_future = roth_actual * fv_factor

    return {
        "traditional_future": trad_future,
        "traditional_after_tax": trad_after_tax,
        "roth_future": roth_future,
        "better": "Traditional" if trad_after_tax > roth_future else "Roth",
        "difference": abs(trad_after_tax - roth_future),
    }


# ──────────────────────────────────────────────
# REFERENCE DATA
# ──────────────────────────────────────────────

@app.get("/api/col-index")
def api_col_index():
    return COL_INDEX


# ──────────────────────────────────────────────
# AUTH + DATA PERSISTENCE
# ──────────────────────────────────────────────

@app.post("/api/auth/signup")
def api_signup(input: AuthInput):
    try:
        response = supabase.auth.sign_up({"email": input.email, "password": input.password})
        if response.user:
            return {
                "user": {"id": response.user.id, "email": response.user.email},
                "access_token": response.session.access_token,
            }
        raise HTTPException(400, "Sign up failed")
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/auth/login")
def api_login(input: AuthInput):
    try:
        response = supabase.auth.sign_in_with_password({"email": input.email, "password": input.password})
        if response.user:
            return {
                "user": {"id": response.user.id, "email": response.user.email},
                "access_token": response.session.access_token,
            }
        raise HTTPException(401, "Invalid credentials")
    except Exception:
        raise HTTPException(401, "Invalid email or password")


@app.post("/api/data/save")
def api_save_data(input: UserDataInput):
    try:
        user = supabase.auth.get_user(input.access_token)
        if not user or not user.user:
            raise HTTPException(401, "Invalid token")
        user_id = user.user.id
        response = (
            supabase.table("user_data")
            .upsert({"user_id": user_id, "app_data": input.app_data}, on_conflict="user_id")
            .execute()
        )
        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/data/load")
def api_load_data(access_token: str):
    try:
        user = supabase.auth.get_user(access_token)
        if not user or not user.user:
            raise HTTPException(401, "Invalid token")
        response = (
            supabase.table("user_data")
            .select("app_data")
            .eq("user_id", user.user.id)
            .single()
            .execute()
        )
        return response.data["app_data"] if response.data else {}
    except HTTPException:
        raise
    except Exception:
        return {}


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "4.0"}
