"""
NOVA FX Risk Copilot — Quantitative & Retrieval Layer
Multi-pair simulation (USD/CNH, EUR/CNH, HKD/CNH, GBP/CNH) · MA5 · Colloquial RAG
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    REQUESTS_AVAILABLE = False

import numpy as np
import pandas as pd

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment,misc]
    FAISS_AVAILABLE = False

try:
    import streamlit as st
except ImportError:  # pragma: no cover — optional when backend imported outside Streamlit
    st = None  # type: ignore[assignment]


def get_secure_secret(key: str) -> str:
    """Resolve a credential from Streamlit secrets, then OS environment."""
    if st is not None:
        try:
            secrets_obj = getattr(st, "secrets", None)
            if secrets_obj is not None and key in secrets_obj:
                value = secrets_obj[key]
                if value is not None and str(value).strip():
                    return str(value).strip()
        except (FileNotFoundError, KeyError, TypeError, RuntimeError):
            pass
    return os.environ.get(key, "").strip()


def get_secure_openai_key() -> str:
    return get_secure_secret("OPENAI_API_KEY")


def get_secure_alpha_vantage_key() -> str:
    return get_secure_secret("ALPHA_VANTAGE_KEY")


EMBED_DIM = 384
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_vector_index: Any = None
_indexed_docs_corpus: list[dict[str, str]] = []
_embedding_matrix: np.ndarray | None = None
_rag_lock = threading.RLock()

HTTP_TIMEOUT_SEC = 10.0
_MAX_USER_QUERY_LEN = 500
_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"system[_\s-]*prompt\s*override", re.IGNORECASE),
    re.compile(r"override\s+system[_\s-]*prompt", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:a\s+)?(?:system|developer|admin)\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
)

SUPPORTED_PAIRS = ["USD/CNH", "EUR/CNH", "HKD/CNH", "GBP/CNH"]
DEFAULT_PAIR = "USD/CNH"

# ISO 4217 desk conventions (offshore trading vs onshore policy parity)
OFFSHORE_QUOTE_CCY = "CNH"
ONSHORE_QUOTE_CCY = "CNY"
CHINA_MACRO_ANCHOR = "China"

# VAT net-payout overlay — aligned with net_payout_rolling_hedge_eu_vat / amazon_uk_gbp_vat_funnel KB
VAT_MAP: dict[str, float] = {
    "USD/CNH": 0.0,
    "EUR/CNH": 0.20,
    "GBP/CNH": 0.20,
    "HKD/CNH": 0.0,
}
VAT_STANDARD_DEDUCTION_RATE = VAT_MAP["EUR/CNH"]


def _cofer_reserve_pct(cofer: dict[str, Any], currency: str = ONSHORE_QUOTE_CCY) -> Any:
    """COFER sovereign reserve share; tolerates legacy ``RMB`` keys in cached KB snapshots."""
    if currency == ONSHORE_QUOTE_CCY:
        return cofer.get(ONSHORE_QUOTE_CCY, cofer.get("RMB"))
    return cofer.get(currency)


def _fmt_cofer_reserve_label(cofer: dict[str, Any]) -> str:
    """Display-safe COFER CNY/RMB share for macro UI blocks."""
    raw = _cofer_reserve_pct(cofer)
    if raw is None:
        return "N/A"
    try:
        return f"{float(raw)}"
    except (TypeError, ValueError):
        return str(raw)

PAIR_META: dict[str, dict[str, Any]] = {
    "USD/CNH": {
        "label": "US Dollar vs Offshore Yuan (CNH)",
        "base_ccy": "USD",
        "symbol": "$",
        "center": 7.22,
        "floor": 7.10,
        "ceiling": 7.40,
        "init": 7.24,
        "shock": 0.008,
        "reversion": 0.35,
        "cycle_amp": 0.06,
        "seed": 2026,
    },
    "EUR/CNH": {
        "label": "Euro vs Offshore Yuan (CNH)",
        "base_ccy": "EUR",
        "symbol": "€",
        "center": 7.85,
        "floor": 7.65,
        "ceiling": 8.05,
        "init": 7.87,
        "shock": 0.010,
        "reversion": 0.30,
        "cycle_amp": 0.05,
        "seed": 3026,
    },
    "HKD/CNH": {
        "label": "Hong Kong Dollar vs Offshore Yuan (CNH)",
        "base_ccy": "HKD",
        "symbol": "HK$",
        "center": 0.925,
        "floor": 0.905,
        "ceiling": 0.945,
        "init": 0.932,
        "shock": 0.0025,
        "reversion": 0.55,
        "peg_mid": 0.925,
        "peg_band": 0.012,
        "weakening_total": 0.018,
        "seed": 4026,
    },
    "GBP/CNH": {
        "label": "British Pound vs Offshore Yuan (CNH)",
        "base_ccy": "GBP",
        "symbol": "£",
        "center": 9.05,
        "floor": 8.75,
        "ceiling": 9.35,
        "init": 9.08,
        "shock": 0.011,
        "reversion": 0.32,
        "cycle_amp": 0.05,
        "seed": 5026,
    },
}

YFINANCE_PAIR_TICKERS: dict[str, str] = {
    "USD/CNH": "USDCNH=X",
    "EUR/CNH": "EURCNH=X",
    "HKD/CNH": "HKDCNH=X",
    "GBP/CNH": "GBPCNH=X",
}

ALPHA_VANTAGE_PAIR_MAP: dict[str, tuple[str, str]] = {
    "USD/CNH": ("USD", "CNH"),
    "EUR/CNH": ("EUR", "CNH"),
    "HKD/CNH": ("HKD", "CNH"),
    "GBP/CNH": ("GBP", "CNH"),
}

_COLLOQUIAL_ROUTER: dict[str, list[str]] = {
    "losing money": ["hedge", "forward", "locking", "exposure", "loss", "bleeding"],
    "lose money": ["hedge", "forward", "locking", "exposure", "loss"],
    "lost money": ["hedge", "forward", "locking", "exposure", "loss"],
    "bleeding": ["hedge", "forward", "locking", "exposure", "loss", "bleed"],
    "bleed": ["hedge", "forward", "locking", "exposure", "loss"],
    "lost": ["hedge", "forward", "locking", "exposure", "loss"],
    "stop bleeding": ["hedge", "forward", "locking", "exposure"],
    "burning": ["hedge", "forward", "exposure", "loss"],
    "cheap": ["macro", "trend", "q2", "seasonal", "spot"],
    "good price": ["macro", "trend", "q2", "forward", "locking"],
    "great rate": ["macro", "trend", "forward", "locking"],
    "expensive": ["hedge", "forward", "locking", "exposure"],
    "safe": ["forward", "locking", "audit", "compliance", "hedge"],
    "safer": ["forward", "locking", "audit", "compliance", "hedge"],
    "worry": ["forward", "locking", "audit", "exposure", "hedge"],
    "worried": ["forward", "locking", "audit", "exposure", "hedge"],
    "scared": ["forward", "locking", "hedge", "exposure"],
    "nervous": ["forward", "locking", "hedge", "exposure"],
    "payout": ["forward", "exposure", "settlement", "receivable", "amazon"],
    "getting paid": ["forward", "exposure", "settlement", "receivable"],
    "amazon": ["amazon", "seller", "case", "seasonal", "exposure"],
    "uk": ["gbp", "vat", "uk", "british", "pound", "amazon"],
    "british": ["gbp", "uk", "pound", "vat"],
    "pound": ["gbp", "uk", "british"],
    "gbp": ["gbp", "uk", "vat", "british"],
    "fortnight": ["forward", "settlement", "locking", "gbp"],
    "tiktok": ["tiktok", "seller", "case", "seasonal"],
    "june": ["q2", "seasonal", "macro", "june", "settlement"],
    "next month": ["forward", "exposure", "settlement", "locking"],
    "trap": ["seasonal", "macro", "q2", "risk"],
    "seasonal": ["seasonal", "macro", "q2", "settlement"],
    "vat": ["vat", "euro", "eur", "remittance", "europe"],
    "europe": ["euro", "eur", "vat", "remittance"],
    "euro": ["euro", "eur", "vat", "remittance"],
    "hong kong": ["hkd", "peg", "depreciation", "hkma"],
    "hkd": ["hkd", "peg", "depreciation", "hkma"],
    "peg": ["hkd", "peg", "hkma", "depreciation"],
    "depreciation": ["hkd", "depreciation", "peg", "defense"],
}

IMF_MACRO_KB_ID = "imf_macro_inflation_base"

_FX_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "id": "high_frequency_micro_payouts",
        "pairs": "USD/CNH,EUR/CNH,HKD/CNH,GBP/CNH",
        "domain": "Cross-Border E-Commerce Micro Revenues & Natural Hedging",
        "title": "Natural Hedging and Cost Control for High-Frequency Micro-Payouts",
        "professional_definition": (
            "Cross-border e-commerce sellers experience fragmented, small-scale, multi-currency revenue streams "
            "with severe margin attrition from immediate bank settlements due to high platform spreads. "
            "Implementing Natural Hedging by holding foreign balance in digital multi-currency accounts "
            "to directly settle overseas Meta/Google ad spend and FBA warehousing fees eliminates double-conversion frictional costs."
        ),
        "layman_translation": (
            "Cross-border e-commerce sellers receive small, scattered retail payouts in USD or EUR every day "
            "(low-value, high-frequency, multi-currency). Constantly cashing out and converting these funds back home "
            "means getting eaten alive by high double-conversion fees and exchange rate spreads from platforms and banks. "
            "The most practical move is 'collect foreign currency abroad, and pay foreign ad spend and warehousing fees directly.' "
            "Avoid converting funds back to local currency unless absolutely necessary, thereby achieving a natural hedge."
        ),
        "actionable_playbook": (
            "1) Use the Live Risk Desk to monitor the 30-Day Range and MA5 trend trigger before initializing cash repatriation. "
            "2) Map upcoming overseas obligations (Google/Meta Ads billings, FBA storage fees) against wallet balances. "
            "3) Retain matching foreign currency equivalents in offshore accounts for direct payment, bypassing double-conversion. "
            "4) Secure authorized API tokens from storefronts (Amazon/Shopify) to clear automated Anti-Money Laundering (AML) controls."
        ),
        "tags": "high-frequency,micro,payout,natural hedge,spread,advertising,fba,amazon,shopify,bleeding,loss,fee",
    },
    {
        "id": "net_payout_rolling_hedge_eu_vat",
        "pairs": "EUR/CNH,USD/CNH",
        "domain": "EU VAT Multi-Factor Net Repatriation Strategy",
        "title": "Net Payout Rolling Hedge: Avoiding Over-Hedging Traps Under EU VAT Rules",
        "professional_definition": (
            "Under Council Directive (EU) 2017/2455 and 2019/1995 (EU VAT 2021 Rules), e-commerce platforms automatically "
            "withhold 19%-23% VAT and commissions at the point of sale. Hedging based on gross sales creates an over-hedging trap "
            ",forcing sellers to buy expensive currency in the spot market for forward contract maturity physical delivery. "
            "Sellers must execute Net Payout Hedging aligned with Bollinger Bands upper barrier volatility metrics."
        ),
        "layman_translation": (
            "For Europe-station sellers with a 30-to-90 day payment cycle, remember: never lock in exchange rates with banks "
            "based on your gross sales. EU e-commerce law (VAT rules) requires platforms to force-deduct around 20% for VAT "
            "and commissions right when a sale happens. Over-hedging means you won't have enough Euros on hand for physical delivery "
            "upon contract maturity; this turns 'hedging' into 'currency speculation' and can lead to catastrophic margin calls. "
            "You must deduct the taxes first and only perform dynamic, layered rolling hedges on your net payout exposure "
            "when triggered at the statistically high Bollinger Upper Band."
        ),
        "actionable_playbook": (
            "1) Deduct an automatic 20% buffer from raw gross sales to establish the true Net Risk Exposure. "
            "2) Check backend indicators: If Spot price breaks above the 20-day Bollinger Upper Band and sits higher than the MA30 macro trend line, the multi-factor risk score triggers a high-level shield. "
            "3) Execute an aggressive 80% lock on 30-to-90 day net payout tranches via rolling forward options. "
            "4) If Spot remains flat or below MA5/MA30, drop the hedge ratio to 30% baseline to retain upside flexibility."
        ),
        "tags": "vat,eu,euro,repatriation,over-hedge,net payout,bollinger,ma30,rolling hedge,gross,commission,lock",
    },
    {
        "id": "compliance_safe_harbor_allowance",
        "pairs": "USD/CNH,EUR/CNH,HKD/CNH,GBP/CNH",
        "domain": "SAFE Compliance and Capital Item Discretionary Settlements",
        "title": "SAFE Compliance: Discretionary Settlement and Petty Cash Exemptions",
        "professional_definition": (
            "According to the State Administration of Foreign Exchange (SAFE) reforms on capital items, "
            "enterprises enjoy Discretionary Settlement. However, banks must enforce stringent ex-post auditing "
            "and trade background verification. To maintain business continuity, SMEs should utilize the "
            "Petty Cash Safe Harbor capped at an aggregate of USD 200,000 equivalent per institution per calendar month."
        ),
        "layman_translation": (
            "According to the latest SAFE regulations on capital account foreign exchange settlement, the government allows "
            "companies to settle forex at their own discretion. However, to prevent hot money speculation, banks enforce "
            "extremely strict post-settlement audits. If small and medium-sized entities want to avoid having their cash flow "
            "frozen by banks, they should take advantage of the 'petty cash exemption safe harbor': as long as the cumulative "
            "monthly petty cash settlement per institution does not exceed the equivalent of $200,000, you can be temporarily "
            "exempt from submitting tedious trade background documentation to the bank. But for any large, one-time lump-sum "
            "settlements, you must prepare all underlying supporting documents, or the bank will reject it outright."
        ),
        "actionable_playbook": (
            "1) Track all discretionary capital inflows, cross-border loans, or foreign investment injections inside backend ledgers. "
            "2) For urgent operational cash flow, restrict monthly settlement requests to under the equivalent of USD 200,000 via the petty cash route. "
            "3) Ensure complete commercial invoicing and logistics documentation are systematically archived for any single lump-sum settlement exceeding the safe harbor threshold. "
            "4) Promptly submit documentation within 20 business days if banks issue a temporary transactional exception."
        ),
        "tags": "safe,compliance,allowance,petty cash,discretionary,regulation,audit,bank,document,proof,limit,200k",
    },
    {
        "id": IMF_MACRO_KB_ID,
        "pairs": "USD/CNH,EUR/CNH,HKD/CNH,GBP/CNH",
        "domain": "IMF Macroeconomics & Purchasing Power Parity (PPP)",
        "title": "IMF CPI Spread Anchor: Long-Cycle PPP Gravity for Cross-Border FX",
        "macro_indicators": {
            "as_of_date": "2026-06",
            "inflation_cpi_yoy": {
                "US": 3.228,
                "Eurozone": 2.635,
                "UK": 3.203,
                "Hong Kong": 2.100,
                "China": 1.216,
            },
            "gdp_growth_forecast": {
                "US": 2.324,
                "Eurozone": 1.087,
                "UK": 0.797,
                "Hong Kong": 2.420,
                "China": 4.413,
            },
            "cofer_sovereign_reserves_share": {
                "USD": 56.77,
                "EUR": 20.25,
                "GBP": 4.41,
                "CNY": 1.95,
            },
        },
        "professional_definition": (
            "International Monetary Fund (IMF) World Economic Outlook and International Financial Statistics "
            "publish low-frequency structural inflation differentials (CPI) and GDP growth baselines that function "
            "as the gravitational anchor for long-horizon (90-day+) exchange rate determination under Purchasing "
            "Power Parity (PPP). When short-term spot rates materially deviate from the 30-day moving average (MA30), "
            "NOVA references the IMF inflation anchor to auto-correct long-term hedging ratios—preventing tactical "
            "over-reaction to noise while preserving conservative protection on 90-to-180-day receivable tails."
        ),
        "layman_translation": (
            "The inflation and GDP figures published by the IMF are 'slow-moving variables' that act like a long-term "
            "gravitational pull on exchange rates, keeping them from spinning out of control based on just a day or two of volatility. "
            "When the spot price diverges too far from the 30-day moving average (MA30), the system uses the IMF inflation anchor "
            "to recalibrate long-term forward hedge ratios. This prevents you from panic-buying or chasing short-term market hype, "
            "while securing a solid safety cushion for cross-border payouts on 90-to-180-day payment cycles."
        ),
        "actionable_playbook": (
            "1) For 90-to-180-day cross-border receivables, budget a conservative hedge sleeve using IMF-implied CPI spread "
            "vs your home jurisdiction (typically 60–75% of net exposure when spot > MA30 + Bollinger upper). "
            "2) Reconcile monthly: if realized spot diverges >2% from IMF PPP fair-value band, add 10% incremental forward "
            "notional rather than spot-panic conversion. "
            "3) Segment EU VAT-adjusted net EUR and USD marketplace gross flows separately before applying PPP correction. "
            "4) Document hedge rationale citing IMF CPI differential and MA30 deviation for bank compliance packs."
        ),
        "tags": "imf,ppp,inflation,cpi,macro,anchor,ma30,long-term,90-day,180-day,hedge,budget,receivable",
    },
{
        "id": "amazon_us_usd_payout_funnel",
        "pairs": "USD/CNH",
        "domain": "Amazon US Marketplace & USD Cash Flow Hedging",
        "title": "Amazon US USD Payout Deduction Funnel and Dynamic Margin Hedging Framework",
        "professional_definition": (
            "Hedging tactical USD exposure based on Gross Sales induces severe over-hedging due to multi-tiered Amazon marketplace deductions. "
            "The net settlement base must dynamically subtract category-specific referral fees (typically 15%), incremental FBA fulfillment/storage fees, "
            "and pay-with-balance Sponsored Products/Brands advertising expenses (typically 10%-20% of GMV) which serve as a natural internal offset. "
            "Temporal alignment must account for the 14-day bi-weekly payout rhythm or the Delivery Date Reserve (DD+7) compliance lag before triggering forward execution."
        ),
        "layman_translation": (
            "For Amazon US, never blindly lock in forward exchange rates based directly on front-end Gross Sales; "
            "doing so creates a speculative risk of 'over-hedging.' The system must first deduct the 15% standard referral fee, "
            "tiered FBA storage fees based on unit count and volume, and on-site ad spend that is taken directly out of your USD account balance "
            "(which usually eats up 10% to 20% of sales). These ad expenses are already naturally offset and consumed before payouts are even issued, "
            "so there is no need to double-hedge them. At the same time, you must build the 14-day bi-weekly payout cycle or the DD+7 "
            "(7 days deferred post-delivery) reserve account hold into the overall time horizon of your forward contracts."
        ),
        "actionable_playbook": (
            "1) Retrieve the real-time 'Total-Deduction-Ratio' via Amazon SP-API Financial Rates Service to establish the Net Payout Base. "
            "2) Issue a short-term 14-day USD/CNH forward contract locked strictly at 70% of the calculated net cash velocity. "
            "3) Dynamically adjust the hedging ratio to an 80% ceiling whenever macroeconomic indicators trigger a bearish USD signal."
        ),
        "data_sources": {
            "amazon_referral_fees": "https://sellercentral.amazon.com/help/hub/reference/G200336920",
            "amazon_fba_fees": "https://sellercentral.amazon.com/help/hub/reference/G201411300",
            "amazon_ads_billing": "https://advertising.amazon.com/help",
            "amazon_unavailable_balance": "https://sellercentral.amazon.com/help/hub/reference/G200185030"
        }
    },
    {
        "id": "amazon_eu_eur_gbp_vat_funnel",
        "pairs": "EUR/CNH",
        "domain": "Amazon EU Marketplace & Multi-Currency VAT Mitigation",
        "title": "Amazon Europe EUR Structural Payout Leakage and VAT Marketplace Facilitator Mitigation",
        "professional_definition": (
            "European marketplace revenue incurs immediate liquidity structural fragmentation due to the VAT Marketplace Facilitator legislation, "
            "where Amazon automatically deducts standard VAT (e.g., 20% for UK/France, 19% for Germany) at the point of sale. "
            "Furthermore, deploying the native Amazon Currency Converter (ACCS) introduces an aggressive 1.5% to 3.0% FX markup premium. "
            "Risk systems must route funds into localized multi-currency digital wallets to preserve native currency integrity, "
            "and calculate Forward Points using 21-to-28 day window forwards to offset severe Account Level Reserve bottlenecks."
        ),
        "layman_translation": (
            "When running an Amazon Europe store (Eurozone), remember that due to mandatory Marketplace Facilitator tax laws in the EU and UK, "
            "the platform automatically skims off VAT—like 20% in the UK and 19% in Germany—the exact moment a sale happens. The funds you see in your "
            "seller backend are already the 'net amount excluding tax,' so tax must be stripped from your hedging baseline. "
            "Furthermore, never use Amazon's official ACCS conversion system to repatriate funds, or you'll get hit with a steep 1.5% to 3% FX markup fee. "
            "You must link a localized, overseas multi-currency digital wallet. Also, given the rolling account reserve of 7 to 14 days required by European consumer "
            "protection laws, your payout cycle gets dragged out to 21-28 days. The system needs to accurately calculate forward points and use Window Forwards to lock rates."
        ),
        "actionable_playbook": (
            "1) Strip the respective European VAT layer instantly from the gross exposure framework prior to any ledger valuation. "
            "2) Enforce localized Euro bank routing via external payment providers to completely bypass Amazon's 3.0% ACCS friction. "
            "3) Deploy a 21-day tiered EUR/CNH Window Forward contract to match the deferred transaction latency caused by the European Consumer Protection Act."
        ),
        "data_sources": {
            "amazon_accs_terms": "https://sellercentral.amazon.com/help/hub/reference/G200381210",
            "eu_commission_vat": "https://vat-one-stop-shop.ec.europa.eu/index_en",
            "uk_gov_vat_rules": "https://www.gov.uk/guidance/vat-and-overseas-sellers-using-online-marketplaces-to-sell-goods-in-the-uk",
            "amazon_eu_reserve": "https://sellercentral.amazon.co.uk/help/hub/reference/G200185030"
        }
    },
    {
        "id": "shopify_independent_gateway_funnel",
        "pairs": "USD/CNH,EUR/CNH",
        "domain": "Shopify Independent Stores & Payment Gateway Risk Analytics",
        "title": "Shopify Independent Store Gateway Friction, Rolling Reserve Disruption and Cross-Currency Mitigation",
        "professional_definition": (
            "Unlike centralized marketplaces, Shopify stores exhibit multi-gateway cash flow scattering governed by Stripe and PayPal protocols. "
            "Stripe imposes a T+2 to T+7 rolling settlement baseline alongside a 1.0% to 2.0% multi-currency FX markup for non-USD conversions. "
            "Crucially, high-growth accounts face aggressive PayPal Rolling Reserves, where 10% of gross revenue is locked for a 90-day duration. "
            "This splits the asset pool into highly mismatched temporal horizons, rendering standard spot hedging ineffective and demanding structured forward matching."
        ),
        "layman_translation": (
            "Shopify independent stores don't have a unified payout date; it depends entirely on gateways like Stripe and PayPal. "
            "While Stripe offers quick T+2 to T+7 settlements, letting it automatically convert customer payments in EUR or CAD into USD in the backend "
            "will cost you a hidden 1% to 2% FX markup. The ultimate black swan for independent stores is PayPal's 'Rolling Reserve' system: "
            "if your sales skyrocket, PayPal will often force-lock 10% of every order's revenue for 90 days. This means 10% of your capital gets its payout timeline "
            "stretched out by 3 months, during which exchange rate volatility can severely eat away at your net margins. The system must slice these store assets "
            "by their time horizons and hedge them separately."
        ),
        "actionable_playbook": (
            "1) Parse the webhooks from payment gateways to capture real-time 'Reserve_Ratio' alterations and isolate locked capital. "
            "2) Enforce 'Natural Hedging' by utilizing free USD balances directly for Meta/Google advertising expenses, neutralizing exchange friction. "
            "3) Map the 10% locked PayPal reserve to a dedicated 90-day long-tenor USD/CNH or EUR/CNH forward contract to freeze cross-period volatility."
        )
    },
    {
        "id": "hongkong_offshore_lers_cips_funnel",
        "pairs": "HKD/CNH,USD/HKD",
        "domain": "Hong Kong Offshore Banking & Corporate Capital Aggregation",
        "title": "Hong Kong Offshore Corporate Capital Aggregation, LERS Peg Boundary Arbitrage and CIPS Clean Settlement Protocols",
        "professional_definition": (
            "Hong Kong offshore legal entities utilize centralized banking infrastructures (e.g., HSBC, BOC HK) to aggregate global USD and EUR receivables. "
            "Under the Linked Exchange Rate System (LERS), USD/HKD trades within a strict pegged band of 7.75 to 7.85, introducing localized range-bound anomalies "
            "when Libor/Hibor spreads widen. However, the final cross-border leg into Mainland China (HKD/CNH) is fully floating. "
            "Sellers should leverage CIPS (Cross-border Interbank Payment System) under SAFE Circular [2020] No.11 to convert offshore FX into CNH directly, "
            "ensuring seamless 'Three-Way Matching' compliant customs clearance."
        ),
        "layman_translation": (
            "Cross-border e-commerce sellers commonly set up offshore companies in Hong Kong (using accounts like HSBC or WorldFirst) "
            "as a centralized hub to pool global USD and EUR revenues. Even though HKD operates under the 'Linked Exchange Rate System' in Hong Kong—meaning "
            "it is tightly pegged to the USD between 7.75 and 7.85 with extremely low risk—the story changes when you finally convert HKD into offshore RMB (CNH) "
            "to bring it back home. That leg is fully floating and highly volatile. To bring funds back legally and transparently, businesses should "
            "follow SAFE Circular [2020] No.11 to directly convert offshore funds into CNH on the Hong Kong side. Then, route it back to the mainland parent company "
            "as a trade payment via the CIPS system, achieving automated tax-exempt reconciliation based on the 'Three-Way Matching' of your store's backend data."
        ),
        "actionable_playbook": (
            "1) Monitor the USD/HKD parity limits; structure automated buy orders if the cross-rate tests the 7.85 weak-side boundary. "
            "2) Maximize CIPS utilization by routing offshore corporate cash flows directly as CNH to eliminate secondary local currency conversion markups. "
            "3) Execute a 7-day rolling FX Swap contract on HKD/CNH to align sudden offshore liquidity drawdowns with onshore supply chain payroll deadlines."
        )
    },
    {
        "id": "cross_border_payroll_window_netting",
        "pairs": "HKD/CNH",
        "domain": "Cross-Border Payroll & Supply Chain Liability Matching",
        "title": "Window Forward Netting: Mitigating Rigid Payroll Deadlines Under HKD Weakness",
        "professional_definition": (
            "SMEs with offshore entities (e.g., Hong Kong retail/parent companies) collecting HKD revenue face "
            "rigid CNH liabilities (Shenzhen/Dongguan factory payroll, rent, supply chain costs) on fixed monthly dates. "
            "To resolve the vulnerability of forced immediate spot conversion during adverse rate movements, "
            "systems must deploy Window Forward Netting combining Spot-to-MA5 indicators with multi-currency overdraft facilities."
        ),
        "layman_translation": (
            "While Hong Kong physical stores or parent companies earn HKD, their R&D plants and supply chain factories in Shenzhen or Dongguan "
            "have fixed, rigid RMB expenses every month (such as payroll, raw materials, and factory rent). Every payday is a hard deadline; "
            "no matter how terrible the exchange rate is that day, business owners are forced to make large spot conversions. If the HKD happens to experience "
            "a sharp, short-term drop against the RMB, it creates massive exchange rate losses. The system uses APIs to dynamically track the Spot, MA5, and RSI trends "
            "of HKD/CNH. During windows of HKD strength (e.g., when Spot > MA5), the AI prompts business owners to convert funds early in batches to lock in the total "
            "RMB payroll for the next 1 to 3 months. When the rate is awful, you can temporarily draw down on a Hong Kong local bank HKD short-term credit line, "
            "waiting for a technical rebound before final settlement."
        ),
        "actionable_playbook": (
            "1) Connect internal HR/ERP systems via API to feed exact monthly CNH payroll and vendor liability schedules. "
            "2) Monitor tech indicators: If Spot price sits above MA5 and RSI is neutral, trigger layered forward conversions for a 90-day window. "
            "3) If Spot crashes below MA5, freeze immediate settlement and draw down on temporary HKD revolving credit lines to defer settlement. "
            "4) Pre-compile labor contracts, tax/social security rolls, and trade invoices to clear domestic bank Anti-Money Laundering (AML) controls within 48 hours."
        ),
        "tags": "payroll,hongkong,factory,supply chain,mismatch,deadline,window forward,credit line,ma5,rsi,aml,compliance,hkd,cnh",
    },
    {
        "id": "government_backed_liquidity_enhancement",
        "pairs": "USD/CNH,EUR/CNH",
        "domain": "SME Liquidity Preservation via Government-Backed Credit Enhancement",
        "title": "Government-Guaranteed Hedging: Eliminating Margin Barriers for Liquidity-Strapped SMEs",
        "professional_definition": (
            "SMEs often operate completely unhedged ('naked exposure') due to an inability to meet commercial banks' "
            "stringent derivative margin demands (typically 5%-10% cash freeze), which severely locks up scarce operational liquidity. "
            "Deploying Government-Backed Financing Guarantee Schemes allows credit-strapped enterprises "
            "to execute essential forward or option locks with zero upfront cash margin requirements."
        ),
        "layman_translation": (
            "Many businesses want to lock in rates because of high market volatility, but they cannot afford the forex derivative margins required by banks "
            "(traditional risk controls typically freeze 5% to 10% cash upfront) or they worry it will tie up precious operational liquidity. Since micro and small "
            "enterprises generally have lower credit ratings, they are locked out of risk hedging, forced to 'run naked' against exchange rate fluctuations. "
            "Alternatively, when RMB cash gets tight, they are forced to sell forex at terrible rates, falling into a financial dilemma of 'selling cheap' or 'forced default.' "
            "The solution is to bring in local government-backed financing guarantee agencies to boost credit for small businesses. This allows companies to execute "
            "FX hedging without paying upfront cash margins or eating into their pure credit lines."
        ),
        "actionable_playbook": (
            "1) Apply for inclusion in the local commerce bureau or SAFE-approved enterprise whitelist for government-backed FX guarantees. "
            "2) Map operational cash-flow deficits to ensure hedging facilities are used strictly for real trade backgrounds rather than speculation. "
            "3) Execute standard Forward Contracts or Options with collaborating commercial banks under the 0-margin framework. "
            "4) In the event of market movement against the forward contract, verify that the government guarantee platform seamlessly offsets variance calls until final physical settlement."
        ),
        "tags": "government,guarantee,margin,liquidity,cash flow,sme,credit enhancement,zero-margin,forward,safe,whitelist,cnh",
    },
    {
        "id": "cross_border_payment_cost_optimization",
        "pairs": "USD/GBP,EUR/GBP,USD/CNH",
        "domain": "Cross-Border Payment Friction Minimization & Multi-Currency Netting",
        "title": "Multi-Currency Accounts and Natural Hedging: Eliminating Hidden Cost and SWIFT Leakages",
        "professional_definition": (
            "SMEs encounter massive hidden costs (3-5% retail bank spreads and intermediary SWIFT fees) during international payments. "
            "By setting up localized multi-currency accounts (e.g., World Account) to accept and hold foreign currencies directly, "
            "businesses can match receivables with upcoming payable obligations in the same denomination, completely optimizing international cost flows."
        ),
        "layman_translation": (
            "Many UK and global SMEs suffer heavy average losses from exchange rate volatility, with their profits swallowed by banks' non-transparent 'hidden costs.' "
            "Traditional banks usually charge a 3% to 5% spread, and the SWIFT system skims off extra fees via intermediary banks. The solution is to set up a multi-currency "
            "account (like World Account) to collect and hold foreign currencies. You can then directly use the USD or other foreign currencies you receive to pay future foreign currency "
            "bills (such as overseas supplier invoices). This avoids the double-conversion friction of constant exchange and achieves a natural hedge. However, you must stay alert to "
            "the risk of cash flow disruption caused by strict AML (Anti-Money Laundering) checks, as compliance reviews can cause funds to be delayed in transit or frozen for several days."
        ),
        "actionable_playbook": (
            "1) Replace traditional local bank payout details with digital multi-currency account receiving coordinates for international buyers. "
            "2) Retain foreign currency earnings inside the offshore multi-currency wallets to meet future raw material or digital platform expenses. "
            "3) Map international supplier payment due dates to maximize the natural netting timeline before forced conversion. "
            "4) Maintain clear, automated electronic trade logs and shipping bills to instantly clear strict automated Anti-Money Laundering (AML) triggers."
        ),
        "tags": "hidden cost,spread,swift,world account,natural hedge,multi-currency,frictional cost,repatriation,aml,compliance,gbp,usd",
    },
    {
        "id": "traditional_export_exposure_option_shield",
        "pairs": "USD/CNH",
        "domain": "Low-Margin Export Enterprise Exposure Management",
        "title": "Short-Term Currency Options: Erecting Breakwaters for Rapid Export Volume Growth",
        "professional_definition": (
            "As SME export volumes scale rapidly across diverse emerging and standard markets, their unhedged net foreign exchange "
            "exposure expands proportionally. In the absence of an internal structured risk management mechanism, corporate net profits "
            "become highly vulnerable to spot market swings. Deploying short-term (e.g., 30-day) vanilla FX options provides a precise "
            "risk breakwater to guarantee baseline product pricing."
        ),
        "layman_translation": (
            "As an enterprise's export volumes grow rapidly across markets like Mexico, Vietnam, and Italy, its foreign exchange risk exposure "
            "creeps up alongside it. However, because many SMEs lack hedging tools and haven't built a proper exchange rate risk management framework, "
            "their export collections are left completely naked to exchange rate swings, leaving their hard-earned profits easily devoured by one-sided market trends. "
            "The most direct and standard solution is to have commercial banks, under the guidance of SAFE, tailor short-term (such as 30-day) RMB exchange rate "
            "options for small-scale exporters. This builds a rock-solid 'breakwater' against exchange rate volatility to safeguard your USD and other foreign currency revenues."
        ),
        "actionable_playbook": (
            "1) Consolidate the forward sales pipeline and define the accurate short-term FX net exposure coming from overseas buyers. "
            "2) Contract with local specialized banks to purchase structured 30-day FX options aligned with shipping dates. "
            "3) Utilize the option structure to guarantee a definitive floor price for export revenue repatriation while maintaining upside potential. "
            "4) Systematically log derivative contract parameters into financial statements for corporate risk governance audits."
        ),
        "tags": "export,exposure,option,breakwater,short-term,shield,vanilla option,safe,revenue,protection,usd,cnh",
    },
    {
        "id": "micro_merchant_one_yuan_option_shield",
        "pairs": "USD/CNH",
        "domain": "Micro-Exporter Non-Linear Hedging & Low-Cost Option Structures",
        "title": "One-Yuan Option Program: Low-Cost Floor Protection for Irregular Micro Cash Flows",
        "professional_definition": (
            "Micro-exporters and individual merchants struggle with highly irregular payment reconciliation timelines and lack the capacity "
            "for active portfolio tracking. Traditional currency options carry heavy upfront premium costs that wipe out small margins. "
            "Utilizing localized Micro-Premium Put Options establishes a non-linear protection mechanism—guaranteeing "
            "a minimum floor rate while preserving unlimited upside with negligible sunk cost."
        ),
        "layman_translation": (
            "Individual merchants and micro-exporters deal with unpredictable payout timings and non-stop exchange rate changes, making it "
            "impossible to guess what the rate will be in three months. Their underlying need is a flexible hedging method that doesn't require "
            "staring at market charts every day or getting tightly locked into rigid hedging instruments. Traditional forward settlements lock you in; "
            "if the RMB depreciates significantly, you miss out on better rates. The industry's innovative fix is the bank-backed 'One-Yuan Option' program: "
            "merchants pay just 1 RMB in premium to buy a USD put option. This structure guarantees a floor price if the market crashes, yet leaves the upside uncapped "
            "(you can drop the option if spot rates are better), completely eliminating the heavy upfront premium costs of traditional derivatives."
        ),
        "actionable_playbook": (
            "1) Log into the digital banking portal or Micro-Merchant Agent interface to check current institutional promotional option pools. "
            "2) For incoming unstructured foreign payments (e.g., USD 10,000 to USD 50,000 tranches), purchase the 30-day 'One-Yuan' Put Option. "
            "3) If USD depreciates at contract maturity, execute automated exercise at the locked floor price to guarantee baseline profit. "
            "4) If USD appreciates, allow the 1-Yuan option to automatically expire and immediately settle via online spot FX to capture maximum upside market gains."
        ),
        "tags": "micro-merchant,individual,option,put option,one-yuan,premium,flexible,floor price,upside,low-cost,usd,cnh",
    },
    {
        "id": "industry_furniture_usd_mismatch_trap",
        "pairs": "USD/CNH",
        "domain": "Vertical Industry - Heavy Furniture & Long-cycle Supply Chain",
        "title": "Macro Currency Mismatch, Cash Conversion Elongation and Extension Risks in Heavy Furniture Industry",
        "professional_definition": (
            "Heavy furniture cross-border commerce experiences extreme cash flow elongation governed by ocean freight lag "
            "and prolonged 3PL/FBA warehousing cycles, stretching the Cash Conversion Cycle (CCC) to 90-120 days. "
            "Executing Fixed-Date Forward contracts based on predictive ERP gross receivables introduces catastrophic cash flow misalignments. "
            "If customs inspections or maritime delays defer USD entry, sellers face liquidity exhaustion at mature delivery dates without physical currency. "
            "Risk matrices must strictly ban Fixed Forwards and mandate stochastic Window Forwards matching real-time inventory velocity."
        ),
        "layman_translation": (
            "Shipping heavy furniture like mattresses and sofas is a textbook high-ticket, slow-turnover, long-cycle business. "
            "From paying domestic factory suppliers and managing first-leg ocean freight to warehousing overseas and getting final payouts, "
            "the entire capital tie-up cycle stretches out to 90 or 120 days. During this 3-to-4-month exposure window, if a seller uses a "
            "'Fixed-Date Forward' contract to lock in USD based on estimated arrival dates, they walk straight into a trap. If they hit customs audits, "
            "shipping delays, or peak-season overseas warehouse congestion, their USD arrival gets deferred—leaving them with zero physical USD to deliver "
            "when the forward contract matures. Being forced to execute a rollover or buy spot USD to fulfill the contract creates unpredictable secondary "
            "exchange losses and triggers liquidity exhaustion. For heavy furniture, fixed forwards are strictly banned; you must use Window Forwards."
        ),
        "actionable_playbook": (
            "1) Prohibit Fixed Forwards entirely; mandate 'Window Forwards' (择期远期) to allow a flexible 30-day delivery buffering window. "
            "2) Map 3PL inventory clearance velocity directly to the treasury dashboard to dynamically adjust or pare down the target hedging size. "
            "3) Maintain a 15% liquid buffer pool in CNH local credit lines to absorb potential regulatory scrutiny or margin calls without operational disruption."
        ),
        "data_sources": {
            "cbp_antidumping_rules": "https://www.cbp.gov/",
            "safe_capital_settlement_rules": "http://www.safe.gov.cn/"
        }
    },
    {
        "id": "industry_apparel_eur_high_return_trap",
        "pairs": "EUR/CNH",
        "domain": "Vertical Industry - Fast Fashion Apparel & High Return Rate Dilution",
        "title": "High Return Rate Return-Logistics Erosion and Phantom Exposure Traps in Fast-Fashion E-Commerce",
        "professional_definition": (
            "Fast-fashion apparel e-commerce in European zones encounters massive phantom exposure due to localized structural return rates (30%-50%). "
            "Hedging based on gross checkout data creates an artificial forward long position because reverse logistics, refunds, and return shipping "
            "destabilize net EUR inflows post-facto. Financial layers must deploy a trailing historical decay multiplier to recalibrate the effective hedge ratio."
        ),
        "layman_translation": (
            "The most lethal pain point for fast-fashion e-commerce (like women's apparel and trendy clothing) in Europe is the brutally high return rate, "
            "which typically runs between 30% and 50%. European shoppers are backed by incredibly strict consumer protection laws and are used to 'buying more, "
            "returning most.' If your risk framework blindly locks in EUR forward contracts based on front-end checkout sales numbers, you get caught in a "
            "'phantom exposure' trap because those returns turn into customer refunds over the next 14 to 30 days, causing your actual EUR cash-in to heavily shrink. "
            "As a result, the seller ends up holding an excessive EUR forward long position. When returns spike, you are forced into 'over-delivery' in the derivative "
            "market, flipping safe risk hedging into unhedged currency speculation. Apparel brands must inject historical return rates as a dynamic decay factor into their model."
        ),
        "actionable_playbook": (
            "1) Enforce a historical decay coefficient (typically 0.55x-0.70x) onto gross EUR checkout metrics prior to derivative assignment. "
            "2) Conduct weekly re-balancing on all outstanding EUR/CNH forward contracts to align with active reverse-logistics webhook triggers. "
            "3) Direct the unhedged phantom residue currency into localized EUR wallets to act as a natural cash reserve for future supplier settlement."
        ),
        "data_sources": {
            "eu_consumer_rights_directive": "https://eur-lex.europa.eu/",
            "ecl_academic_research": "https://pdf.hanspub.org/ecl_2316207.pdf"
        }
    },
    {
        "id": "amazon_uk_gbp_vat_funnel",
        "pairs": "GBP/CNH",
        "domain": "Amazon UK Marketplace & GBP Cash Flow Hedging",
        "title": "Amazon UK GBP Payout Structural Leakage, HMRC VAT Facilitator and Forward Alignment",
        "professional_definition": (
            "UK marketplace revenue experiences intensive financial friction due to HMRC VAT Marketplace Facilitator legislation, "
            "where Amazon automatically exfiltrates the 20% standard UK VAT at the point of electronic sale. "
            "Hedging tactical GBP/CNH exposure based on raw sales data introduces severe over-hedging artifacts. "
            "Furthermore, treasury structures must map the 14-day bi-weekly settlement velocity and the Delivery Date Reserve (DD+7) latency "
            "to window derivatives, offsetting macro-driven Sterling intraday gaps generated by Bank of England interest rate resolutions."
        ),
        "layman_translation": (
            "When running an Amazon UK store, the platform automatically skims off the 20% standard UK VAT right at the moment a sale happens "
            "due to mandatory Marketplace Facilitator tax laws enforced by HMRC. Therefore, enterprises must never blindly lock in forward exchange rates "
            "based directly on front-end gross GBP sales numbers; you must completely strip this 20% tax buffer from your baseline first, "
            "or you will run into severe 'over-hedging' bad debts. At the same time, your GBP payouts face a 14-day bi-weekly disbursement cycle "
            "plus a delivery-date reserve hold (DD+7). During periods of macro volatility—such as when the Bank of England (BoE) announces interest rate "
            "decisions—the system must automatically deploy Window Forwards to precisely align with this 14-to-21 day payout delay on your net GBP exposure."
        ),
        "actionable_playbook": (
            "1) Strip the 20% HMRC VAT layer instantly from gross British Pound metrics prior to calculating any hedging ledger valuation. "
            "2) Map the 14-day Amazon UK billing cycle webhook directly to the treasury desk to define the net executable cash velocity. "
            "3) Execute short-term 14-to-21 day GBP/CNH Window Forwards to immunize the net proceeds against Bank of England (BoE) interest rate volatility."
        ),
        "data_sources": {
            "hmrc_uk_vat_rules": "https://www.gov.uk/guidance/vat-and-overseas-sellers-using-online-marketplaces-to-sell-goods-in-the-uk",
            "amazon_uk_deferred_payments": "https://sellercentral.amazon.co.uk/help/hub/reference/G200185030"
        }
    },
    {
        "id": "derivative_zero_cost_collar_strategy",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH",
        "domain": "FX Derivatives & Treasury Risk Management",
        "title": "Zero-Cost Collar Option Structure Selection and Premium Neutralization Logic",
        "professional_definition": (
            "A Zero-Cost Collar involves the simultaneous acquisition of an out-of-the-money (OTM) FX Put Option "
            "and the writing of an OTM FX Call Option, calibrated such that the premium payable and premium receivable "
            "achieve exact algebraic neutralization (Net Premium = 0). The operational framework restricts the spot conversion "
            "within a strict mathematical band [Floor Strike, Cap Strike], shielding the enterprise from downside breakeven breach "
            "while truncating asymmetric upside volatility, requiring strict temporal alignment with cash flow settlement horizons."
        ),
        "layman_translation": (
            "A zero-cost collar (or range forward structure) is an advanced hedging tool for trade enterprises that requires zero upfront premium costs "
            "because the price to buy the put option is exactly offset by the premium gained from selling the call option. It builds a 'safety corridor' "
            "for your business: the floor is your breakeven lifeline—ensuring that even if exchange rates plummet through the floor, you can still settle "
            "your forex at this guaranteed baseline rate; the cap is your profit ceiling surrendered to the bank—meaning if exchange rates rocket past the cap, "
            "你 must passively settle your forex at that capped price. When recommending this plan, the system must ensure the option's maturity date "
            "perfectly matches the actual payment terms of your Amazon store or trade orders, avoiding liquidity defaults where you 'have no physical currency to deliver' upon contract maturity."
        ),
        "actionable_playbook": (
            "1) Map the Floor Strike directly to the enterprise's cost-accounting breakeven FX rate to enforce a hard margin floor. "
            "2) Solve for the optimal Cap Strike using real-time implied volatility matrices to maximize the upside revenue ceiling under zero-premium constraints. "
            "3) Synchronize the option valuation date precisely with the forecasted electronic marketplace payout or trade bill collection counterparty ledger."
        ),
        "data_sources": {
            "safe_china_fx_neutrality": "http://www.safe.gov.cn/",
            "isda_fx_derivatives_standards": "https://www.isda.org/"
        }
    },
    {
        "id": "derivative_fx_swap_liquidity_misalignment",
        "pairs": "GBP/USD, USD/CNH, GBP/CNH",
        "domain": "FX Derivatives & Multicurrency Cash Optimization",
        "title": "FX Swap Engineering: Resolving Cross-Border Liquidity Misalignment via Covered Interest Parity",
        "professional_definition": (
            "An FX Swap combines a simultaneous spot foreign exchange transaction and an opposing forward foreign exchange contract "
            "with identical notionals but opposite directions. The pricing differential is deterministically bound by Swap Points "
            "derived from Covered Interest Parity (CIP) using the interest rate differential between the respective central banks "
            "(e.g., BoE vs. Fed). This financial mechanism eradicates dual-layered conversion bid-ask spreads, allowing automated "
            "intercompany currency relocation while maintaining the structural long/short composition of the baseline treasury portfolio."
        ),
        "layman_translation": (
            "An FX Swap is a cross-currency financial bridge used to solve the dilemma where 'a company is not short on cash overall, but is extremely dry on "
            "a specific currency locally.' For instance, say you have pooled a large amount of GBP in your UK store, but your US store desperately needs USD "
            "to pay for supplier freight. By using a swap tool, the system lets you temporarily exchange your GBP with a bank today to take and use the USD, "
            "while simultaneously agreeing to swap back that exact amount of GBP using USD 30 days later. This entire process completely avoids the double-conversion "
            "bid-ask spread losses of turning 'GBP into CNY, and then CNY into USD.' When launching this feature, the system needs to calculate the 'Swap Point' costs "
            "strictly based on the interest rate differentials between the two nations, and monitor the arrival velocity of your forward USD payouts to prevent delivery failures at maturity."
        ),
        "actionable_playbook": (
            "1) Isolate localized currency shortfalls (e.g., immediate USD freight bills) against idle cash pockets (e.g., mature Amazon UK GBP reserves). "
            "2) Calculate theoretical Swap Points using real-time SOFR and SONIA interbank curves to evaluate the pricing fairness offered by liquidity providers. "
            "3) Calibrate the forward leg maturity date to perfectly align with the secondary marketplace's historical median payout distribution velocity."
        ),
        "data_sources": {
            "bis_fx_survey_metrics": "https://www.bis.org/publ/rpfx22.htm",
            "federal_reserve_swap_framework": "https://www.federalreserve.gov/monetarypolicy/bst_liquidityswaps.htm"
        }
    },
    {
        "id": "derivative_forward_rollover_early_delivery",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH, HKD/CNH",
        "domain": "FX Derivatives & Flexible Treasury Operations",
        "title": "Tactical FX Forward Adjustments: Managing Supply Chain Disruptions via Roll-overs and Early Deliveries",
        "professional_definition": (
            "FX Forward Roll-overs and Early Deliveries act as active structural compliance shock-absorbers "
            "when physical cash conversion velocity diverges from historical ERP parameters due to maritime bottlenecks or customs detentions. "
            "A Roll-over simultaneously closes out the maturing forward commitment at the current spot rate and opens a new forward contract "
            "maturing at a deferred date, pricing the differential purely through rolling Swap Points. Conversely, Early Delivery "
            "permits immediate drawdown of capital when marketplace cash velocity accelerates, preventing operational deadweight loss."
        ),
        "layman_translation": (
            "Roll-overs and Early Deliveries are flexible time-adjustment tools used to prevent 'hard defaults' on your forward exchange contracts. "
            "When overseas customs inspections, warehouse backlogs, or shipping delays mean your USD or GBP cannot arrive on time, the system automatically "
            "triggers a roll-over, helping you apply to the bank for an extended delivery window to avoid forced liquidations. Conversely, when sales boom "
            "and payouts hit your account ahead of schedule, the system supports early delivery so you can pull back your RMB cash sooner. The entire process "
            "only adjusts for minor swap point variations, heavily safeguarding the cash flow chain of your cross-border supply chain."
        ),
        "actionable_playbook": (
            "1) Interface directly with 3PL supply-chain webhooks to instantly trigger a Roll-over assessment if maritime transit delay exceeds 5 business days. "
            "2) Calculate the net present value (NPV) impact of forward points prior to confirming an early delivery to avoid structural alpha erosion. "
            "3) Dynamically adjust the operational safety window inside the treasury ledger, limiting contract maturities to split tranches to avoid concentrated rollover default risk."
        ),
        "data_sources": {
            "cfets_derivative_rules": "http://www.cfets.com.cn/",
            "safe_logistics_flexibility_notice": "http://www.safe.gov.cn/"
        }
    },
    {
        "id": "macro_fed_fomc_nfp_volatility_shield",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH",
        "domain": "Macroeconomics & Central Bank Policy Transmission",
        "title": "Central Bank Event Risk Mitigation: Algorithmic Hedging Under FOMC and NFP Volatility Shocks",
        "professional_definition": (
            "Macroeconomic event risks, institutionalized by the Federal Open Market Committee (FOMC) rate decisions and "
            "Non-Farm Payroll (NFP) releases, induce non-linear intraday volatility spikes and aggressive bid-ask spread widening. "
            "The system triggers a defensive macro-shocker protocol, shifting the foundational hedge ratio based on the "
            "implied volatility term structure and forward-guidance semantic delta. This algorithmic overlay inoculates "
            "the multi-currency spot exposure from fatal market gaps and institutional slippage during structural DXY regime shifts."
        ),
        "layman_translation": (
            "The nights when the Federal Reserve announces interest rate decisions (FOMC), and when US Non-Farm Payroll (NFP) or CPI data "
            "are released, are known as 'tsunami nodes' in the forex market. The exact second the data hits the public, the Dollar Index "
            "and offshore RMB usually experience stormy, massive price gaps of hundreds of pips, and bank spreads instantly blow wide open. "
            "Trying to execute spot settlements at this exact moment will expose you to severe losses due to slippage. "
            "When monitoring these heavy-hitting macro event windows, the system must either increase the forward hedge ratio ahead of time "
            "or use option tools to build a 'breakeven shield.' You must never 'run naked' on the night of data releases, effectively "
            "dodging the risk of a one-sided market crash triggered by a Federal Reserve policy pivot."
        ),
        "actionable_playbook": (
            "1) Parse the rolling Economic Calendar via API to systematically lock down treasury transactional capabilities 30 minutes pre- and post-FOMC announcements. "
            "2) Dynamically scaling up the baseline USD/CNH forward hedging target ratio by an incremental 15% if the Fed Dot Plot telegraphs a hawkish lower-bound shift. "
            "3) Deploy OTM European Put Options as a structural shield 48 hours prior to Non-Farm Payroll releases to institutionalize a deterministic margin floor against negative macro surprises."
        ),
        "data_sources": {
            "federal_reserve_fomc_calendar": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "pboc_macro_risk_guidelines": "http://www.pbc.gov.cn/"
        }
    },
    {
        "id": "macro_ecb_boe_joint_divergence_shield",
        "pairs": "EUR/CNH, GBP/CNH",
        "domain": "Macroeconomics & Cross-Currency Divergence Models",
        "title": "European Macroeconomic Transmission: Mitigating EUR/GBP Scissor Arbitrage and Co-Integrated Event Risks",
        "professional_definition": (
            "Cross-currency divergence risks between the European Central Bank (ECB) and the Bank of England (BoE) "
            "induce dynamic correlation decay inside localized multi-currency treasury portfolios. Under asynchronous "
            "monetary policy cycles—such as asymmetric rate cuts or divergent quantitative tightening (QT) velocities—"
            "the mathematical co-integration between EUR/CNH and GBP/CNH undergoes rapid short-term breakdown. "
            "The system deploys an independent multi-asset matrix that isolates local CPI/HICP signals, dynamically "
            "decoupling Euro and Sterling directional parameters to safeguard against severe net margin dilution."
        ),
        "layman_translation": (
            "Monetary policy decisions from the European Central Bank (ECB) and the Bank of England (BoE) often look identical on the surface "
            "but are completely misaligned in reality, walking straight into a 'double scissor-spread trap.' When cross-border sellers collect both EUR and GBP, "
            "blindly assuming that 'Euro and Sterling move as one' and using a unified hedging ratio will backfire. If the ECB is forced to cut interest rates "
            "due to a sluggish European economy (weakening the Euro) while the BoE is forced to keep rates high to fight sticky inflation (strengthening Sterling), "
            "your risk exposure valuation will instantly distort. This leaves you with severe over-hedged bad debts in the GBP zone or entirely unhedged 'naked' exposure "
            "in the EUR zone. During overlapping central bank meeting windows, the system must activate asymmetric risk multipliers to lock down EUR/CNH and GBP/CNH "
            "forward exposures independently, flattening the chain-reaction shocks of European macro events."
        ),
        "actionable_playbook": (
            "1) Execute a structural correlation test on the EUR/GBP cross-rate 72 hours prior to any joint ECB/BoE monetary policy announcement week. "
            "2) Automatically decouple any unified 'European Cluster' hedging proxies; enforce separate, asset-specific EUR/CNH and GBP/CNH forward contracts. "
            "3) Scale down immediate spot execution weights and shift 25% of unhedged European receivables into rolling OTM Collar options to form a soft protection buffer against central bank verbal guidance surprises."
        ),
        "data_sources": {
            "ecb_monetary_policy_hub": "https://www.ecb.europa.eu/press/govcdec/html/index.en.html",
            "boe_mpc_decision_hub": "https://www.bankofengland.co.uk/monetary-policy"
        }
    },
    {
        "id": "macro_pboc_counter_cyclical_factor_shield",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH, HKD/CNH",
        "domain": "Macroeconomics & Sovereign Interventions Framework",
        "title": "Sovereign Intervention Protocols: Managing CNH Liquidity Squeezes and Counter-Cyclical Factor Deviations",
        "professional_definition": (
            "Offshore Renminbi (CNH) risk matrices encounter unique macro shocks driven by People's Bank of China (PBOC) "
            "macro-prudential policy implementations. When directional devaluation momentum intensifies, the central bank "
            "activates the Counter-Cyclical Factor within the daily USD/CNY fixing equation and dynamically adjusts "
            "the foreign exchange risk reserve ratio. These interventions intentionally induce non-linear deviations "
            "between the official fixing baseline and market spot rates, severely compressing offshore short liquidity "
            "and distorting CIP-derived forward swap points, necessitating adaptive correlation bounds inside the treasury layer."
        ),
        "layman_translation": (
            "When the offshore Renminbi (CNH) experiences extreme, one-sided volatility, it often triggers policy interventions by the People's Bank of China (PBOC) "
            "—such as deploying the counter-cyclical factor, raising forex risk reserve ratios, or issuing central bank bills offshore. "
            "When the system detects that the daily RMB midpoint fixing is significantly stronger than the previous day's market close (meaning the counter-cyclical factor "
            "is visibly at work), or if the offshore RMB borrowing rate (CNH Hibor) spikes abnormally, it indicates the 'national team' is orchestrating a 'short squeeze.' "
            "In the short term, the market is highly prone to massive, policy-driven sharp reversals spanning hundreds of pips. "
            "At this point, the system must force a reduction in the weight of trend-following hedges and restrict blindly entering into long-term forward locks at high levels, "
            "preventing you from falling into a trap of severe floating losses ignited by a sudden policy U-turn."
        ),
        "actionable_playbook": (
            "1) Parse daily CFETS mid-rate fixings at 09:15 Beijing Time via API; compute the mathematical delta against previous market closing spot."
            "2) Enforce an immediate 'Hedging Velocity Ceiling' if the fixing deviation exceeds a 300 pips regulatory policy premium."
            "3) Interrogate real-time CNH Hibor spikes prior to quoting FX Swaps, automatically re-pricing forward points to mirror the localized liquidity drawdowns enforced by sovereign entities."
        ),
        "data_sources": {
            "cfets_mid_rate_fixings": "http://www.cfets.com.cn/",
            "pboc_macro_prudential_hub": "http://www.pbc.gov.cn/"
        }
    },
    {
        "id": "macro_geopolitical_tail_risk_freight_shield",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH",
        "domain": "Macroeconomics & Supply Chain Financial Co-Integration",
        "title": "Geopolitical Tail Risk Engineering: Cross-Asset Hedging Under SCFI Freight Volatility and GPR Shocks",
        "professional_definition": (
            "Geopolitical tail risks trigger non-linear cross-asset co-movements where systemic currency volatility "
            "co-integrates with hyper-inflationary freight cost impulses (e.g., abrupt SCFI/FBX index spikes). "
            "Under conditions of maritime chokepoint blockades or structural tariff regimes, the traditional cash flow "
            "velocity parameters undergo immediate deceleration, while spot operational currency demands for immediate "
            "logistics liquidation scale exponentially. The system shifts from basic static delta hedging to dynamic "
            "liquidity-buffered stress structures to insulate corporate net asset valuations from binary macroeconomic shocks."
        ),
        "layman_translation": (
            "Geopolitical conflicts (such as the Red Sea crisis or escalating trade wars) often trigger a dual tail-risk stranglehold "
            "of exchange rate shocks combined with skyrocketing freight costs. When ocean freight rates (the SCFI index) multiply "
            "due to supply chain paralysis, an enterprise's profit margins are squeezed to the absolute limit, and they are simultaneously "
            "forced to pay massive amounts of spot USD or GBP cash to logistics providers. During such times, the system must never "
            "blindly put your total overseas sales into long-term forward exchange locks; it must forcefully retain a 'spot USD liquidity buffer pool' "
            "of at least 20% specifically dedicated to cash freight payments. At the same time, the system needs to monitor the Geopolitical Risk (GPR) "
            "Index. The moment the index breaks upward, it automatically dials back the hedging weight of long-term phantom exposures, "
            "preventing situations where overseas customs detentions cause forward contracts to mature with no physical currency available for delivery."
        ),
        "actionable_playbook": (
            "1) Track global containerized freight metrics via SSE/Baltic API webhooks daily; establish an automated risk trigger if weekly SCFI acceleration exceeds 15%. "
            "2) Enforce an immediate 'Logistics Liquidity Buffer', segregating a percentage of incoming foreign spot reserves from automated forward conversion funnels to cover spot freight payments. "
            "3) Stress-test the cash flow decay rate against active Geopolitical Risk (GPR) Index levels, shifting the core hedging posture toward highly flexible Short-Term Window Forwards or OTM Put Options."
        ),
        "data_sources": {
            "shanghai_shipping_exchange_data": "https://www.sse.net.cn/",
            "fed_geopolitical_risk_index": "https://www.matteoiacoviello.com/gpr.htm"
        }
    },
    {
        "id": "macro_boj_carry_trade_unwinding_shield",
        "pairs": "USD/CNH, EUR/CNH, GBP/CNH",
        "domain": "Macroeconomics & Global Liquidity Contraction Models",
        "title": "Global Liquidity Shocks: Mitigating Yen Carry Trade Unwinding and BoJ Hawkish Pivot Risks",
        "professional_definition": (
            "Global Carry Trade Unwinding, triggered by a hawkish Bank of Japan (BoJ) monetary policy pivot "
            "or the abandonment of Yield Curve Control (YCC), induces synchronized cross-asset margin liquidation. "
            "Speculative capital abruptly buys back JPY by shedding high-yielding long targets, injecting localized "
            "devaluation pressure into Western core assets while causing an un-forecasted, passive parabolic surge "
            "in Offshore Renminbi (CNH) macro-valuations. Treasury nodes must invoke systemic correlation multipliers "
            "to aggressively hedge sudden long-currency, short-CNH settlement structural exposures."
        ),
        "layman_translation": (
            "A rate hike or an exit from low-interest-rate policies by the Bank of Japan (BoJ) frequently triggers a global 'Yen carry trade unwinding storm.' "
            "To repay their Yen-denominated debts in a flash, global hedge funds will aggressively liquidate core assets like USD or GBP in a concentrated timeframe "
            "and convert them back into Yen to clear positions and bring capital home. This mechanism causes major overseas foreign currency assets "
            "to suffer a guillotine-style, passive, massive one-sided plunge against the offshore Renminbi (with the CNH surging like lightning within days). "
            "The moment the system catches a heavy hawkish warning from a BoJ monetary policy meeting, it must instantly and forcefully raise the 'defensive hedging ratio' "
            "for all sellers holding USD and GBP exposures. Sellers are strictly prohibited from running naked or engaging in spot speculation during these windows, "
            "all-out protecting settlement profit margins from severe hits caused by global capital relocation."
        ),
        "actionable_playbook": (
            "1) Screen the rolling Tokyo overnight interbank pricing index (TONAR) to continuously quantify systemic Yen funding spikes. "
            "2) Mandate an automatic 20% elevation in baseline EUR/CNH and GBP/CNH forward hedge execution volumes if BoJ minutes signal a policy band extension. "
            "3) Substitute standard linear spot execution pathways with Out-of-the-Money (OTM) Put Options 72 hours prior to Japan's interest rate decision to insulate corporate equity from policy-induced liquidity vacuums."
        ),
        "data_sources": {
            "bis_global_liquidity_indicators": "https://www.bis.org/",
            "boj_monetary_policy_archive": "https://www.boj.or.jp/en/mopo/mpms_release/index.htm"
        }
    },
    {
        "id": "fx_microstructure_liquidity_squeeze",
        "pairs": "USD/CNH,EUR/CNH,HKD/CNH",
        "domain": "FX Microstructure & Central Bank Liquidity Management",
        "title": "CNH Order Flow Toxicity, CCS Dislocation, and Offshore Squeeze Mechanics",
        "imf_indicators": "Offshore CNH Pool Scale (HKMA/HKAB), CNH HIBOR Fixings, Central Bank Bill Issuance Curve, Cross-Currency Swap (CCS) Spreads, FX Option Implied Volatility Surface Skew",
        "core_anchors": {
            "microstructure_friction": "Offshore CNH market operates as a friction-heavy parallel pool where liquidity can evaporate non-linearly due to clearing bank bottlenecks and cross-border settlement quotas.",
            "squeeze_transmission": "When CNH HIBOR spikes (via targeted central bank bill issuance or commercial bank hoarding), the cost of shorting CNH scales parabolically, forcing speculative short-covering.",
            "ccs_dislocation": "The deviation of CNH Cross-Currency Swap (CCS) from standard covered interest parity (CIP) measures the localized 'scarcity premium' of dollar liquidity vs. offshore CNH velocity."
        },
        "no_arbitrage_condition": "No-Arbitrage under structural friction requires: $R_{offshore\\_funding} = R_{onshore\\_interbank} + \\Delta_{capital\\_controls} + \\Omega_{liquidity\\_premium}$, where $\\Omega$ diverges aggressively during long-short speculative standoffs.",
        "mathematical_representation": {
            "implied_forward_points": "F_t - S_t = S_t \\times \\left( \\frac{1 + R^{CNH}_{HIBOR, \\tau} \\times \\frac{Days}{360}}{1 + R^{USD}_{SOFR, \\tau} \\times \\frac{Days}{360}} - 1 \\right)",
            "order_flow_toxicity": "VPIN_{CNH} = \\frac{\\sum_{\\tau=1}^V |V_\\tau^B - V_\\tau^S|}{V \\times N} \\implies \\text{When } VPIN > \\alpha, \\text{ market makers pull liquidity, accelerating the squeeze.}"
        },
        "nova_imf_macro_reasoning_rules": (
            "1. MONITOR CNH HIBOR curves daily; an inverted curve (O/N > 1M) indicates acute offshore pooling constraints or structural intervention.\n"
            "2. TRACK the basis spread between USD/CNY (Onshore) and USD/CNH (Offshore). If $|CNY - CNH| > \\text{Threshold}$, execute a cross-border friction arbitrage or risk-reversal hedging strategy.\n"
            "3. CROSS-REFERENCE implied volatility skew (25D Risk Reversal); a heavy bias towards USD call/CNH put coupled with dropping offshore deposits triggers the 'Liquidity Squeeze' alert.\n"
            "4. EVALUATE central bank bill maturities in Hong Kong. If bill rolling decreases during depreciation cycles, anticipate direct forward-market or swap-market intervention."
        ),
        "vector_description": (
            "Microstructure analysis of the offshore CNH market, focusing on liquidity squeezes, order flow toxicity (VPIN), CNH HIBOR anomalies, Cross-Currency Swap (CCS) deviations from CIP, and central bank defensive intervention dynamics in parallel currency pools."
        )
    },
    {
        "id": "compliance_safe_three_way_matching",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH,HKD/CNH",
        "domain": "Cross-Border Regulatory Compliance & Operational Risk",
        "title": "SAFE Bona Fide Trade Background Enforcement: Mitigating Cross-Border Three-Way Matching Failures",
        "imf_indicators": "SAFE Compliance Settlement Quota, Cross-Border Capital Flow Velocity, Order-to-Freight Manifest Match Ratio, Clearing Bank Authenticity Log",
        "core_anchors": {
            "authenticity_friction": "State Administration of Foreign Exchange (SAFE) enforces a strict bona fides test requiring absolute synchronization between cross-border cash velocity, digital storefront invoices, and physical 3PL freight manifests.",
            "settlement_bottleneck": "When massive forward contract maturities strike without matching digital order flows, clearing banks immediately freeze incoming onshore liquid wires under anti-underground banking guidelines.",
            "operational_remediation": "Treasury automation must programmatically map transaction-level platform webhooks directly to the derivative delivery manifest to eliminate accounting mismatches."
        },
        "no_arbitrage_condition": "Compliance clearance under regulation requires: $\\sum \\text{Derivatives Delivery}_{t} \\le \\sum \\text{Storefront Checkout Data}_{t-\\epsilon} + \\sum \\text{Reverse Logistics Adj}$, eliminating artificial speculative capital inflows.",
        "mathematical_representation": {
            "three_way_deviation": "\\Delta_{Compliance} = \\| \\mathbf{V}_{Cash\\_Flow} - (\\mathbf{V}_{Storefront\\_Sales} \\odot \\mathbf{V}_{Freight\\_Manifest}) \\|",
            "audit_trigger": "\\text{If } \\Delta_{Compliance} > \\theta, \\text{ trigger mandatory automated contract roll-over or freeze spot conversionGreen Channel.}"
        },
        "nova_imf_macro_reasoning_rules": (
            "1. EXTRACT transaction logs from Amazon/Shopify daily; compute cumulative standard deviations against active derivative delivery dates.\n"
            "2. IF a mismatch between physical container dispatch and forward delivery date exceeds 5 business days, trigger automated compliance window adjustments.\n"
            "3. STRICTLY restrict long-term outbound cross-border intercompany liquid pooling transfers unless backed by verified invoice matrices."
        ),
        "vector_description": (
            "Regulatory compliance frameworks for SAFE guidelines (Notice 11). Focuses on cross-border three-way matching (storefront invoices, cash receipts, and freight manifests), preventing underground banking contamination, and automated transactional audit log verification."
        )
    },
    {
        "id": "compliance_aml_gateway_freeze_contagion",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH",
        "domain": "Anti-Money Laundering (AML) & Payment Gateway Microstructure",
        "title": "Payment Gateway Collateral Damage: Mitigating AML Algorithmic Freezes and Exposure Contagion",
        "professional_definition": (
            "Multi-account global merchant cash pooling networks face high systemic operational risk stemming from "
            "automated payment gateway (e.g., Stripe, PayPal) AML algorithmic intervention. Sudden cross-border store-level "
            "fraud or high chargeback anomalies trigger automated account lockouts, halting multi-currency velocity. "
            "If treasury modules continue to execute rigid forward commitments on frozen underlying cash pools, "
            "the enterprise encounters instant liquidity collapse and forced technical default. Risk layers must "
            "deploy cross-entity systemic contagion multipliers to instantly decouple infected nodes from active hedging pools."
        ),
        "layman_translation": (
            "When using multiple legal entities and storefronts for capital aggregation, cross-border e-commerce sellers are highly vulnerable "
            "to automated Anti-Money Laundering (AML) algorithmic interventions by overseas payment gateways (like Stripe or PayPal), often triggered "
            "by unusual refund spikes or high chargeback rates in a single store. Once a storefront hits a mandatory 'Fund Freeze,' "
            "expected USD or GBP payout velocities instantly break down. If the system continues to blindly execute rigid forward commitments in the background, "
            "the enterprise will slip into sudden liquidity shock due to a total lack of available cash. The treasury module must monitor "
            "the risk control status of multi-account gateways in real time. The moment a gateway's freeze fuse is lit, it must instantly cut off "
            "the asset coupling of that specific storefront and dynamically scale down the active hedging size."
        ),
        "actionable_playbook": (
            "1) Parse real-time payment gateway webhook API responses for rolling fraud metrics and rolling reserves thresholds hourly. "
            "2) Dynamically adjust the operational cash pool scaling weight down by a 'Contagion Multiplier' (typically 0.40x) if any sibling entity triggers an AML audit flag. "
            "3) Redirect the remaining active, uninfected cash velocity into short-term multi-currency localized digital wallets to preserve baseline operational liquidity."
        ),
        "data_sources": {
            "fatf_cross_border_aml_standards": "https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Fatf-recommendations.html",
            "stripe_global_merchant_terms": "https://stripe.com/legal/ssa"
        }
    },
    {
        "id": "accounting_ifrs9_hedge_effectiveness",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH,HKD/CNH",
        "domain": "Hedge Accounting, Corporate Governance & Financial Audit",
        "title": "Corporate Governance under IFRS 9 / CAS 22: Algorithmic Hedge Effectiveness Testing and Audit Trail Integrity",
        "professional_definition": (
            "To qualify for Cash Flow Hedge accounting treatments under IFRS 9 and CAS 22, derivatives execution must pass "
            "strict structural and quantitative prospective and retrospective hedge effectiveness tests, preventing arbitrary "
            "income statement volatility. The system deploys the Dollar-Offset Method and linear regression models to verify "
            "that changes in the valuation of forward/option derivatives accurately neutralize the designated underlying marketplace "
            "receivables exposure within the rigorous $80\\% - 125\\%$ compliance bandwidth, generating an invariant immutable audit trail."
        ),
        "layman_translation": (
            "In the annual audits of public companies and mid-to-large cross-border enterprises, strict accounting guidelines (IFRS 9 / CAS 22) dictate "
            "that fx hedging transactions must never be classified as 'foreign exchange currency speculation'; otherwise, their mark-to-market P&L swings "
            "will directly wreck the company's official corporate income statements. The system must automatically generate a Hedge Designation Document "
            "at the inception of the trade, and automatically execute the Dollar-Offset Method at each quarter-end. This verifies that the valuation swings "
            "of forward contracts highly neutralize the fluctuations of overseas marketplace receivables—strictly locking the ratio within the mandatory "
            "80% to 125% highly effective hedge bandwidth. The automated quantitative audit trail printed out by the system serves as the company's sole financial passport "
            "to clear Big Four external audits smoothly."
        ),
        "actionable_playbook": (
            "1) Generate a cryptographic, timestamped Hedge Designation Document at the exact entry second of any forward or option transaction. "
            "2) Calculate the Retrospective Dollar-Offset ratio at each financial reporting period-end; instantly isolate any ineffective hedge drift to separate ledger accounts. "
            "3) Compile and export the standardized 'IFRS 9 Audit-Ready Matrix' automatically to the enterprise resource planning (ERP) database for Big Four auditor review."
        ),
        "data_sources": {
            "ifrs_9_financial_instruments_hub": "https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/",
            "mof_china_cas22_rules": "http://kjs.mof.gov.cn/"
        }
    },

]

_PAIR_CURRENCY_TOKEN: dict[str, str] = {
    "USD/CNH": "usd",
    "EUR/CNH": "eur",
    "HKD/CNH": "hkd",
    "GBP/CNH": "gbp",
}


def get_imf_macro_knowledge_entry() -> dict[str, Any] | None:
    """Return the structured IMF macro baseline entry from the vector knowledge base."""
    for doc in _FX_KNOWLEDGE_BASE:
        if doc.get("id") == IMF_MACRO_KB_ID:
            return doc
    return None


def _imf_as_of_display_label(as_of_date: str) -> str:
    """Convert '2026-06' → 'June 2026' for LLM-readable macro headers."""
    try:
        parsed = datetime.strptime(str(as_of_date), "%Y-%m")
        return parsed.strftime("%B %Y")
    except ValueError:
        return str(as_of_date)


def _doc_macro_indicators_text(doc: dict[str, Any]) -> str:
    """Serialize quantitative IMF indicators for dense/sparse retrieval."""
    indicators = doc.get("macro_indicators")
    if not isinstance(indicators, dict):
        return ""

    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    cofer = indicators.get("cofer_sovereign_reserves_share", {})
    if not isinstance(cpi, dict) or not isinstance(gdp, dict) or not isinstance(cofer, dict):
        return ""

    return (
        f"IMF macro indicators as of {indicators.get('as_of_date', '')} "
        f"CPI YoY US {cpi.get('US')}% Eurozone {cpi.get('Eurozone')}% "
        f"UK {cpi.get('UK')}% Hong Kong {cpi.get('Hong Kong')}% China {cpi.get('China')}% "
        f"GDP forecast US {gdp.get('US')}% Eurozone {gdp.get('Eurozone')}% "
        f"UK {gdp.get('UK')}% Hong Kong {gdp.get('Hong Kong')}% China {gdp.get('China')}% "
        f"COFER reserves USD {cofer.get('USD')}% EUR {cofer.get('EUR')}% "
        f"GBP {cofer.get('GBP')}% CNY {_fmt_cofer_reserve_label(cofer)}% "
        "PPP inflation spread long-term hedge budget 90-day 180-day"
    )


def format_imf_macro_indicators_block(
    macro_indicators: dict[str, Any] | None = None,
) -> str:
    """Format IMF quantitative block for [REAL-TIME MACRO INTELLIGENCE LAYER]."""
    entry = get_imf_macro_knowledge_entry()
    indicators = macro_indicators
    if indicators is None and entry is not None:
        indicators = entry.get("macro_indicators")
    if not isinstance(indicators, dict):
        return ""

    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    cofer = indicators.get("cofer_sovereign_reserves_share", {})
    if not isinstance(cpi, dict) or not isinstance(gdp, dict) or not isinstance(cofer, dict):
        return ""

    as_of_label = _imf_as_of_display_label(str(indicators.get("as_of_date", "N/A")))
    return (
        f"- IMF Long-Term Macro Anchor (As of {as_of_label}):\n"
        f"  * CPI Inflation YoY: US: {cpi.get('US', 'N/A')}%, "
        f"Eurozone: {cpi.get('Eurozone', 'N/A')}%, "
        f"UK: {cpi.get('UK', 'N/A')}%, Hong Kong: {cpi.get('Hong Kong', 'N/A')}%, "
        f"China: {cpi.get('China', 'N/A')}%\n"
        f"  * GDP Growth Forecast: US: {gdp.get('US', 'N/A')}%, "
        f"Eurozone: {gdp.get('Eurozone', 'N/A')}%, "
        f"UK: {gdp.get('UK', 'N/A')}%, Hong Kong: {gdp.get('Hong Kong', 'N/A')}%, "
        f"China: {gdp.get('China', 'N/A')}%\n"
        f"  * Global COFER Sovereign Reserves: USD: {cofer.get('USD', 'N/A')}%, "
        f"EUR: {cofer.get('EUR', 'N/A')}%, GBP: {cofer.get('GBP', 'N/A')}%, "
        f"CNY: {_fmt_cofer_reserve_label(cofer)}%"
    )


def compute_imf_ppp_spreads(
    pair: str = DEFAULT_PAIR,
    macro_indicators: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Structural CPI spread vs China (home jurisdiction) for desk pair routing."""
    entry = get_imf_macro_knowledge_entry()
    indicators = macro_indicators or (entry or {}).get("macro_indicators", {})
    cpi = indicators.get("inflation_cpi_yoy", {}) if isinstance(indicators, dict) else {}
    china_cpi = float(cpi.get(CHINA_MACRO_ANCHOR, cpi.get("China", 0.3)))

    destination_key = {
        "USD/CNH": "US",
        "EUR/CNH": "Eurozone",
        "HKD/CNH": "Hong Kong",
        "GBP/CNH": "UK",
    }.get(pair, "US")
    destination_cpi = float(cpi.get(destination_key, china_cpi))
    return {
        "china_cpi": china_cpi,
        "destination_cpi": destination_cpi,
        "cpi_spread_vs_china": round(destination_cpi - china_cpi, 2),
        "destination_label": destination_key,
    }


def _english_retrieval_anchor_text(doc: dict[str, Any]) -> str:
    """English-only dense retrieval anchor for MiniLM FAISS indexing."""
    blocks = [
        str(doc.get("domain", "") or "").strip(),
        str(doc.get("title", "") or "").strip(),
        str(doc.get("professional_definition", "") or "").strip(),
        str(doc.get("actionable_playbook", "") or "").strip(),
        str(doc.get("vector_description", "") or "").strip(),
    ]
    for key in (
        "core_anchors",
        "no_arbitrage_condition",
        "mathematical_representation",
        "nova_imf_macro_reasoning_rules",
    ):
        payload = doc.get(key)
        if payload is None:
            continue
        if isinstance(payload, (dict, list)):
            blocks.append(json.dumps(payload, ensure_ascii=False))
        else:
            blocks.append(str(payload).strip())
    macro = _doc_macro_indicators_text(doc)
    if macro:
        blocks.append(macro)
    tags = doc.get("tags")
    if tags:
        blocks.append(str(tags))
    return " || ".join(b for b in blocks if b)


def _load_embedding_model_impl():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL_NAME)


if st is not None:

    @st.cache_resource(show_spinner=False)
    def load_embedding_model():
        return _load_embedding_model_impl()

else:

    def load_embedding_model():
        return _load_embedding_model_impl()


def _network_error_types() -> tuple[type[BaseException], ...]:
    """Exception types captured for outbound HTTP/HTTPS fetches."""
    errs: tuple[type[BaseException], ...] = (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        ValueError,
        OSError,
    )
    if REQUESTS_AVAILABLE:
        import requests.exceptions as req_exc

        errs = errs + (req_exc.RequestException,)
    return errs


def _fetch_json_url(
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """GET JSON with strict timeout; returns None on any network/parse failure."""
    hdrs = headers or {}
    try:
        if REQUESTS_AVAILABLE:
            response = requests.get(url, timeout=HTTP_TIMEOUT_SEC, headers=hdrs)
            response.raise_for_status()
            payload = response.json()
        else:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as response:
                payload = json.load(response)
        return payload if isinstance(payload, dict) else None
    except _network_error_types():
        return None


def sanitize_user_query(query: str) -> str:
    """
    Prompt-injection guard: strip override phrases and cap length before RAG/LLM routing.
    """
    if not isinstance(query, str):
        return ""
    text = query.strip()
    for pattern in _PROMPT_INJECTION_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_USER_QUERY_LEN]


def _indexed_vector_count() -> int:
    """Active row count in the FAISS index or NumPy embedding fallback."""
    if FAISS_AVAILABLE and _vector_index is not None:
        return int(_vector_index.ntotal)
    if _embedding_matrix is not None:
        return int(_embedding_matrix.shape[0])
    return 0


def _vector_corpus_stale(knowledge_base: list[dict[str, Any]], force_rebuild: bool) -> bool:
    """True when corpus length or index rows diverge from the live knowledge base."""
    if force_rebuild:
        return True
    kb_len = len(knowledge_base)
    if kb_len == 0:
        return False
    if len(_indexed_docs_corpus) != kb_len:
        return True
    if _indexed_vector_count() != kb_len:
        return True
    return False


def init_vector_index(
    force_rebuild: bool = False,
    knowledge_base: list[dict[str, Any]] | None = None,
) -> None:
    """
    Build or refresh a local FAISS IndexFlatIP over L2-normalized MiniLM embeddings.
    Automatically re-indexes when _FX_KNOWLEDGE_BASE grows or _indexed_docs_corpus drifts.
    """
    global _vector_index, _indexed_docs_corpus, _embedding_matrix

    with _rag_lock:
        kb = knowledge_base if knowledge_base is not None else _FX_KNOWLEDGE_BASE
        if not _vector_corpus_stale(kb, force_rebuild) and _vector_index_ready():
            return

        _indexed_docs_corpus = list(kb)
        _vector_index = None
        _embedding_matrix = None

        if not kb:
            return

        try:
            texts = [_english_retrieval_anchor_text(doc) for doc in kb]
            model = load_embedding_model()
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(embeddings)
            else:
                _l2_normalize_inplace(embeddings)

            if FAISS_AVAILABLE:
                index = faiss.IndexFlatIP(EMBED_DIM)
                index.add(embeddings)
                _vector_index = index
            else:
                _embedding_matrix = embeddings

        except Exception:
            _vector_index = None
            _embedding_matrix = None
            _indexed_docs_corpus = list(kb)


def initialize_vector_db(knowledge_base: list[dict[str, Any]]) -> None:
    """Backward-compatible alias — always syncs corpus size with the knowledge base."""
    init_vector_index(force_rebuild=True, knowledge_base=knowledge_base)


def _l2_normalize_inplace(vectors: np.ndarray) -> None:
    """NumPy L2 normalize (FAISS-free fallback)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms


def _dense_vector_scores(query_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity scores vs all indexed docs (query_vec must be L2-normalized)."""
    with _rag_lock:
        if FAISS_AVAILABLE and _vector_index is not None and _vector_index.ntotal > 0:
            scores, _ = _vector_index.search(query_vec, _vector_index.ntotal)
            return scores[0]

        if _embedding_matrix is not None and _embedding_matrix.size > 0:
            return (_embedding_matrix @ query_vec[0]).astype(np.float32)

    return np.array([], dtype=np.float32)


init_vector_index()


def format_data_sources_block(doc: dict[str, Any]) -> str:
    """
    Format optional `data_sources` for LLM context / UI.
    Returns empty string when the key is absent or empty — never raises KeyError.
    """
    sources = doc.get("data_sources")
    if not sources or not isinstance(sources, dict):
        return ""

    lines: list[str] = []
    for label, url in sources.items():
        if not url or not isinstance(url, str):
            continue
        human_label = str(label).replace("_", " ").strip().title()
        lines.append(f"  * {human_label}: {url}")

    if not lines:
        return ""

    return "**Authoritative data sources:**\n" + "\n".join(lines)


def _data_sources_corpus_text(doc: dict[str, Any]) -> str:
    """Flatten data_sources labels/URLs for sparse + dense retrieval."""
    sources = doc.get("data_sources")
    if not sources or not isinstance(sources, dict):
        return ""
    parts = [f"{k} {v}" for k, v in sources.items() if v]
    return "official references " + " ".join(parts) if parts else ""


def _doc_corpus(doc: dict[str, Any]) -> str:
    structural_parts: list[str] = []
    for key in (
        "core_anchors",
        "no_arbitrage_condition",
        "mathematical_representation",
        "nova_imf_macro_reasoning_rules",
        "vector_description",
    ):
        payload = doc.get(key)
        if payload is None:
            continue
        if isinstance(payload, (dict, list)):
            structural_parts.append(json.dumps(payload, ensure_ascii=False))
        else:
            structural_parts.append(str(payload))
    return " ".join(
        p
        for p in (
            doc.get("title", ""),
            doc.get("domain", ""),
            doc.get("professional_definition", ""),
            doc.get("actionable_playbook", ""),
            doc.get("tags", ""),
            doc.get("pairs", ""),
            _doc_macro_indicators_text(doc),
            _data_sources_corpus_text(doc),
            *structural_parts,
        )
        if p
    )


def _serialize_structural_doc_payload(label: str, payload: Any) -> str:
    """Serialize dense microstructure / macro reasoning blocks for LLM context."""
    if payload is None:
        return ""
    if isinstance(payload, (dict, list)):
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        return f"**{label}:**\n```json\n{serialized}\n```"
    text = str(payload).strip()
    return f"**{label}:** {text}" if text else ""


def _format_doc_for_context(doc: dict[str, Any]) -> str:
    """Single playbook block for RAG / LLM context assembly (English-only telemetry)."""
    domain = doc.get("domain", "FX Risk")
    title = doc.get("title", "Playbook")
    parts = [
        f"**[{domain}] {title}** ({doc.get('pairs', 'ALL')})",
        f"**Professional definition:** {doc.get('professional_definition', '')}",
    ]
    playbook = doc.get("actionable_playbook", "")
    if playbook:
        parts.append(f"**Actionable playbook:** {playbook}")
    vector_desc = doc.get("vector_description", "")
    if vector_desc:
        parts.append(f"**Vector description:** {vector_desc}")

    for key, label in (
        ("core_anchors", "Core anchors"),
        ("no_arbitrage_condition", "No-arbitrage condition"),
        ("mathematical_representation", "Mathematical representation"),
        ("nova_imf_macro_reasoning_rules", "NOVA IMF macro reasoning rules"),
    ):
        block = _serialize_structural_doc_payload(label, doc.get(key))
        if block:
            parts.append(block)

    body = "\n\n".join(p for p in parts if p and str(p).strip())
    imf_block = format_imf_macro_indicators_block(doc.get("macro_indicators"))
    if imf_block:
        body += f"\n\n**IMF quantitative baseline (structured):**\n{imf_block}"

    sources_block = format_data_sources_block(doc)
    if sources_block:
        body += f"\n\n{sources_block}"
    return body


def assemble_retrieved_playbooks_context(
    docs: list[dict[str, Any]] | None,
    *,
    header: str = "[RETRIEVED CORPORATE FX PLAYBOOKS — RAG]",
) -> str:
    """Concatenate matched KB articles for prompt injection; skips empty retrieval."""
    if not docs:
        return ""

    blocks = [_format_doc_for_context(doc) for doc in docs if doc]
    blocks = [b for b in blocks if b.strip()]
    if not blocks:
        return ""

    return f"{header}\n\n" + "\n\n---\n\n".join(blocks)


def _month_end_pulse(dates: pd.DatetimeIndex, scale: float) -> np.ndarray:
    pulse = np.zeros(len(dates))
    for i, d in enumerate(dates):
        if d.day >= 25:
            pulse[i] = -scale * (d.day - 24) / 7
    return pulse


def _calendar_date_index(periods: int, end: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Real business-day calendar ending today — professional desk X-axis."""
    end_ts = (end or pd.Timestamp.now()).normalize()
    return pd.date_range(end=end_ts, periods=periods, freq="B")


def _coerce_calendar_dates(series: pd.Series, periods: int) -> pd.DatetimeIndex:
    """Ensure OHLC frames always carry parseable calendar dates, never ordinal labels."""
    parsed = pd.to_datetime(series, errors="coerce")
    if len(parsed) == periods and parsed.notna().all():
        idx = pd.DatetimeIndex(parsed)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx.normalize()
    return _calendar_date_index(periods)


def generate_fx_kline(pair: str = DEFAULT_PAIR, days: int = 30) -> pd.DataFrame:
    """Simulate daily FX bars for supported pairs with pair-specific dynamics."""
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")

    meta = PAIR_META[pair]
    rng = np.random.default_rng(meta["seed"])
    dates = _calendar_date_index(days)
    n = len(dates)
    t = np.linspace(0, 1, n)

    floor = meta["floor"]
    ceiling = meta["ceiling"]
    center = meta["center"] + meta.get("cycle_amp", 0.04) * np.sin(2 * np.pi * t)
    month_pulse = _month_end_pulse(dates, scale=0.025 if pair != "HKD/CNH" else 0.004)

    close = np.empty(n)
    close[0] = meta["init"]

    if pair == "HKD/CNH":
        peg_mid = meta["peg_mid"]
        weakening = np.linspace(0, meta["weakening_total"], n)
        for i in range(1, n):
            shock = rng.normal(0, meta["shock"])
            target = peg_mid - weakening[i] + 0.003 * np.sin(4 * np.pi * t[i])
            reversion = (target - close[i - 1]) * meta["reversion"]
            close[i] = close[i - 1] + shock + reversion + month_pulse[i]
            if close[i] < floor + 0.002:
                close[i] = floor + rng.uniform(0.002, 0.006)
            elif close[i] > ceiling - 0.002:
                close[i] = ceiling - rng.uniform(0.002, 0.006)
    else:
        for i in range(1, n):
            shock = rng.normal(0, meta["shock"])
            reversion = (center[i] - close[i - 1]) * meta["reversion"]
            close[i] = close[i - 1] + shock + reversion + month_pulse[i]
            if close[i] > ceiling - 0.01:
                close[i] = ceiling - rng.uniform(0.01, 0.03)
            elif close[i] < floor + 0.01:
                close[i] = floor + rng.uniform(0.01, 0.03)

    close = np.clip(close, floor, ceiling)
    decimals = 4

    open_ = np.roll(close, 1)
    open_[0] = close[0] + rng.normal(0, meta["shock"] * 0.6)
    wick = 0.018 if pair != "HKD/CNH" else 0.004
    high = np.maximum(open_, close) + rng.uniform(wick * 0.2, wick, n)
    low = np.minimum(open_, close) - rng.uniform(wick * 0.2, wick, n)
    high = np.clip(high, floor, ceiling)
    low = np.clip(low, floor, ceiling)

    vol_scale = 1e6 if pair != "HKD/CNH" else 3e5
    vol_proxy = np.abs(np.diff(close, prepend=close[0])) * vol_scale + rng.integers(
        int(vol_scale * 2), int(vol_scale * 6), n
    )
    volatility = pd.Series(close).pct_change().abs().fillna(0) * 100

    df = pd.DataFrame(
        {
            "date": dates,
            "open": np.round(open_, decimals),
            "high": np.round(high, decimals),
            "low": np.round(low, decimals),
            "close": np.round(close, decimals),
            "volume": vol_proxy.astype(int),
            "volatility_pct": np.round(volatility.values * 100, 3),
        }
    )
    df["ma5"] = df["close"].rolling(window=5, min_periods=1).mean()
    df.attrs["pair"] = pair
    df.attrs["data_source"] = "simulated"
    return df


def _spot_frame_to_kline(spot_df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Convert real Spot/MA5 frame into Plotly-ready OHLCV bars.
    Open/High/Low are proportionally derived from Close (~0.5% bands).
    """
    meta = PAIR_META[pair]
    rng = np.random.default_rng(meta["seed"])
    floor, ceiling = meta["floor"], meta["ceiling"]

    close = spot_df["Spot"].astype(float).values
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    if n > 1:
        open_[1:] = close[:-1]

    band = close * 0.005
    wick = rng.uniform(0.15, 1.0, n) * band
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick

    open_ = np.clip(open_, floor, ceiling)
    high = np.clip(high, floor, ceiling)
    low = np.clip(low, floor, ceiling)

    vol_scale = 1e6 if pair != "HKD/CNH" else 3e5
    vol_proxy = np.abs(np.diff(close, prepend=close[0])) * vol_scale + rng.integers(
        int(vol_scale * 2), int(vol_scale * 6), n
    )
    volatility = pd.Series(close).pct_change().abs().fillna(0) * 100

    ma5 = (
        spot_df["MA5"].astype(float).values
        if "MA5" in spot_df.columns
        else pd.Series(close).rolling(5, min_periods=1).mean().values
    )

    chart_dates = _coerce_calendar_dates(spot_df["date"], n)
    df = pd.DataFrame(
        {
            "date": chart_dates,
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "volume": vol_proxy.astype(int),
            "volatility_pct": np.round(volatility.values * 100, 3),
            "ma5": np.round(ma5, 4),
        }
    )
    df.attrs["pair"] = pair
    df.attrs["data_source"] = spot_df.attrs.get("data_source", "unknown")
    df.attrs["ticker"] = spot_df.attrs.get(
        "ticker", YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")
    )
    return df


def fetch_fx_kline(pair: str = DEFAULT_PAIR, days: int = 30) -> pd.DataFrame:
    """Live desk K-line: yfinance Spot ingestion → OHLCV candlestick frame."""
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")
    spot_df = fetch_real_market_data(days=days, pair=pair)
    return _spot_frame_to_kline(spot_df, pair)


def fetch_usdcnh_kline(days: int = 30) -> pd.DataFrame:
    """Backward-compatible alias."""
    return fetch_fx_kline("USD/CNH", days=days)


def get_fx_metrics(df: pd.DataFrame, pair: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {}

    pair = pair or df.attrs.get("pair", DEFAULT_PAIR)
    meta = PAIR_META[pair]
    base = meta["base_ccy"]

    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill()
    ma5_col = (
        pd.to_numeric(df["ma5"], errors="coerce").ffill().bfill()
        if "ma5" in df.columns
        else close.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )

    latest = df.iloc[-1]
    prev_raw = float(close.iloc[-2]) if len(close) >= 2 else float(close.iloc[-1])
    prev = prev_raw if prev_raw and not np.isnan(prev_raw) else float(close.iloc[-1]) or 1.0
    spot_raw = float(close.iloc[-1])
    ma5_raw = float(ma5_col.iloc[-1])
    pct = (spot_raw - prev) / prev * 100 if prev else 0.0

    p75 = float(close.quantile(0.75))
    p25 = float(close.quantile(0.25))
    spot = spot_raw if spot_raw and not np.isnan(spot_raw) else prev
    ma5 = ma5_raw if ma5_raw and not np.isnan(ma5_raw) else spot

    latest_date_raw = latest["date"]
    if hasattr(latest_date_raw, "strftime"):
        latest_date_str = latest_date_raw.strftime("%Y-%m-%d")
    else:
        latest_date_str = str(latest_date_raw)[:10]

    if spot >= p75:
        risk_level = "HIGH"
        risk_color = "#ef4444"
        if pair == "EUR/CNH":
            risk_advice = (
                "EUR/CNH is elevated: lock 60–75% of net EUR receivables (post-VAT) "
                "with a 3-month forward before remittance week."
            )
        elif pair == "HKD/CNH":
            risk_advice = (
                "HKD is weak vs CNH (low HKD/CNH): accelerate CNH conversion for "
                "approved onshore payables; tranche-convert HKD receipts rather than hoarding."
            )
        elif pair == "GBP/CNH":
            risk_advice = (
                "GBP/CNH is elevated: lock 60–70% of net GBP receivables (post-UK VAT) "
                "with a 3-month forward before Amazon UK remittance week."
            )
        else:
            risk_advice = (
                "USD/CNH is elevated: lock 60–80% of confirmed USD receivables "
                "with a 3-month forward to protect conversion value."
            )
    elif spot <= p25:
        risk_level = "LOW"
        risk_color = "#22c55e"
        if pair == "HKD/CNH":
            risk_advice = (
                "HKD/CNH is firm (HKD strong): favorable for spot CNH conversion; "
                "convert ~40% now, keep buffer for peg-volatility headlines."
            )
        elif pair == "EUR/CNH":
            risk_advice = (
                "EUR/CNH is attractive: spot-convert ~40% of net EUR; "
                "keep 30% unhedged for potential ECB-driven rebound."
            )
        elif pair == "GBP/CNH":
            risk_advice = (
                "GBP/CNH is attractive: spot-convert ~40% of net GBP; "
                "keep 30% unhedged for potential BoE-driven rebound on UK payouts."
            )
        else:
            risk_advice = (
                "USD/CNH is favorable: spot-convert ~40% now; "
                "keep residual exposure for potential USD rebound."
            )
    else:
        risk_level = "MEDIUM"
        risk_color = "#f59e0b"
        risk_advice = (
            f"Neutral {pair} zone: ladder forwards—add 10% hedge for each +50 bps move."
        )

    ccy_word = base
    ma_signal = (
        f"{ccy_word} strengthening (spot > MA5)"
        if spot > ma5
        else f"CNH firming vs {ccy_word} (spot < MA5)"
    )
    if spot > ma5 and risk_level != "LOW":
        risk_advice += f" | MA5 trigger: short-term {ccy_word} strength—prioritize locking."

    return {
        "pair": pair,
        "pair_label": meta["label"],
        "base_ccy": base,
        "symbol": meta["symbol"],
        "latest_date": latest_date_str,
        "spot": spot,
        "ma5": ma5,
        "pct_change": round(float(pct), 3),
        "high_30d": float(df["high"].max()),
        "low_30d": float(df["low"].min()),
        "p75": float(p75),
        "p25": float(p25),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_advice": risk_advice,
        "ma_signal": ma_signal,
        "data_source": df.attrs.get("data_source", "unknown"),
        "ticker": df.attrs.get("ticker", YFINANCE_PAIR_TICKERS.get(pair, "")),
    }


def _extract_close_column(historical_df: pd.DataFrame) -> pd.Series:
    """Resolve Yahoo/Alpha Vantage close column (`Close` or `close` or `Spot`)."""
    for col in ("Close", "close", "Spot"):
        if col in historical_df.columns:
            return pd.to_numeric(historical_df[col], errors="coerce")
    return pd.Series(dtype=float)


def resolve_vat_rate(pair: str, custom_vat_rate: float | None = None) -> float:
    """Pair VAT baseline with EU slider override for EUR/CNH desks."""
    base_rate = float(VAT_MAP.get(pair, 0.0))
    if pair == "EUR/CNH" and custom_vat_rate is not None:
        return float(custom_vat_rate)
    return base_rate


def calculate_vat_net_payout_series(
    historical_df: pd.DataFrame,
    gross_sales_base: float,
    pair: str,
    custom_vat_rate: float | None = None,
) -> pd.Series:
    """
    Synthesize VAT net payout: gross_sales_base × (1 − VAT) × historical close.
    EUR/CNH accepts a dynamic member-state VAT override from the UI slider.
    """
    close = _extract_close_column(historical_df)
    if close.empty:
        return pd.Series(dtype=float)

    vat_rate = resolve_vat_rate(pair, custom_vat_rate)
    gross_base = max(0.0, float(gross_sales_base))
    payout = gross_base * (1.0 - vat_rate) * close
    return _sanitize_spot_series(payout)


def compute_rsi_and_macd(
    payout_series: pd.Series,
    rsi_period: int = 14,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, Any]:
    """
    Wilder RSI + MACD on VAT net payout (pandas `.diff()` / `.ewm()` only).
    Returns latest scalar points plus the full indicator frame.
    """
    empty: dict[str, Any] = {
        "rsi_val": 50.0,
        "macd_hist": 0.0,
        "macd_line_val": 0.0,
        "macd_signal_val": 0.0,
        "frame": pd.DataFrame(),
    }
    series = _sanitize_spot_series(payout_series.astype(float))
    if series.empty:
        return empty

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / max(1, int(rsi_period))
    avg_gain = gain.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    ema_fast = series.ewm(span=max(1, int(fast_period)), adjust=False).mean()
    ema_slow = series.ewm(span=max(1, int(slow_period)), adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=max(1, int(signal_period)), adjust=False).mean()
    macd_histogram = macd_line - macd_signal

    frame = pd.DataFrame(
        {
            "net_payout": series,
            "rsi": rsi,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
        }
    )
    idx = -1

    def _safe_scalar(col: str, default: float = 0.0) -> float:
        raw = frame[col].iloc[idx]
        if pd.isna(raw):
            return default
        return float(raw)

    return {
        "rsi_val": round(_safe_scalar("rsi", 50.0), 2),
        "macd_hist": round(_safe_scalar("macd_histogram", 0.0), 6),
        "macd_line_val": round(_safe_scalar("macd_line", 0.0), 6),
        "macd_signal_val": round(_safe_scalar("macd_signal", 0.0), 6),
        "frame": frame,
    }


def _macd_histogram_cross_flags(histogram: pd.Series) -> tuple[bool, bool]:
    """Detect MACD histogram zero-line crosses on the latest bar."""
    if histogram is None or len(histogram) < 2:
        return False, False
    prev = float(histogram.iloc[-2])
    curr = float(histogram.iloc[-1])
    if np.isnan(prev) or np.isnan(curr):
        return False, False
    crossed_below = prev >= 0.0 and curr < 0.0
    crossed_above = prev <= 0.0 and curr > 0.0
    return crossed_below, crossed_above


def compute_net_payout_technical_metrics(
    df: pd.DataFrame,
    pair: str | None = None,
    gross_sales_base: float = 10_000.0,
    custom_vat_rate: float | None = None,
) -> dict[str, Any]:
    """RSI/MACD snapshot on tax-adjusted VAT net payout for the active desk pair."""
    active_pair = pair or df.attrs.get("pair", DEFAULT_PAIR)
    if df.empty or _extract_close_column(df).empty:
        return {}

    vat_rate = resolve_vat_rate(active_pair, custom_vat_rate)
    payout_series = calculate_vat_net_payout_series(
        df,
        gross_sales_base=gross_sales_base,
        pair=active_pair,
        custom_vat_rate=custom_vat_rate,
    )
    ta = compute_rsi_and_macd(payout_series)
    frame = ta.get("frame")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}

    rsi_val = float(ta.get("rsi_val", 50.0))
    macd_hist = float(ta.get("macd_hist", 0.0))
    macd_line = float(ta.get("macd_line_val", 0.0))
    macd_signal = float(ta.get("macd_signal_val", 0.0))
    crossed_below, crossed_above = _macd_histogram_cross_flags(frame["macd_histogram"])

    return {
        "net_payout_rsi": rsi_val,
        "net_payout_macd_line": macd_line,
        "net_payout_macd_signal": macd_signal,
        "net_payout_macd_histogram": macd_hist,
        "macd_histogram_crossed_below": crossed_below,
        "macd_histogram_crossed_above": crossed_above,
        "vat_deduction_rate": vat_rate,
        "gross_sales_base": float(gross_sales_base),
        "net_payout_latest": float(frame["net_payout"].iloc[-1]),
    }


def build_nova_risk_advisory(
    selected_pair: str,
    rsi_val: float,
    macd_hist: float,
    *,
    vat_rate: float | None = None,
    gross_sales_base: float | None = None,
    macd_line: float = 0.0,
    macd_signal: float = 0.0,
) -> dict[str, str]:
    """
    Executive NOVA Risk Advisory — VAT net payout RSI/MACD saturation filter.
    """
    pair = selected_pair or DEFAULT_PAIR
    quote = OFFSHORE_QUOTE_CCY
    vat_pct = int(round(float(vat_rate or VAT_MAP.get(pair, 0.0)) * 100))
    gross_label = (
        f"{float(gross_sales_base):,.0f}" if gross_sales_base is not None else "N/A"
    )

    base_ccy = PAIR_META.get(pair, {}).get("base_ccy", "FX")
    if rsi_val > 70.0 and macd_hist < 0.0:
        return {
            "signal_class": "risk_alert",
            "text": (
                f"📊 **NOVA Risk Advisory ({pair})**: Your custom tax-adjusted VAT Net Payout RSI "
                f"is highly saturated at {rsi_val:.2f} (Overbought), with a bearish MACD crossover "
                f"({macd_hist:.4f}). Momentum is reversing. **Action Recommended**: Lock in your net "
                f"{base_ccy} proceeds immediately using Window Forwards to immunize against {quote} appreciation."
            ),
        }

    if rsi_val < 30.0 and macd_hist > 0.0:
        return {
            "signal_class": "opportunity",
            "text": (
                f"📊 **NOVA Risk Advisory ({pair})**: Net Payout RSI shows oversold exhaustion at "
                f"{rsi_val:.2f} with bullish MACD histogram divergence ({macd_hist:.4f}). "
                f"**Action Recommended**: Selling pressure is fading; delay immediate spot conversion "
                f"to capture upcoming rebounds."
            ),
        }

    return {
        "signal_class": "normal",
        "text": (
            f"📊 **NOVA Risk Advisory ({pair}) — Corporate Audit Log**: VAT Net Payout momentum is "
            f"within neutral stability bounds. Gross revenue base **{gross_label}** · applied VAT "
            f"**{vat_pct}%** · RSI **{rsi_val:.2f}** · MACD Line **{macd_line:.6f}** · Signal "
            f"**{macd_signal:.6f}** · Histogram **{macd_hist:.6f}** ({quote} desk). "
            f"Maintain baseline hedge ladder; no saturation reversal trigger active."
        ),
    }


def build_net_payout_anti_saturation_advisory(metrics: dict[str, Any]) -> dict[str, str]:
    """Backward-compatible wrapper delegating to `build_nova_risk_advisory`."""
    return build_nova_risk_advisory(
        metrics.get("pair", DEFAULT_PAIR),
        float(metrics.get("net_payout_rsi", 50.0)),
        float(metrics.get("net_payout_macd_histogram", 0.0)),
        vat_rate=metrics.get("vat_deduction_rate"),
        gross_sales_base=metrics.get("gross_sales_base"),
        macd_line=float(metrics.get("net_payout_macd_line", 0.0)),
        macd_signal=float(metrics.get("net_payout_macd_signal", 0.0)),
    )


def _multi_factor_series(
    spot: pd.Series,
    ma5: pd.Series | None = None,
) -> dict[str, Any]:
    """Vectorized multi-factor components aligned with the backtest hedge engine."""
    spot = _sanitize_spot_series(spot.astype(float))
    ma5_series = (
        _sanitize_spot_series(ma5.astype(float))
        if ma5 is not None
        else spot.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )
    ma20 = spot.rolling(window=20, min_periods=1).mean().ffill().bfill()
    std20 = spot.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    upper_band = ma20 + 2.0 * std20
    ma30 = spot.rolling(window=30, min_periods=1).mean()

    factor_1 = np.where(spot >= ma5_series, 1.0, 0.0)
    factor_2 = np.where(spot >= upper_band, 1.0, 0.0)
    factor_3 = np.where(spot >= ma30, 1.0, 0.0)
    total_score = (factor_1 * 0.4) + (factor_2 * 0.3) + (factor_3 * 0.3)
    hedge_ratio = np.select(
        [total_score >= 0.7, total_score >= 0.4],
        [0.80, 0.55],
        default=0.30,
    )

    return {
        "ma5_series": ma5_series,
        "ma30": ma30,
        "upper_band": upper_band,
        "factor_1": factor_1,
        "factor_2": factor_2,
        "factor_3": factor_3,
        "total_score": total_score,
        "hedge_ratio": hedge_ratio,
    }


def get_live_desk_multi_factor_metrics(
    df: pd.DataFrame,
    pair: str | None = None,
    gross_sales_base: float = 10_000.0,
    custom_vat_rate: float | None = None,
) -> dict[str, Any]:
    """Live desk metrics + MA30, Bollinger upper, multi-factor score, and hedge ratio."""
    metrics = get_fx_metrics(df, pair=pair)
    if not metrics or df.empty:
        return metrics

    spot_series = df["close"].astype(float)
    ma5_input = df["ma5"].astype(float) if "ma5" in df.columns else None
    mf = _multi_factor_series(spot_series, ma5_input)
    idx = -1

    spot = float(spot_series.iloc[idx])
    ma5_val = float(mf["ma5_series"].iloc[idx])
    ma30_val = float(mf["ma30"].iloc[idx])
    bollinger_upper = float(mf["upper_band"].iloc[idx])
    total_score = float(mf["total_score"][idx])
    hedge_ratio = float(mf["hedge_ratio"][idx])

    if total_score >= 0.7:
        mf_risk = "HIGH"
    elif total_score >= 0.4:
        mf_risk = "MEDIUM"
    else:
        mf_risk = "LOW"

    metrics.update(
        {
            "ma30": ma30_val,
            "bollinger_upper": bollinger_upper,
            "multi_factor_score": round(total_score, 3),
            "factor_trend": float(mf["factor_1"][idx]),
            "factor_bollinger": float(mf["factor_2"][idx]),
            "factor_macro": float(mf["factor_3"][idx]),
            "multi_factor_risk_rating": mf_risk,
            "recommended_hedge_ratio": hedge_ratio,
        }
    )
    metrics.update(
        compute_net_payout_technical_metrics(
            df,
            pair=metrics.get("pair"),
            gross_sales_base=gross_sales_base,
            custom_vat_rate=custom_vat_rate,
        )
    )
    nova_advisory = build_nova_risk_advisory(
        metrics.get("pair", DEFAULT_PAIR),
        float(metrics.get("net_payout_rsi", 50.0)),
        float(metrics.get("net_payout_macd_histogram", 0.0)),
        vat_rate=metrics.get("vat_deduction_rate"),
        gross_sales_base=metrics.get("gross_sales_base", gross_sales_base),
        macd_line=float(metrics.get("net_payout_macd_line", 0.0)),
        macd_signal=float(metrics.get("net_payout_macd_signal", 0.0)),
    )
    metrics["net_payout_signal_class"] = nova_advisory.get("signal_class", "normal")
    metrics["net_payout_advisory"] = nova_advisory.get("text", "")
    metrics["nova_risk_advisory"] = metrics["net_payout_advisory"]
    metrics["nova_risk_signal_class"] = metrics["net_payout_signal_class"]
    return metrics


def build_live_market_context_prompt(metrics: dict[str, Any]) -> str:
    """High-priority live desk snapshot injected ahead of each copilot user turn."""
    pair = metrics.get("pair", DEFAULT_PAIR)
    spot = float(metrics.get("spot", 0.0))
    ma5_val = float(metrics.get("ma5", 0.0))
    ma30_val = float(metrics.get("ma30", spot))
    bollinger_upper = float(metrics.get("bollinger_upper", spot))
    risk_level = metrics.get("multi_factor_risk_rating", metrics.get("risk_level", "MEDIUM"))
    hedge_ratio = float(metrics.get("recommended_hedge_ratio", 0.55))
    data_source = metrics.get("data_source", "unknown")
    latest_date = metrics.get("latest_date", "")

    return (
        "[SYSTEM LIVE MARKET CONTEXT - HIGH PRIORITY]\n"
        f"- Trading Desk: {pair}\n"
        f"- Data Feed: {data_source} (as of {latest_date})\n"
        f"- Real-time Spot: {spot:.4f}\n"
        f"- MA5 Dynamics: {ma5_val:.4f}\n"
        f"- MA30 Macro Trend: {ma30_val:.4f}\n"
        f"- Bollinger Upper Barrier (20d, 2σ): {bollinger_upper:.4f}\n"
        f"- Multi-Factor Score (F1×0.4 + F2×0.3 + F3×0.3): {metrics.get('multi_factor_score', 0):.3f}\n"
        f"- Current System Multi-Factor Risk Rating: {risk_level}\n"
        f"- Calculated Multi-Factor Hedge Ratio: {hedge_ratio:.1%}\n"
        f"- Factor Flags — Trend(MA5): {int(metrics.get('factor_trend', 0))} | "
        f"Bollinger: {int(metrics.get('factor_bollinger', 0))} | "
        f"Macro(MA30): {int(metrics.get('factor_macro', 0))}\n"
        f"- VAT Net Payout RSI (applied VAT {int(float(metrics.get('vat_deduction_rate', 0)) * 100)}%): "
        f"{float(metrics.get('net_payout_rsi', 50.0)):.2f}\n"
        f"- VAT Net Payout MACD Histogram: "
        f"{float(metrics.get('net_payout_macd_histogram', 0.0)):.6f}\n"
        f"- Anti-Saturation Copilot Signal: "
        f"{metrics.get('net_payout_signal_class', 'normal').upper()}\n"
        f"- Advisory: {metrics.get('net_payout_advisory', 'N/A')}\n"
        "\n"
        'Cross-reference the "Corporate FX Risk Management Manual" (retrieved via RAG) '
        "with the LIVE snapshot above to deliver an explicit, mathematical execution advisory. "
        "If the user states any numeric payout or receivable exposure, you MUST compute the "
        "precise base-currency notional to forward-lock using the Calculated Multi-Factor Hedge "
        "Ratio and Real-time Spot, and show implied CNH settlement."
    )


# ---------------------------------------------------------------------------
# Macro Intelligence Layer — China Money · ^TNX Fed Proxy · IMF (static KB)
# ---------------------------------------------------------------------------
CHINA_MONEY_CCPR_URL = (
    "https://www.chinamoney.com.cn/ags/ms/cm-u-bk-ccpr/CcprHisNew"
)
TNX_FEDWATCH_PROXY_TICKER = "^TNX"
_MACRO_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CHINAMONEY_CACHE_FILE = _MACRO_CACHE_DIR / "chinamoney_ccpr.json"
_chinamoney_memory_cache: dict[str, Any] | None = None

NOVA_AGENT_SYSTEM_INSTRUCTIONS = """
You are NOVA, an institutional-grade cross-border FX Risk Copilot built for global enterprise treasuries. You process multi-factor market data, macro IMF policy boundaries, and tech indicators into robust hedging strategies.

For every single query, you must structure your response using the exact three-stage analytical pipeline below. Your text must be professional, compact, and completely in English (Reuters standard), weaving dynamic data points naturally into the reasoning layer:

### ⚡ NOVA Multi-Factor Risk Advisory

**1. Quantitative Ingestion & Market Signals [DATA]**
- State the relative positioning of the current Spot Rate against the MA5/MA30 factor corridor and the 30-Day Range boundaries.
- Seamlessly weave in active numbers from the macro engine: the specific IMF 2026 growth anchor status, current sovereign yield spreads (like Alpha Vantage 10Y Treasury), and the CFETS Central Parity boundary drift.

**2. Transmission Channel & Treasury Risk Impact [ANALYSIS]**
- Explain how the observed macro rate/policy spreads interact with short-term quantitative momentum (RSI/MACD saturation or exhaustion traps).
- Detail the impact on cross-border merchant micro-revenues (e.g., how platform payout cycles or VAT withholding adjustments are affected by this technical/macro convergence).

**3. Optimized Execution Framework [CONCLUSION]**
- Deliver a crystal-clear, non-ambiguous risk management directive (e.g., Layered Window Forward Execution, Delayed Discretionary Spot Clearance, or Meta/Google Ad Spend Asset Matching).
- Specify exact execution metrics: hedge ratios (e.g., 35% Layered Hedge) and protective price brackets to immunize cash flows from immediate volatility shocks.

Negative Constraint: Do not print raw JSON dictionary dumps, and never exceed 200 words in total. Maintain a cohesive, bulleted, and executive-ready typography.
"""

NOVA_IMF_MACRO_REASONING_RULES = (
    "IMF MACRO APPLICATION RULES (mandatory when IMF quantitative baseline is present):\n"
    "1) Analyze Inflation Spreads (PPP): Explicitly calculate the structural CPI gap between the "
    "destination market (US at 3.4% or Eurozone at 2.4%) and the home jurisdiction (China at 0.3%). "
    "State the spread in percentage points and explain that, under Purchasing Power Parity, a higher "
    "inflation differential erodes the destination currency's purchasing power over a 90–180-day tail, "
    "justifying a strategic forward hedge on cross-border receivables.\n"
    "2) Synthesize GDP Velocity: Reference GDP growth forecasts (US 2.7% vs Eurozone 0.8% vs China 4.6%) "
    "to explain relative capital-flow and yield support—e.g., stronger US GDP velocity vs Eurozone stagnation "
    "supports USD over EUR on a fundamental basis when budgeting multi-currency locks.\n"
    "3) Sovereign Trust Check: Cite COFER sovereign reserve shares (USD 58.4%, EUR 19.8%, CNY 2.3%) to "
    "reassure the merchant that USD liquidity remains the ultimate structural anchor during extreme market "
    "panics; use this when users ask about long-term lock-in, treasury safety, or 90–180-day budgeting.\n"
    "4) For long-term lock-in or financial budgeting questions, integrate IMF metrics with the live desk "
    "MA30/Bollinger/multi-factor hedge ratio—never invent IMF figures beyond those injected."
)


def build_nova_system_instructions(include_imf_rules: bool = True) -> str:
    """Assemble NOVA system prompt with optional IMF reasoning enforcement."""
    if include_imf_rules and get_imf_macro_knowledge_entry():
        return f"{NOVA_AGENT_SYSTEM_INSTRUCTIONS}\n\n{NOVA_IMF_MACRO_REASONING_RULES}"
    return NOVA_AGENT_SYSTEM_INSTRUCTIONS


def _beijing_trading_date() -> str:
    """Calendar date in Asia/Shanghai for intraday China Money cache keys."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d")


def _load_chinamoney_disk_cache() -> dict[str, Any] | None:
    if not _CHINAMONEY_CACHE_FILE.exists():
        return None
    try:
        with _CHINAMONEY_CACHE_FILE.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("cache_date") == _beijing_trading_date():
            return payload.get("result")
    except (json.JSONDecodeError, OSError, TypeError):
        return None
    return None


def _save_chinamoney_disk_cache(result: dict[str, Any]) -> None:
    try:
        _MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with _CHINAMONEY_CACHE_FILE.open("w", encoding="utf-8") as handle:
            json.dump(
                {"cache_date": _beijing_trading_date(), "result": result},
                handle,
                ensure_ascii=False,
            )
    except OSError:
        pass


def _parse_chinamoney_ccpr_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract latest official USD/CNY central parity from China Money JSON."""
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return None

    heads = payload.get("data", {}).get("head", [])
    if not isinstance(heads, list) or "USD/CNY" not in heads:
        return None

    usd_idx = heads.index("USD/CNY")
    latest = records[0]
    values = latest.get("values")
    if not isinstance(values, list) or len(values) <= usd_idx:
        return None

    try:
        usd_cny = float(values[usd_idx])
    except (TypeError, ValueError):
        return None

    return {
        "status": "ok",
        "usd_cny": usd_cny,
        "parity_date": str(latest.get("date", "")),
        "source": "chinamoney",
        "policy_note": (
            "Official onshore CNY (RMB) central parity (USD/CNY) published by CFETS/China Money; "
            "use as regulatory policy boundary vs offshore CNH spot."
        ),
    }


def fetch_china_money_central_parity() -> dict[str, Any]:
    """
    China Money dynamic policy boundary layer.
    Intraday cache (memory + local file). On any failure → {"status": "failed"} only.
    """
    global _chinamoney_memory_cache

    today = _beijing_trading_date()
    if (
        _chinamoney_memory_cache is not None
        and _chinamoney_memory_cache.get("cache_date") == today
    ):
        cached = _chinamoney_memory_cache.get("result")
        if isinstance(cached, dict):
            return cached

    disk_cached = _load_chinamoney_disk_cache()
    if isinstance(disk_cached, dict) and disk_cached.get("status") == "ok":
        _chinamoney_memory_cache = {"cache_date": today, "result": disk_cached}
        return disk_cached

    payload = _fetch_json_url(
        CHINA_MONEY_CCPR_URL,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; NOVA-FX-Copilot/1.0; +quant-desk)",
            "Referer": "https://www.chinamoney.com.cn/",
        },
    )
    if payload is None:
        return {"status": "failed"}

    try:
        parsed = _parse_chinamoney_ccpr_payload(payload)
        if parsed is None:
            return {"status": "failed"}

        _chinamoney_memory_cache = {"cache_date": today, "result": parsed}
        _save_chinamoney_disk_cache(parsed)
        return parsed
    except (TypeError, ValueError, KeyError):
        return {"status": "failed"}


def _analyze_tnx_yield_series(
    close: pd.Series,
    *,
    fedwatch_data_source: str = "yfinance",
    ticker_label: str | None = None,
) -> dict[str, Any]:
    """Compute 10Y yield MA5/MA30, Bollinger middle band, and dovish death-cross signal."""
    yields = close.astype(float).dropna().sort_index().tail(30)
    if len(yields) < 10:
        raise ValueError("insufficient TNX history")

    ma5 = yields.rolling(window=5, min_periods=1).mean()
    ma30 = yields.rolling(window=30, min_periods=1).mean()
    bb_mid = yields.rolling(window=20, min_periods=1).mean()

    current = float(yields.iloc[-1])
    ma5_val = float(ma5.iloc[-1])
    ma30_val = float(ma30.iloc[-1])
    bb_mid_val = float(bb_mid.iloc[-1])

    death_cross = ma5_val < ma30_val and current < bb_mid_val
    dovish_signal = death_cross
    rate_cut_probability = 0.80 if dovish_signal else max(0.15, min(0.45, 0.35))

    return {
        "status": "ok",
        "simulated": False,
        "fedwatch_data_source": fedwatch_data_source,
        "ticker": ticker_label or TNX_FEDWATCH_PROXY_TICKER,
        "current_yield": round(current, 4),
        "ma5_yield": round(ma5_val, 4),
        "ma30_yield": round(ma30_val, 4),
        "bollinger_middle": round(bb_mid_val, 4),
        "death_cross": death_cross,
        "dovish_signal": dovish_signal,
        "rate_cut_probability": rate_cut_probability,
        "signal_label": (
            "DOVISH_FED_CUT_BET (Death Cross + below Bollinger mid)"
            if dovish_signal
            else "NEUTRAL_OR_HAWKISH_BIAS"
        ),
        "usd_downside_risk": "ELEVATED" if dovish_signal else "MODERATE",
    }


def _sanitize_spot_series(series: pd.Series) -> pd.Series:
    """Forward/backward-fill yfinance spot paths to block NaN propagation in indicators."""
    cleaned = pd.to_numeric(series, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if cleaned.isna().all():
        return cleaned.fillna(0.0)
    return cleaned


def _sanitize_spot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Harden date/Spot frames (GBPCNH=X and peers) before MA/volatility math."""
    if df is None or df.empty or "Spot" not in df.columns:
        return df
    out = df.copy()
    out["Spot"] = _sanitize_spot_series(out["Spot"])
    if out["Spot"].isna().any():
        out["Spot"] = out["Spot"].ffill().bfill().fillna(0.0)
    if "MA5" in out.columns:
        out["MA5"] = (
            out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
        )
    return out.dropna(subset=["Spot"]).reset_index(drop=True)


def _extract_yfinance_close_series(raw: pd.DataFrame) -> pd.Series:
    """Normalize yfinance OHLCV/MultiIndex frames to a single Close price series."""
    if raw is None or raw.empty:
        raise ValueError("empty yfinance dataframe")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    elif "Close" in raw.columns:
        close = raw["Close"]
    else:
        close = raw.iloc[:, 0]

    series = pd.to_numeric(close, errors="coerce").dropna()
    if series.empty:
        raise ValueError("no valid close prices in yfinance frame")
    return series.sort_index()


def _fetch_tnx_yield_series_yfinance() -> pd.Series:
    """Tier 1 — ^TNX daily closes via yfinance."""
    import yfinance as yf

    raw = yf.download(
        TNX_FEDWATCH_PROXY_TICKER,
        period="3mo",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    return _extract_yfinance_close_series(raw)


def _alphavantage_treasury_payload_error(payload: dict[str, Any]) -> bool:
    """True when Alpha Vantage returns rate-limit or error envelopes instead of yields."""
    if not isinstance(payload, dict):
        return True
    data = payload.get("data")
    if isinstance(data, list) and len(data) >= 10:
        return False
    return any(key in payload for key in ("Note", "Information", "Error Message"))


def _parse_alphavantage_treasury_yield_series(payload: dict[str, Any]) -> pd.Series:
    """Parse TREASURY_YIELD daily 10-year rows into a chronologically sorted yield series."""
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing Alpha Vantage treasury data rows")

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_value = row.get("value")
        raw_date = row.get("date")
        if raw_value in (None, ".", ""):
            continue
        try:
            value = float(raw_value)
            dates.append(pd.to_datetime(raw_date))
            values.append(value)
        except (TypeError, ValueError):
            continue

    if len(values) < 10:
        raise ValueError("insufficient Alpha Vantage treasury yield history")

    series = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    return series


def _fetch_tnx_yield_series_alphavantage() -> pd.Series:
    """Tier 2 — 10Y Treasury daily yields via Alpha Vantage TREASURY_YIELD."""
    api_key = get_secure_alpha_vantage_key()
    if not api_key:
        raise ValueError("Alpha Vantage API key not configured (st.secrets / ALPHA_VANTAGE_KEY)")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TREASURY_YIELD&interval=daily&maturity=10year"
        f"&apikey={api_key}"
    )

    payload = _fetch_json_url(url)
    if payload is None:
        raise ValueError("Alpha Vantage TREASURY_YIELD network timeout or transport error")

    if _alphavantage_treasury_payload_error(payload):
        raise ValueError("Alpha Vantage TREASURY_YIELD payload error or rate limited")

    return _parse_alphavantage_treasury_yield_series(payload)


def _simulate_tnx_fedwatch_proxy() -> dict[str, Any]:
    """Structural simulation for demo continuity when yfinance is unavailable."""
    rng = np.random.default_rng(2026)
    base_yield = 4.15
    trend = np.linspace(0.08, -0.12, 30)
    noise = rng.normal(0, 0.02, 30)
    simulated = pd.Series(base_yield + trend + noise)

    result = _analyze_tnx_yield_series(
        simulated,
        fedwatch_data_source="simulated",
        ticker_label="SIMULATED_10Y_YIELD",
    )
    result["simulated"] = True
    result["rate_cut_probability"] = 0.80
    result["dovish_signal"] = True
    result["death_cross"] = True
    result["signal_label"] = "SIMULATED_DOVISH_FED_CUT_BET (Demo continuity mode)"
    result["usd_downside_risk"] = "ELEVATED"
    return result


def fetch_tnx_fedwatch_proxy() -> dict[str, Any]:
    """
    CME FedWatch cross-asset proxy — 3-tier escalation for US 10Y Treasury yields:
      Tier 1: yfinance ^TNX
      Tier 2: Alpha Vantage TREASURY_YIELD (10year, daily)
      Tier 3: structural simulation (demo continuity)
    Death cross: MA5 < MA30 and spot yield below 20d Bollinger middle → dovish / USD risk-off.
    """
    # Tier 1 — Primary: yfinance ^TNX
    try:
        close = _fetch_tnx_yield_series_yfinance()
        return _analyze_tnx_yield_series(
            close,
            fedwatch_data_source="yfinance",
            ticker_label=TNX_FEDWATCH_PROXY_TICKER,
        )
    except Exception:
        pass

    # Tier 2 — Secondary: Alpha Vantage TREASURY_YIELD
    try:
        close = _fetch_tnx_yield_series_alphavantage()
        return _analyze_tnx_yield_series(
            close,
            fedwatch_data_source="alphavantage",
            ticker_label="TREASURY_YIELD_10Y",
        )
    except Exception:
        pass

    # Tier 3 — Ultimate fallback: structural simulation
    print(
        "Warning: Both yfinance and Alpha Vantage links interrupted. "
        "Entering structural simulation mode.",
        flush=True,
    )
    return _simulate_tnx_fedwatch_proxy()


def build_macro_intelligence_layer(
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
    imf_indicators: dict[str, Any] | None = None,
    *,
    include_imf: bool = True,
) -> str:
    """
    Assemble [REAL-TIME MACRO INTELLIGENCE LAYER].
    China Money lines appear ONLY on successful fetch — zero placeholders on failure.
    IMF structured baseline is always injected when available in the knowledge base.
    """
    china_money = china_money if china_money is not None else fetch_china_money_central_parity()
    tnx_proxy = tnx_proxy if tnx_proxy is not None else fetch_tnx_fedwatch_proxy()

    lines: list[str] = ["[REAL-TIME MACRO INTELLIGENCE LAYER]"]

    if include_imf:
        imf_block = format_imf_macro_indicators_block(imf_indicators)
        if imf_block:
            lines.append(imf_block)

    if china_money.get("status") == "ok" and china_money.get("usd_cny") is not None:
        lines.extend(
            [
                "- **China Money Policy Boundary (CFETS Central Parity)**",
                f"  - USD/CNY official parity: **{float(china_money['usd_cny']):.4f}** "
                f"(fixing date: {china_money.get('parity_date', 'N/A')})",
                f"  - Interpretation: {china_money.get('policy_note', '')}",
            ]
        )

    if tnx_proxy.get("status") == "ok":
        feed = tnx_proxy.get("fedwatch_data_source", "unknown")
        if tnx_proxy.get("simulated"):
            feed_label = " [SIMULATED]"
        elif feed == "alphavantage":
            feed_label = " [Alpha Vantage TREASURY_YIELD 10Y]"
        elif feed == "yfinance":
            feed_label = " [yfinance ^TNX]"
        else:
            feed_label = ""
        ticker = tnx_proxy.get("ticker", TNX_FEDWATCH_PROXY_TICKER)
        lines.extend(
            [
                f"- **CME FedWatch Proxy — US 10Y Treasury ({ticker}){feed_label}**",
                f"  - Current yield: **{tnx_proxy.get('current_yield', 0):.4f}%** | "
                f"MA5: **{tnx_proxy.get('ma5_yield', 0):.4f}%** | "
                f"MA30: **{tnx_proxy.get('ma30_yield', 0):.4f}%** | "
                f"Bollinger mid (20d): **{tnx_proxy.get('bollinger_middle', 0):.4f}%**",
                f"  - Death cross (MA5 < MA30) + below Bollinger mid: **{tnx_proxy.get('death_cross')}**",
                f"  - Signal: **{tnx_proxy.get('signal_label')}** | "
                f"Implied Fed cut probability: **{float(tnx_proxy.get('rate_cut_probability', 0)):.0%}** | "
                f"USD downside risk: **{tnx_proxy.get('usd_downside_risk')}**",
            ]
        )

    if len(lines) == 1:
        return ""

    return "\n".join(lines)


def _rag_retrieved_imf_entry(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return IMF KB entry when RAG retrieval surfaces the macro inflation anchor."""
    for doc in docs:
        if doc.get("id") == IMF_MACRO_KB_ID:
            return doc
    return get_imf_macro_knowledge_entry()


def _build_imf_macro_application_advisory(
    pair: str,
    metrics: dict[str, Any] | None = None,
    macro_indicators: dict[str, Any] | None = None,
) -> str:
    """Desk-facing IMF PPP / GDP / COFER synthesis for long-horizon hedge questions."""
    indicators = macro_indicators
    if indicators is None:
        entry = get_imf_macro_knowledge_entry()
        indicators = (entry or {}).get("macro_indicators")
    if not isinstance(indicators, dict):
        return ""

    spreads = compute_imf_ppp_spreads(pair, indicators)
    cpi = indicators.get("inflation_cpi_yoy", {})
    gdp = indicators.get("gdp_growth_forecast", {})
    cofer = indicators.get("cofer_sovereign_reserves_share", {})

    dest = spreads["destination_label"]
    spread = spreads["cpi_spread_vs_china"]
    ma30 = metrics.get("ma30") if metrics else None
    hedge = metrics.get("recommended_hedge_ratio") if metrics else None

    lines = [
        "**📊 IMF Macro Application (PPP · GDP · COFER)**",
        f"- CPI spread ({dest} vs China): **{spread:+.1f} pp** "
        f"({spreads['destination_cpi']}% vs {spreads['china_cpi']}%) → "
        "PPP implies destination-currency purchasing power erosion on 90–180d receivables; "
        "bias toward strategic forward hedge.",
        f"- GDP velocity: US **{gdp.get('US')}%** vs Eurozone **{gdp.get('Eurozone')}%** "
        f"vs China **{gdp.get('China')}%** → relative capital-flow support for USD over EUR.",
        f"- COFER anchor: USD **{cofer.get('USD')}%** global reserves share → "
        "structural liquidity backstop in tail-risk panics.",
    ]
    if ma30 is not None and hedge is not None:
        lines.append(
            f"- Desk integration ({pair}): MA30 **{float(ma30):.4f}** × "
            f"multi-factor hedge **{float(hedge):.1%}** — align IMF 90–180d budget "
            "with technical overlay."
        )
    return "\n".join(lines)


def generate_dynamic_agent_prompt(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    retrieved_docs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Final brain: aggregate IMF baseline, China Money (success-only), ^TNX Fed proxy, live desk.
    Returns full agent prompt string plus structured macro state for the UI/RAG pipeline.
    """
    user_query = sanitize_user_query(user_query)
    imf_entry = _rag_retrieved_imf_entry(retrieved_docs or [])
    imf_indicators = (imf_entry or {}).get("macro_indicators")
    if not isinstance(imf_indicators, dict):
        imf_entry = get_imf_macro_knowledge_entry()
        imf_indicators = (imf_entry or {}).get("macro_indicators")

    china_money = fetch_china_money_central_parity()
    tnx_proxy = fetch_tnx_fedwatch_proxy()
    macro_layer = build_macro_intelligence_layer(
        china_money, tnx_proxy, imf_indicators=imf_indicators
    )
    live_desk_layer = build_live_market_context_prompt(metrics or {}) if metrics else ""
    system_instructions = build_nova_system_instructions(include_imf_rules=True)

    playbooks_layer = assemble_retrieved_playbooks_context(retrieved_docs)

    sections = [system_instructions]
    if macro_layer:
        sections.append(macro_layer)
    if live_desk_layer:
        sections.append(live_desk_layer)
    if playbooks_layer:
        sections.append(playbooks_layer)
    sections.append(f"User question:\n{user_query}")

    agent_prompt = "\n\n".join(sections)
    active_pair = (metrics or {}).get("pair", DEFAULT_PAIR)

    return {
        "agent_prompt": agent_prompt,
        "system_instructions": system_instructions,
        "macro_layer": macro_layer,
        "imf_macro_block": format_imf_macro_indicators_block(imf_indicators),
        "imf_macro_indicators": imf_indicators if isinstance(imf_indicators, dict) else None,
        "imf_ppp_spreads": compute_imf_ppp_spreads(active_pair, imf_indicators)
        if isinstance(imf_indicators, dict)
        else None,
        "live_desk_layer": live_desk_layer,
        "user_query": user_query,
        "china_money": china_money if china_money.get("status") == "ok" else None,
        "tnx_proxy": tnx_proxy if tnx_proxy.get("status") == "ok" else None,
        "macro_available": bool(macro_layer),
        "imf_retrieved": bool(imf_entry),
        "playbooks_layer": playbooks_layer,
        "retrieved_docs": list(retrieved_docs or []),
    }


_EXPOSURE_MULTIPLIERS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:billion|bn)\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*(?:million|mil|mio|mn)\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*百万", re.IGNORECASE), 1_000_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*万", re.IGNORECASE), 10_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*[kK]\b"), 1_000),
    (re.compile(r"([\d,]+(?:\.\d+)?)\s*m\b(?![a-z])", re.IGNORECASE), 1_000_000),
]


def _parse_notional_exposure(query: str, base_ccy: str) -> float | None:
    """Extract a numeric receivable/payout exposure from natural language."""
    normalized = query.replace(",", "")
    for pattern, multiplier in _EXPOSURE_MULTIPLIERS:
        match = pattern.search(normalized)
        if match:
            return float(match.group(1)) * multiplier

    ccy_pattern = (
        r"(?:usd|eur|hkd|dollar|dollars|euro|euros|"
        + re.escape(base_ccy.lower())
        + r"|\$|€|£)\s*([\d,]+(?:\.\d+)?)"
    )
    match = re.search(ccy_pattern, normalized, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))

    match = re.search(
        r"([\d,]+(?:\.\d+)?)\s*(?:usd|eur|hkd|gbp|dollar|dollars|euro|euros|pound|pounds)\b",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1).replace(",", ""))

    plain = re.search(r"\b([\d,]{4,}(?:\.\d+)?)\b", normalized)
    if plain:
        value = float(plain.group(1).replace(",", ""))
        if value >= 1_000:
            return value
    return None


def _build_execution_advisory(query: str, metrics: dict[str, Any]) -> str:
    """Quantitative execution ticket when user states a numeric exposure."""
    notional = _parse_notional_exposure(query, metrics.get("base_ccy", "USD"))
    if notional is None or notional <= 0:
        return ""

    hedge_ratio = float(metrics.get("recommended_hedge_ratio", 0.55))
    spot = float(metrics.get("spot", 0.0))
    ma5_val = float(metrics.get("ma5", spot))
    base = metrics.get("base_ccy", "USD")
    symbol = metrics.get("symbol", "$")
    pair = metrics.get("pair", DEFAULT_PAIR)
    mf_risk = metrics.get("multi_factor_risk_rating", metrics.get("risk_level", "MEDIUM"))

    lock_notional = notional * hedge_ratio
    residual_notional = notional - lock_notional
    cnh_at_spot_locked = lock_notional * spot
    cnh_at_spot_residual = residual_notional * spot
    cnh_at_spot_total = notional * spot
    effective_rate = (hedge_ratio * ma5_val) + ((1.0 - hedge_ratio) * spot)
    blended_cnh = notional * effective_rate

    return (
        f"**⚡ Real-Time Execution Ticket — {pair}**\n"
        f"- Stated exposure: **{symbol}{notional:,.2f} {base}**\n"
        f"- Multi-factor risk rating: **{mf_risk}** → hedge ratio **{hedge_ratio:.1%}**\n"
        f"- **Forward-lock now:** **{symbol}{lock_notional:,.2f} {base}** "
        f"({hedge_ratio:.1%} of gross exposure)\n"
        f"- **Keep unhedged / spot-flex:** **{symbol}{residual_notional:,.2f} {base}** "
        f"({1.0 - hedge_ratio:.1%} residual)\n"
        f"- Implied CNH at live spot **{spot:.4f}**: locked leg ≈ **¥{cnh_at_spot_locked:,.2f}** | "
        f"residual ≈ **¥{cnh_at_spot_residual:,.2f}** | total ≈ **¥{cnh_at_spot_total:,.2f}**\n"
        f"- NOVA blended effective rate (hedge×MA5 + residual×Spot): **{effective_rate:.4f}** → "
        f"modeled CNH receipt ≈ **¥{blended_cnh:,.2f}**\n"
        f"- **Desk instruction:** Execute a **{hedge_ratio:.0%}** forward/swap on "
        f"**{symbol}{lock_notional:,.2f} {base}** at or better than **{spot:.4f}**; "
        f"retain **{symbol}{residual_notional:,.2f}** for upside if spot mean-reverts below MA5."
    )


# Lightweight cross-lingual bridge: high-frequency Chinese FX terms → English KB tags
_CROSS_LANG_MAPPING: dict[str, list[str]] = {
    "锁汇": ["hedge", "forward", "lock", "locking"],
    "远期": ["forward", "lock", "locking"],
    "对冲": ["hedge", "exposure"],
    "亏损": ["bleeding", "loss"],
    "亚马逊": ["amazon", "seller"],
    "买": ["buy", "payout"],
    "结汇": ["convert", "settlement"],
    "风险": ["risk", "exposure"],
    "合规": ["audit", "compliance", "safe"],
    "宏观": ["macro", "trend", "q2"],
    "汇率": ["macro", "spot", "trend"],
    "港币": ["hkd", "peg", "depreciation"],
    "港元": ["hkd", "peg", "depreciation"],
    "欧元": ["eur", "euro", "vat"],
    "英镑": ["gbp", "uk", "british", "pound", "vat"],
    "美元": ["usd", "forward", "exposure"],
    "结售汇": ["forward", "settlement", "convert"],
    "人民币": ["cnh", "cny", "forward", "settlement", "convert"],
    "离岸人民币": ["cnh", "offshore", "hibor", "squeeze"],
}

_CURRENCY_ATTACH_CODES = ("usd", "eur", "hkd", "gbp", "cnh", "cny")


def _split_attached_currency_codes(text: str) -> str:
    """Fix typos like 'USDnext' / 'EURpayout' → 'usd next' / 'eur payout'."""
    for code in _CURRENCY_ATTACH_CODES:
        text = re.sub(
            rf"\b({code})(?=[a-z])",
            r"\1 ",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _tokenize(text: str) -> set[str]:
    """
    Tokenize with colloquial English expansion, currency typo splitting,
    and Chinese→English semantic bridge for KB alignment.
    """
    normalized = _split_attached_currency_codes(text.lower())
    tokens: set[str] = set()

    for phrase, mapped in sorted(_COLLOQUIAL_ROUTER.items(), key=lambda x: -len(x[0])):
        if phrase in normalized:
            tokens.update(mapped)
            tokens.add(phrase.replace(" ", "_"))

    for zh_term, en_tags in _CROSS_LANG_MAPPING.items():
        if zh_term in text:
            tokens.update(en_tags)

    tokens.update(re.findall(r"[a-z0-9]{2,}", normalized))
    return tokens


def _doc_matches_pair(doc: dict[str, str], pair: str) -> bool:
    applicable = doc.get("pairs", "")
    return pair in applicable or applicable == "ALL"


def _vector_index_ready() -> bool:
    with _rag_lock:
        kb_len = len(_FX_KNOWLEDGE_BASE)
        if kb_len == 0:
            return False
        if len(_indexed_docs_corpus) != kb_len:
            return False
        if FAISS_AVAILABLE and _vector_index is not None:
            return _vector_index.ntotal == kb_len
        return (
            _embedding_matrix is not None
            and _embedding_matrix.shape[0] == kb_len
            and bool(_indexed_docs_corpus)
        )


def _repository_fallback(pair: str, top_k: int) -> list[dict[str, str]]:
    """Safe slice when FAISS index is unavailable or returns no pair-matched hits."""
    fallback = [d for d in _FX_KNOWLEDGE_BASE if _doc_matches_pair(d, pair)]
    return (fallback or _FX_KNOWLEDGE_BASE)[:top_k]


def _compute_sparse_keyword_score(
    query: str,
    q_tokens: set[str],
    doc: dict[str, str],
    pair: str,
) -> float:
    """Sparse track: token overlap, tag hits, domain boosts, currency precision."""
    q_lower = query.lower()
    score = 0.0
    doc_tokens = _tokenize(_doc_corpus(doc))
    tags = doc.get("tags", "")

    overlap = q_tokens & doc_tokens
    score += len(overlap) * 2.0

    for tok in q_tokens:
        if tok in tags:
            score += 3.0

    domain_boosts: dict[str, list[str]] = {
        "hedge": ["forward_locking", "cross_border_case_amazon", "eur_vat_remittance"],
        "forward": ["forward_locking", "policy_compliance"],
        "locking": ["forward_locking", "policy_compliance"],
        "macro": ["q2_2026_macro_usd", "eur_macro_q2"],
        "q2": ["q2_2026_macro_usd", "eur_macro_q2"],
        "amazon": ["cross_border_case_amazon"],
        "tiktok": ["cross_border_case_tiktok"],
        "vat": ["eur_vat_remittance"],
        "euro": ["eur_vat_remittance", "eur_macro_q2"],
        "eur": ["eur_vat_remittance", "eur_macro_q2"],
        "hkd": ["hkd_peg_defense", "hkd_treasury_case"],
        "peg": ["hkd_peg_defense"],
        "depreciation": ["hkd_peg_defense", "hkd_treasury_case"],
        "loss": ["forward_locking", "cross_border_case_amazon", "hkd_treasury_case"],
        "bleeding": ["forward_locking", "cross_border_case_amazon", "hkd_treasury_case"],
    }
    pair_doc_boost = {
        "USD/CNH": ["q2_2026_macro_usd", "cross_border_case_amazon"],
        "EUR/CNH": ["eur_vat_remittance", "eur_macro_q2"],
        "HKD/CNH": ["hkd_peg_defense", "hkd_treasury_case"],
        "GBP/CNH": ["net_payout_rolling_hedge_eu_vat", "high_frequency_micro_payouts"],
    }

    for kw, ids in domain_boosts.items():
        if kw in q_tokens and doc["id"] in ids:
            score += 4.0

    if doc["id"] in pair_doc_boost.get(pair, []):
        score += 2.5

    pair_ccy = _PAIR_CURRENCY_TOKEN.get(pair, "")
    if pair_ccy and pair_ccy in q_tokens:
        if pair_ccy in tags or pair_ccy in doc_tokens:
            score += 2.0

    if "amazon" in q_lower and doc["id"] == "cross_border_case_amazon":
        score += 3.0
    if any(p in q_lower for p in ("losing money", "bleed", "bleeding", "stop bleeding")):
        if doc["id"] in ("forward_locking", "cross_border_case_amazon", "hkd_treasury_case"):
            score += 5.0
    if pair == "EUR/CNH" and any(w in q_lower for w in ("vat", "euro", "europe")):
        if doc["id"] == "eur_vat_remittance":
            score += 6.0
    if pair == "GBP/CNH" and any(w in q_lower for w in ("vat", "uk", "gbp", "british", "pound", "amazon")):
        if doc["id"] == "net_payout_rolling_hedge_eu_vat":
            score += 6.0
    if pair == "HKD/CNH" and any(w in q_lower for w in ("peg", "hkd", "hong kong", "depreciation")):
        if doc["id"] == "hkd_peg_defense":
            score += 6.0
    if any(z in query for z in ("锁汇", "远期", "结售汇")) and doc["id"] == "forward_locking":
        score += 6.0
    if any(z in query for z in ("亏损", "对冲")) and doc["id"] in (
        "forward_locking",
        "cross_border_case_amazon",
    ):
        score += 4.0

    return score


def _ensure_vector_index() -> None:
    """Retry or resize index when corpus and knowledge base lengths diverge."""
    if _FX_KNOWLEDGE_BASE:
        init_vector_index(force_rebuild=not _vector_index_ready())


def _simple_semantic_search(
    query: str,
    pair: str = DEFAULT_PAIR,
    top_k: int = 3,
) -> list[dict[str, str]]:
    """
    Hybrid retriever: dense FAISS cosine (IndexFlatIP + L2 norm) +
    sparse keyword overlap fusion.
    final_score = vector_score + (keyword_score * 0.15)
    """
    query = sanitize_user_query(query)
    if not query:
        return _repository_fallback(pair, top_k)

    _ensure_vector_index()
    if not _vector_index_ready():
        return _repository_fallback(pair, top_k)

    try:
        model = load_embedding_model()
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if FAISS_AVAILABLE:
            faiss.normalize_L2(q_emb)
        else:
            _l2_normalize_inplace(q_emb)

        vector_scores = _dense_vector_scores(q_emb)
        if vector_scores.size == 0:
            return _repository_fallback(pair, top_k)

        q_tokens = _tokenize(query)
        scored: list[tuple[float, dict[str, str]]] = []

        with _rag_lock:
            corpus_snapshot = list(_indexed_docs_corpus)

        for idx, vector_score in enumerate(vector_scores):
            if idx >= len(corpus_snapshot):
                break
            doc = corpus_snapshot[int(idx)]
            if not _doc_matches_pair(doc, pair):
                continue

            keyword_score = _compute_sparse_keyword_score(query, q_tokens, doc, pair)
            final_score = float(vector_score) + keyword_score * 0.15
            scored.append((final_score, doc))

        if not scored:
            return _repository_fallback(pair, top_k)

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]
    except Exception:
        return _repository_fallback(pair, top_k)


def _detect_intent(query: str, tokens: set[str]) -> str:
    q = query.lower()
    if tokens & {"hedge", "forward", "locking", "bleeding", "bleed", "loss"} or any(
        p in q for p in ("losing money", "stop bleeding", "bleed", "lost money")
    ):
        return "hedge"
    if tokens & {"vat", "euro", "eur", "remittance", "uk", "gbp", "british", "pound"}:
        return "eur_vat"
    if tokens & {"hkd", "peg", "depreciation"} or "hong kong" in q:
        return "hkd_peg"
    if tokens & {"macro", "trend", "q2", "seasonal", "june", "cheap"} or "trap" in q:
        return "macro"
    if tokens & {"amazon", "tiktok", "seller", "payout"}:
        return "seller_case"
    if tokens & {"safe", "worry", "worried", "audit", "compliance"}:
        return "compliance"
    return "general"


def _pair_action(intent: str, pair: str) -> str:
    if intent == "eur_vat" or (pair == "EUR/CNH" and intent in ("hedge", "general")):
        return (
            "**EUR / VAT playbook**\n"
            "1. Model net EUR after VAT—not platform gross GMV.\n"
            "2. Forward-hedge 65% of net EUR due in 60–90 days when EUR/CNH is rich.\n"
            "3. Align conversions with OSS/VAT filing dates to avoid double FX hits.\n"
            "4. Keep invoices that satisfy both tax authority and bank FX desks."
        )
    if intent == "hkd_peg" or (pair == "HKD/CNH" and intent in ("hedge", "general")):
        return (
            "**HKD peg-stress playbook**\n"
            "1. Watch HKD/CNH vs 30-day band; weakness → tranche conversions (40/30/30).\n"
            "2. Pre-fund 2 weeks of onshore CNY payroll during HKD weakening phases.\n"
            "3. Use 2–4 week forwards on net CNH shortfall—avoid speculative HKD hoarding.\n"
            "4. Consolidate USD+HKD exposure before adding new hedges."
        )
    if intent == "hedge":
        if pair == "EUR/CNH":
            return (
                "**Executable EUR hedge plan**\n"
                "1. Export 90-day net EUR receivable calendar (post-VAT).\n"
                "2. Forward-sell **65%** of EUR due in 60+ days (3M tenor).\n"
                "3. Keep **35%** spot; add **+10%** if EUR/CNH hits a new 30-day high.\n"
                "4. Log notionals in your exposure ledger for audit."
            )
        if pair == "HKD/CNH":
            return (
                "**Executable HKD hedge plan**\n"
                "1. Map weekly HKD inflows vs CNH/onshore CNY outflows.\n"
                "2. Forward/conversion hedge **50–60%** of 30-day net CNH need.\n"
                "3. Keep **40%** flexible for peg-stabilization bounces.\n"
                "4. Pause discretionary HKD cash piles during weak-HKD headlines."
            )
        if pair == "GBP/CNH":
            return (
                "**Executable GBP hedge plan**\n"
                "1. Export 90-day net GBP receivable calendar (post-UK VAT).\n"
                "2. Forward-sell **65%** of GBP due in 60+ days (3M tenor).\n"
                "3. Keep **35%** spot; add **+10%** if GBP/CNH hits a new 30-day high.\n"
                "4. Log notionals for HMRC + bank FX audit readiness."
            )
        return (
            "**Executable USD hedge plan**\n"
            "1. Export 90-day USD receivable calendar from marketplace reports.\n"
            "2. Forward-sell **70%** of USD due in 60+ days (3M tenor).\n"
            "3. Keep **30%** unhedged; add **+10%** on new 30-day highs.\n"
            "4. Log every hedge line for bank audit readiness."
        )
    if intent == "macro":
        ranges = {
            "USD/CNH": "7.15–7.32",
            "EUR/CNH": "7.70–8.00",
            "HKD/CNH": "0.910–0.940 (peg-guided)",
            "GBP/CNH": "8.75–9.35",
        }
        band = ranges.get(pair, "pair-specific simulation band")
        return (
            f"**Macro playbook ({pair})**\n"
            f"1. Base simulation range **{band}**—ladder forwards, no all-in spot bets.\n"
            "2. Front-load conversions before month-end corporate FX rush.\n"
            "3. Stress-test a shock scenario; add +10% hedge two weeks before large CNH conversion windows."
        )
    if intent == "seller_case":
        return (
            "**Marketplace seller checklist**\n"
            "1. Bucket cash flows by currency and 30/60/90-day payout windows.\n"
            "2. Never run naked float through a promo without a hedge plan.\n"
            "3. Post-campaign: convert 40% spot from day 7; roll the rest in 1M forwards.\n"
            "4. Pause adding forwards if spot < MA5 for 3 sessions."
        )
    if intent == "compliance":
        return (
            "**Safety & compliance steps**\n"
            "1. Hedge only documented open receivables—no speculative notional.\n"
            "2. Bundle invoice + logistics + settlement proof for the bank.\n"
            "3. Refresh KYC files quarterly to avoid payout-day hedge delays."
        )
    return (
        f"**General FX policy ({pair})**\n"
        "1. Quantify exposure and payout dates weekly.\n"
        "2. Map hedge ratio to risk light: HIGH 70% / MEDIUM 40% / LOW 20%.\n"
        "3. Reconcile actual payouts vs hedged notionals every month."
    )


def _build_macro_synthesis_advisory(
    metrics: dict[str, Any],
    china_money: dict[str, Any] | None,
    tnx_proxy: dict[str, Any] | None,
) -> str:
    """Desk-facing synthesis tying macro layers to hedge ratio guidance."""
    lines: list[str] = ["**🌐 Macro × Multi-Factor Synthesis**"]
    spot = metrics.get("spot")
    hedge = metrics.get("recommended_hedge_ratio")
    pair = metrics.get("pair", DEFAULT_PAIR)

    if china_money and china_money.get("usd_cny") is not None and spot is not None:
        parity = float(china_money["usd_cny"])
        spot_f = float(spot)
        deviation_bps = (spot_f - parity) / parity * 10_000
        lines.append(
            f"- China Money USD/CNY parity **{parity:.4f}** vs desk spot **{spot_f:.4f}** "
            f"→ offshore deviation **{deviation_bps:+.0f} bps** "
            f"({'CNH rich vs policy mid' if deviation_bps > 0 else 'CNH cheap vs policy mid'})."
        )

    if tnx_proxy and tnx_proxy.get("status") == "ok":
        if tnx_proxy.get("dovish_signal"):
            lines.append(
                f"- ^TNX dovish structure (death cross + below Bollinger mid) → "
                f"**{float(tnx_proxy.get('rate_cut_probability', 0.8)):.0%}** implied Fed-cut pricing; "
                "favor defensive USD/CNH locks on receivables."
            )
        else:
            lines.append(
                "- ^TNX structure neutral/hawkish → USD supported; keep baseline hedge, "
                "avoid over-locking sub-MA5 spot dips."
            )

    if hedge is not None:
        lines.append(
            f"- NOVA multi-factor overlay on **{pair}**: recommended hedge "
            f"**{float(hedge):.1%}** (integrate with IMF 90d+ playbook when tail risk)."
        )

    return "\n".join(lines) if len(lines) > 1 else ""


def _trim_to_word_budget(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def build_executive_treasury_action(
    metrics: dict[str, Any],
    pair: str,
    *,
    imf_indicators: dict[str, Any] | None = None,
    china_money: dict[str, Any] | None = None,
    tnx_proxy: dict[str, Any] | None = None,
) -> str:
    """Tripartite Data → Analysis → Conclusion advisory aligned with NOVA_AGENT_SYSTEM_INSTRUCTIONS."""
    active_pair = metrics.get("pair", pair)
    spot = float(metrics.get("spot", 0.0))
    ma5 = float(metrics.get("ma5", spot))
    ma30 = float(metrics.get("ma30", spot))
    boll_upper = float(metrics.get("bollinger_upper", spot))
    low_30d = float(metrics.get("low_30d", spot * 0.99))
    high_30d = float(metrics.get("high_30d", spot * 1.01))
    hedge_ratio = float(metrics.get("recommended_hedge_ratio", 0.55))
    rsi = float(metrics.get("net_payout_rsi", 50.0))
    macd_hist = float(metrics.get("net_payout_macd_histogram", 0.0))
    vat_rate = float(metrics.get("vat_deduction_rate", 0.0))
    signal = str(metrics.get("net_payout_signal_class", "normal"))

    if spot > ma5 and spot > ma30:
        corridor = (
            f"Spot **{spot:.4f}** trades **above** the MA5/MA30 corridor "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )
    elif spot < ma30:
        corridor = (
            f"Spot **{spot:.4f}** sits **below** MA30 support "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )
    else:
        corridor = (
            f"Spot **{spot:.4f}** is **inside** the MA5/MA30 corridor "
            f"(MA5 {ma5:.4f}, MA30 {ma30:.4f})"
        )

    spreads = (
        compute_imf_ppp_spreads(active_pair, imf_indicators)
        if isinstance(imf_indicators, dict)
        else compute_imf_ppp_spreads(active_pair)
    )
    dest_label = spreads.get("destination_label", "US")
    macro_bits: list[str] = [f"30-day range **{low_30d:.4f}–{high_30d:.4f}**"]

    if isinstance(imf_indicators, dict):
        gdp = imf_indicators.get("gdp_growth_forecast", {})
        china_gdp = float(gdp.get("China", gdp.get(CHINA_MACRO_ANCHOR, 4.6)))
        dest_gdp = float(gdp.get(dest_label, 2.0))
        cpi_spread = spreads.get("cpi_spread_vs_china")
        macro_bits.append(
            f"IMF 2026 GDP anchor **{dest_label} {dest_gdp:.1f}%** vs China **{china_gdp:.1f}%**"
        )
        if cpi_spread is not None:
            macro_bits.append(f"CPI spread **{cpi_spread:+.1f}pp** vs China")

    if tnx_proxy and tnx_proxy.get("status") == "ok":
        macro_bits.append(
            f"US 10Y **{float(tnx_proxy.get('current_yield', 0.0)):.2f}%** "
            f"({tnx_proxy.get('signal_label', 'yield proxy')})"
        )

    if china_money and china_money.get("status") == "ok" and china_money.get("usd_cny"):
        parity = float(china_money["usd_cny"])
        drift_bps = (spot - parity) / parity * 10_000 if parity else 0.0
        macro_bits.append(
            f"CFETS parity **{parity:.4f}** ({drift_bps:+.0f} bps offshore drift)"
        )

    data_line = f"- {corridor}; {', '.join(macro_bits)}."

    if rsi >= 70:
        momentum = "RSI saturation"
    elif rsi <= 30:
        momentum = "RSI exhaustion"
    else:
        momentum = "neutral RSI momentum"
    macd_tone = "bearish MACD histogram" if macd_hist < 0 else "bullish MACD histogram"
    vat_pct = vat_rate * 100.0

    if signal == "risk_alert":
        payout_impact = (
            f"Platform payout cycles face compressed net CNH conversion as VAT-adjusted "
            f"receivables ({vat_pct:.0f}% withholding overlay) meet fading momentum."
        )
    elif signal == "opportunity":
        payout_impact = (
            f"Merchant micro-revenues retain upside optionality—VAT net payouts ({vat_pct:.0f}%) "
            f"can defer spot clearance without breaching remittance SLAs."
        )
    else:
        payout_impact = (
            f"Cross-border payout cadence remains stable; VAT-adjusted net flows ({vat_pct:.0f}%) "
            f"absorb macro noise without forced spot conversion."
        )

    analysis_line = (
        f"- Sovereign yield/CPI spreads interact with **{momentum}** and **{macd_tone}** "
        f"on net-payout technicals. {payout_impact}"
    )

    if signal == "risk_alert":
        directive = "Layered Window Forward Execution"
        bracket = f"protective floor **{low_30d:.4f}**, Bollinger cap **{boll_upper:.4f}**"
    elif signal == "opportunity":
        directive = "Delayed Discretionary Spot Clearance"
        bracket = f"staged entry band **{low_30d:.4f}–{spot:.4f}**"
    else:
        directive = "Rolling Layered Forward + Meta/Google Ad Spend Asset Matching"
        bracket = f"immunize between **{low_30d:.4f}** and **{high_30d:.4f}**"

    conclusion_line = (
        f"- **{directive}** at **{hedge_ratio:.0%} layered hedge**; {bracket}."
    )

    body = (
        "### ⚡ NOVA Multi-Factor Risk Advisory\n\n"
        "**1. Quantitative Ingestion & Market Signals [DATA]**\n"
        f"{data_line}\n\n"
        "**2. Transmission Channel & Treasury Risk Impact [ANALYSIS]**\n"
        f"{analysis_line}\n\n"
        "**3. Optimized Execution Framework [CONCLUSION]**\n"
        f"{conclusion_line}"
    )
    return _trim_to_word_budget(body, max_words=200)


def _merge_retrieved_docs(
    *doc_lists: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Deduplicate RAG hits by document id while preserving rank order."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for docs in doc_lists:
        for doc in docs:
            doc_id = doc.get("id", doc.get("title", ""))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            merged.append(doc)
            if len(merged) >= top_k:
                return merged
    return merged


def fx_rag_build_response_parts(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    pair: str = DEFAULT_PAIR,
    live_context: str | None = None,
    prompt_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Structured RAG payload for Streamlit expander UI (market-aware copilot)."""
    user_query = sanitize_user_query(user_query)
    m = metrics or {}
    active_pair = m.get("pair", pair)

    docs_pass1 = _simple_semantic_search(user_query, pair=active_pair, top_k=3)
    bundle = prompt_bundle or generate_dynamic_agent_prompt(
        user_query, metrics=m, retrieved_docs=docs_pass1
    )
    macro_block = bundle.get("macro_layer", "")
    live_desk_block = live_context or bundle.get("live_desk_layer", "")
    retrieval_query = bundle["agent_prompt"]

    tokens = _tokenize(retrieval_query)
    docs_pass2 = _simple_semantic_search(retrieval_query, pair=active_pair, top_k=3)
    docs = _merge_retrieved_docs(docs_pass1, docs_pass2, top_k=3)

    playbooks_layer = assemble_retrieved_playbooks_context(docs)
    context_block = bundle["agent_prompt"]
    if playbooks_layer and playbooks_layer not in context_block:
        marker = f"User question:\n{user_query}"
        if marker in context_block:
            context_block = context_block.replace(
                marker,
                f"{playbooks_layer}\n\n{marker}",
                1,
            )
        else:
            context_block = f"{context_block}\n\n{playbooks_layer}"

    spot = m.get("spot")
    ma5 = float(m.get("ma5", spot or 0.0))
    ma30 = float(m.get("ma30", spot or 0.0))
    bollinger_upper = float(m.get("bollinger_upper", spot or 0.0))
    risk = m.get("multi_factor_risk_rating", m.get("risk_level", "MEDIUM"))
    hedge_ratio = float(m.get("recommended_hedge_ratio", 0.55))

    market_block = ""
    if spot is not None:
        market_block = (
            f"---\n**Live desk snapshot — {active_pair}** "
            f"(`{m.get('data_source', 'unknown')}`)\n"
            f"- Real-time Spot: **{float(spot):.4f}** | MA5: **{ma5:.4f}** | "
            f"MA30: **{ma30:.4f}** | Bollinger upper: **{bollinger_upper:.4f}**\n"
            f"- Multi-factor risk: **{risk}** | Hedge ratio: **{hedge_ratio:.1%}** "
            f"| Score: **{float(m.get('multi_factor_score', 0)):.3f}**\n"
            f"- Desk note: {m.get('risk_advice', '')}"
        )

    execution_block = _build_execution_advisory(user_query, m)
    executive_action = build_executive_treasury_action(
        m,
        active_pair,
        imf_indicators=bundle.get("imf_macro_indicators")
        if isinstance(bundle.get("imf_macro_indicators"), dict)
        else None,
        china_money=bundle.get("china_money"),
        tnx_proxy=bundle.get("tnx_proxy"),
    )
    action = executive_action

    imf_retrieved = any(doc.get("id") == IMF_MACRO_KB_ID for doc in docs) or bool(
        bundle.get("imf_macro_indicators")
    )

    disclaimer = (
        "Disclaimer: Educational simulation only—not investment advice. "
        "Execute trades with licensed institutions and verified trade documentation."
    )

    return {
        "docs": docs,
        "user_query": user_query,
        "active_pair": active_pair,
        "agent_prompt": context_block,
        "live_context": live_desk_block,
        "macro_layer": macro_block,
        "macro_available": bundle.get("macro_available", False),
        "imf_macro_block": bundle.get("imf_macro_block", ""),
        "imf_macro_indicators": bundle.get("imf_macro_indicators"),
        "imf_ppp_spreads": bundle.get("imf_ppp_spreads"),
        "imf_retrieved": imf_retrieved,
        "china_money": bundle.get("china_money"),
        "tnx_proxy": bundle.get("tnx_proxy"),
        "market_block": market_block,
        "execution_block": execution_block,
        "executive_action": executive_action,
        "action": action,
        "disclaimer": disclaimer,
        "playbook_count": len(docs),
        "playbooks_layer": playbooks_layer,
    }


def fx_rag_answer(
    user_query: str,
    metrics: dict[str, Any] | None = None,
    pair: str = DEFAULT_PAIR,
) -> str:
    """Backward-compatible markdown answer (playbooks omitted—use expanders in app)."""
    parts = fx_rag_build_response_parts(user_query, metrics=metrics, pair=pair)
    active_pair = parts["active_pair"]

    return (
        f"### NOVA FX Risk Copilot · {active_pair}\n\n"
        f"**Your question:** {parts['user_query']}\n\n"
        f"{parts.get('executive_action', parts['action'])}\n\n"
        f"> {parts['disclaimer']}"
    )


# ---------------------------------------------------------------------------
# Contract / invoice audit — PyPDF-style extraction + regex (local-first)
# ---------------------------------------------------------------------------
_AMOUNT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "labeled_total",
        re.compile(
            r"(?:(?:grand\s+)?total|invoice\s+total|amount\s+due|balance\s+due|"
            r"contract\s+value|notional)[:\s]*(?:USD|EUR|HKD|US\$|\$|€)?\s*"
            r"([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "amount_label",
        re.compile(
            r"(?:amount|sum|value)[:\s]*(?:USD|EUR|HKD|\$|€)?\s*([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "currency_prefix",
        re.compile(
            r"(?:USD|EUR|HKD)\s*([\d,]+(?:\.\d{1,2})?)",
            re.IGNORECASE,
        ),
    ),
    (
        "dollar_sign",
        re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)"),
    ),
]

_TENOR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "due_in_days",
        re.compile(
            r"(?:due|payable|payment|settlement|deliver(?:y)?|paid)\s+(?:in|within)\s+"
            r"(\d{1,3})\s+days?",
            re.IGNORECASE,
        ),
    ),
    (
        "net_days",
        re.compile(r"net\s+(\d{1,3})\s+days?", re.IGNORECASE),
    ),
    (
        "terms_days",
        re.compile(
            r"(?:payment\s+terms|terms)[:\s]*(\d{1,3})\s+days?",
            re.IGNORECASE,
        ),
    ),
]

_MARGIN_PATTERN = re.compile(
    r"(?:gross\s+)?margin[:\s]*(\d{1,2}(?:\.\d{1,2})?)\s*%",
    re.IGNORECASE,
)


def _normalize_amount(raw: str) -> float:
    return float(raw.replace(",", "").strip())


def _extract_pdf_text(file_bytes: bytes) -> str:
    """LangChain PyPDFLoader-style workflow: bytes → PdfReader → page text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages: list[str] = []
        for page in reader.pages:
            chunk = page.extract_text()
            if chunk:
                pages.append(chunk)
        return "\n".join(pages).strip()
    except Exception:
        return ""


def _extract_plaintext(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _extract_document_text(filename: str, file_bytes: bytes | None) -> tuple[str, str]:
    """
    Returns (text, source_kind).
    source_kind: pdf | text | empty
    """
    if not file_bytes:
        return "", "empty"

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf_text(file_bytes), "pdf"
    if name.endswith((".txt", ".csv")):
        return _extract_plaintext(file_bytes), "text"
    return "", "empty"


def _parse_amounts_from_text(text: str) -> list[tuple[str, float, str]]:
    """Returns list of (pattern_name, value, raw_match)."""
    hits: list[tuple[str, float, str]] = []
    for name, pattern in _AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1)
            try:
                value = _normalize_amount(raw)
                if value >= 100:
                    hits.append((name, value, raw))
            except ValueError:
                continue
    return hits


def _parse_tenor_from_text(text: str) -> tuple[int | None, str | None, str | None]:
    """Returns (days, pattern_name, raw_match)."""
    for name, pattern in _TENOR_PATTERNS:
        match = pattern.search(text)
        if match:
            days = int(match.group(1))
            if 1 <= days <= 365:
                return days, name, match.group(0)
    return None, None, None


def _parse_margin_from_text(text: str) -> float | None:
    match = _MARGIN_PATTERN.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _choose_best_amount(hits: list[tuple[str, float, str]]) -> tuple[float | None, str | None, str | None]:
    if not hits:
        return None, None, None
    priority = {"labeled_total": 0, "amount_label": 1, "currency_prefix": 2, "dollar_sign": 3}
    hits.sort(key=lambda h: (priority.get(h[0], 9), -h[1]))
    name, value, raw = hits[0]
    return value, name, raw


def _simulate_audit_payload(name: str, pair: str) -> dict[str, Any]:
    """Deterministic fallback when PDF/regex extraction is unavailable."""
    seed = sum(ord(c) for c in name + pair) % 10_000
    rng = np.random.default_rng(seed)
    meta = PAIR_META[pair]

    notional = float(rng.integers(50_000, 500_000))
    tenor_days = int(rng.integers(30, 120))
    margin_pct = float(rng.integers(8, 18))
    recommended_hedge = min(0.85, 0.45 + notional / 1_000_000)

    return {
        "filename": name,
        "pair": pair,
        "currency": meta["base_ccy"],
        "symbol": meta["symbol"],
        "notional": notional,
        "expected_settlement_days": tenor_days,
        "gross_margin_pct": margin_pct,
        "recommended_forward_ratio": round(recommended_hedge, 2),
        "estimated_cnh_at_risk": round(notional * 0.03, 2),
        "status": "PASS",
        "extraction_mode": "simulated_fallback",
        "document_chars": 0,
        "text_preview": "",
        "regex_hits": {},
    }


def _parse_contract_from_text(text: str, pair: str) -> dict[str, Any] | None:
    """Regex extraction layer; returns None if mandatory fields are missing."""
    amount_hits = _parse_amounts_from_text(text)
    notional, amount_rule, amount_raw = _choose_best_amount(amount_hits)
    tenor_days, tenor_rule, tenor_raw = _parse_tenor_from_text(text)

    if notional is None or tenor_days is None:
        return None

    meta = PAIR_META[pair]
    margin_pct = _parse_margin_from_text(text)
    if margin_pct is None:
        margin_pct = float(min(25.0, max(8.0, 12.0 + (notional % 7))))

    recommended_hedge = min(0.85, 0.45 + notional / 1_000_000)

    return {
        "filename": "",
        "pair": pair,
        "currency": meta["base_ccy"],
        "symbol": meta["symbol"],
        "notional": notional,
        "expected_settlement_days": tenor_days,
        "gross_margin_pct": margin_pct,
        "recommended_forward_ratio": round(recommended_hedge, 2),
        "estimated_cnh_at_risk": round(notional * 0.03, 2),
        "status": "PASS",
        "extraction_mode": "pdf_regex",
        "document_chars": len(text),
        "text_preview": text[:280].replace("\n", " "),
        "regex_hits": {
            "amount_rule": amount_rule,
            "amount_raw": amount_raw,
            "tenor_rule": tenor_rule,
            "tenor_raw": tenor_raw,
            "amount_candidates": len(amount_hits),
        },
    }


def audit_contract_file(
    filename: str,
    file_bytes: bytes | None = None,
    pair: str = DEFAULT_PAIR,
) -> dict[str, Any]:
    """
    Audit uploaded contract/invoice:
    1) PDF → pypdf text extraction (PyPDFLoader-style)
    2) Regex for notional + tenor (+ margin when present)
    3) Fallback to deterministic simulation if needed
    """
    name = filename or "contract.pdf"
    sym = PAIR_META[pair]["symbol"]
    base = PAIR_META[pair]["base_ccy"]

    text, source_kind = _extract_document_text(name, file_bytes)
    parsed: dict[str, Any] | None = None

    if text:
        parsed = _parse_contract_from_text(text, pair)
        if parsed is not None:
            parsed["filename"] = name
            if source_kind == "pdf":
                parsed["extraction_mode"] = "pdf_regex"
            else:
                parsed["extraction_mode"] = "text_regex"

    if parsed is None:
        parsed = _simulate_audit_payload(name, pair)
        if text and source_kind == "pdf":
            parsed["extraction_mode"] = "pdf_fallback_simulated"
            parsed["document_chars"] = len(text)
            parsed["text_preview"] = text[:280].replace("\n", " ")
            parsed["regex_hits"] = {
                "note": "PDF text extracted but no amount/tenor regex match",
            }
        elif text and source_kind == "text":
            parsed["extraction_mode"] = "text_fallback_simulated"
            parsed["document_chars"] = len(text)
            parsed["text_preview"] = text[:280].replace("\n", " ")

    notional = parsed["notional"]
    tenor_days = parsed["expected_settlement_days"]
    hedge = parsed["recommended_forward_ratio"]
    mode = parsed["extraction_mode"]

    if mode in ("pdf_regex", "text_regex"):
        hits = parsed.get("regex_hits", {})
        parsed["summary"] = (
            f"**Document read successfully** — extracted **{parsed['document_chars']:,}** "
            f"characters from **{name}** ({mode}).\n\n"
            f"- **Amount detected** via `{hits.get('amount_rule')}`: "
            f"**{hits.get('amount_raw')}** → {base} **{sym}{notional:,.2f}**\n"
            f"- **Tenor detected** via `{hits.get('tenor_rule')}`: "
            f"\"{hits.get('tenor_raw')}\" → **{tenor_days} days**\n"
            f"- Recommended forward hedge: **{hedge:.0%}** | "
            f"CNH at risk (sim.): **¥{parsed['estimated_cnh_at_risk']:,.0f}**\n\n"
            f"_Preview:_ \"{parsed.get('text_preview', '')}…\""
        )
    elif mode == "pdf_fallback_simulated":
        parsed["summary"] = (
            f"**PDF parsed ({parsed.get('document_chars', 0):,} chars)** from **{name}**, "
            f"but no confident amount/tenor regex match — using **deterministic fallback**.\n\n"
            f"- Fallback {base} exposure: **{sym}{notional:,.0f}** | Tenor: **{tenor_days} days** | "
            f"Hedge: **{hedge:.0%}**\n\n"
            f"_Preview:_ \"{parsed.get('text_preview', '')}…\""
        )
    else:
        parsed["summary"] = (
            f"**Simulation mode** for **{name}** ({pair}) — non-PDF or empty upload; "
            f"deterministic seed-based audit.\n\n"
            f"- {base} exposure: **{sym}{notional:,.0f}** | Tenor: **{tenor_days} days** | "
            f"Hedge: **{hedge:.0%}** | CNH at risk: **¥{parsed['estimated_cnh_at_risk']:,.0f}**"
        )

    return parsed


# ---------------------------------------------------------------------------
# Portfolio Risk & Quant Performance Evaluation Engine
# ---------------------------------------------------------------------------
TRADING_DAYS = 365
RISK_FREE_ANNUAL = 0.02


def _generate_fallback_spot_data(days: int, pair: str = DEFAULT_PAIR) -> pd.DataFrame:
    """Pair-aware Spot/MA5 fallback via the legacy simulator."""
    kline = generate_fx_kline(pair=pair, days=days)
    calendar = _calendar_date_index(days)
    df = pd.DataFrame(
        {
            "date": calendar,
            "Spot": kline["close"].values,
            "MA5": kline["ma5"].values,
        }
    )
    df.attrs["data_source"] = "simulated_fallback"
    df.attrs["ticker"] = YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")
    df.attrs["pair"] = pair
    return df


def _generate_fallback_backtest_data(days: int = 180, pair: str = DEFAULT_PAIR) -> pd.DataFrame:
    """Deterministic random-walk fallback when live market feeds are unavailable."""
    return _generate_fallback_spot_data(days=days, pair=pair)


def _alpha_vantage_payload_error(payload: dict[str, Any]) -> bool:
    """True when Alpha Vantage returns rate-limit or error envelopes instead of OHLC."""
    if not isinstance(payload, dict):
        return True
    if payload.get("Time Series FX (Daily)"):
        return False
    return any(key in payload for key in ("Note", "Information", "Error Message"))


def _fetch_alpha_vantage_spot_data(
    days: int,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Ingest Alpha Vantage FX_DAILY close series for a supported pair."""
    api_key = get_secure_alpha_vantage_key()
    if not api_key:
        return None

    from_ccy, to_ccy = ALPHA_VANTAGE_PAIR_MAP[pair]
    url = (
        "https://www.alphavantage.co/query"
        f"?function=FX_DAILY&from_symbol={from_ccy}&to_symbol={to_ccy}"
        f"&apikey={api_key}"
    )

    payload = _fetch_json_url(url)
    if payload is None:
        return None

    if _alpha_vantage_payload_error(payload):
        return None

    series = payload.get("Time Series FX (Daily)")
    if not isinstance(series, dict) or not series:
        return None

    rows: list[dict[str, Any]] = []
    for date_str, bar in series.items():
        if not isinstance(bar, dict):
            continue
        close_raw = bar.get("4. close")
        if close_raw is None:
            continue
        spot = pd.to_numeric(close_raw, errors="coerce")
        if pd.isna(spot):
            continue
        rows.append({"date": pd.to_datetime(date_str), "Spot": float(spot)})

    if not rows:
        return None

    out = pd.DataFrame(rows).dropna(subset=["Spot"]).sort_values("date")
    if len(out) < min_rows:
        return None

    out["Spot"] = out["Spot"].round(4)
    out["MA5"] = out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
    out = _sanitize_spot_dataframe(out.tail(days).reset_index(drop=True))
    out.attrs["data_source"] = "Alpha Vantage"
    out.attrs["ticker"] = f"{from_ccy}/{to_ccy}"
    out.attrs["pair"] = pair
    return out


def _fetch_yfinance_spot_data(
    days: int,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Backup live ingestion via Yahoo Finance when Alpha Vantage is unavailable."""
    ticker = YFINANCE_PAIR_TICKERS.get(pair, "USDCNH=X")

    try:
        import yfinance as yf

        yf_ticker = yf.Ticker(ticker)
        for period in ("1y", "6mo", "3mo", f"{max(days, 30)}d"):
            hist = yf_ticker.history(period=period, interval="1d", auto_adjust=True)
            parsed = _normalize_yfinance_frame(
                hist, days, ticker=ticker, pair=pair, min_rows=min_rows
            )
            if parsed is not None:
                return parsed

        raw = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        parsed = _normalize_yfinance_frame(
            raw, days, ticker=ticker, pair=pair, min_rows=min_rows
        )
        if parsed is not None:
            return parsed

        end = datetime.now()
        start = end - pd.Timedelta(days=int(days * 1.5))
        raw = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        return _normalize_yfinance_frame(
            raw, days, ticker=ticker, pair=pair, min_rows=min_rows
        )
    except Exception:
        return None


def _extract_close_series(raw: pd.DataFrame) -> pd.Series | None:
    """Extract Close prices from flat or MultiIndex yfinance frames."""
    if raw is None or raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if str(col[0]).lower() == "close":
                series = raw[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                return series
        return None

    for name in ("Close", "close", "Adj Close"):
        if name in raw.columns:
            series = raw[name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            return series
    return None


def _normalize_yfinance_frame(
    raw: pd.DataFrame,
    days: int,
    ticker: str,
    pair: str,
    min_rows: int = 10,
) -> pd.DataFrame | None:
    """Parse yfinance OHLCV into date / Spot / MA5."""
    close = _extract_close_series(raw)
    if close is None:
        return None

    if isinstance(raw.index, pd.DatetimeIndex):
        dates = pd.to_datetime(raw.index)
    elif "Date" in raw.columns:
        dates = pd.to_datetime(raw["Date"])
    elif "date" in raw.columns:
        dates = pd.to_datetime(raw["date"])
    else:
        return None

    if isinstance(dates, pd.DatetimeIndex) and dates.tz is not None:
        dates = dates.tz_localize(None)

    spot_series = _sanitize_spot_series(pd.Series(close.values, dtype=float))
    out = pd.DataFrame({"date": dates, "Spot": spot_series.values})
    out = out.dropna(subset=["Spot"]).sort_values("date")
    if len(out) < min_rows:
        return None

    out["Spot"] = out["Spot"].round(4)
    out["MA5"] = out["Spot"].rolling(window=5, min_periods=1).mean().ffill().bfill()
    out = _sanitize_spot_dataframe(out.tail(days).reset_index(drop=True))
    if out is None or len(out) < min_rows:
        return None
    out.attrs["data_source"] = "yfinance"
    out.attrs["ticker"] = ticker
    out.attrs["pair"] = pair
    return out


def fetch_real_market_data(
    days: int = 180,
    pair: str = DEFAULT_PAIR,
) -> pd.DataFrame:
    """
    Market data pipeline: Alpha Vantage FX_DAILY → yfinance backup → simulator fallback.
    Returns date / Spot / MA5 for all SUPPORTED_PAIRS (incl. GBP/CNH).
    """
    if pair not in PAIR_META:
        raise ValueError(f"Unsupported pair: {pair}. Choose from {SUPPORTED_PAIRS}")

    min_rows = max(5, min(20, days // 2))

    alpha_df = _fetch_alpha_vantage_spot_data(days=days, pair=pair, min_rows=min_rows)
    if alpha_df is not None:
        return alpha_df

    yfinance_df = _fetch_yfinance_spot_data(days=days, pair=pair, min_rows=min_rows)
    if yfinance_df is not None:
        return yfinance_df

    return _generate_fallback_backtest_data(days=days, pair=pair)


def generate_backtest_data(days: int = 180) -> pd.DataFrame:
    """Backward-compatible alias for USD/CNH backtest ingestion."""
    return fetch_real_market_data(days=days, pair="USD/CNH")


def _annualized_volatility(daily_returns: pd.Series) -> float:
    std = daily_returns.std(ddof=1)
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    return float(std * np.sqrt(TRADING_DAYS))


def _historical_var_95(daily_cash_flows: pd.Series) -> float:
    """95% Historical VaR: 5th percentile of daily absolute CNH cash flows."""
    clean = daily_cash_flows.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0
    return float(np.percentile(clean.values, 5))


def _maximum_drawdown(cumulative_equity: pd.Series) -> float:
    """Peak-to-trough decline on cumulative equity curve (positive magnitude)."""
    clean = cumulative_equity.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if clean.empty:
        return 0.0
    running_peak = clean.cummax()
    drawdown = (clean - running_peak) / running_peak.replace(0, np.nan)
    drawdown = drawdown.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return float(abs(drawdown.min()))


def _sharpe_ratio(daily_returns: pd.Series, rf_daily: float) -> float:
    std = daily_returns.std(ddof=1)
    if std is None or np.isnan(std) or std == 0:
        return 0.0
    excess = daily_returns.mean() - rf_daily
    return float(excess / std * np.sqrt(TRADING_DAYS))


def _compute_strategy_metrics(
    daily_cnh: pd.Series,
    daily_returns: pd.Series,
    rf_daily: float,
) -> dict[str, Any]:
    cumulative = daily_cnh.cumsum()
    wealth_index = (1.0 + daily_returns).cumprod()
    return {
        "daily_cnh": daily_cnh,
        "daily_returns": daily_returns,
        "cumulative_equity": cumulative,
        "wealth_index": wealth_index,
        "total_value": float(cumulative.iloc[-1]) if len(cumulative) else 0.0,
        "annualized_volatility": _annualized_volatility(daily_returns),
        "var_95": _historical_var_95(daily_cnh),
        "max_drawdown": _maximum_drawdown(cumulative),
        "sharpe_ratio": _sharpe_ratio(daily_returns, rf_daily),
    }


def _compute_multi_factor_hedge_ratios(
    spot: pd.Series,
    ma5: pd.Series | None = None,
) -> np.ndarray:
    """
    Three-factor integrated hedge model (weights 40% / 30% / 30%):
    F1 trend (Spot vs MA5), F2 vol (Spot vs 20d Bollinger upper), F3 macro (Spot vs MA30).
    Score → hedge: ≥0.7 → 80%, [0.4, 0.7) → 55%, <0.4 → 30%.
    """
    return _multi_factor_series(spot, ma5)["hedge_ratio"]


def advanced_quantitative_analysis(
    df_history: pd.DataFrame,
    monthly_revenue: float = 100_000,
) -> dict[str, Any]:
    """
    Compare Unhedged Spot vs NOVA Multi-Factor Dynamic Overlay.
    Factors: MA5 trend, 20d Bollinger upper-band stretch, 30d macro deviation.
    Returns metrics, equity curves, and timelines for dashboard rendering.
    """
    df = _sanitize_spot_dataframe(df_history.copy())
    spot = _sanitize_spot_series(df["Spot"].astype(float))
    ma5 = (
        _sanitize_spot_series(df["MA5"].astype(float))
        if "MA5" in df.columns
        else spot.rolling(window=5, min_periods=1).mean().ffill().bfill()
    )
    daily_usd = monthly_revenue / 30.0
    rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS

    # Strategy A — unhedged spot conversion
    benchmark_cnh = daily_usd * spot
    benchmark_returns = benchmark_cnh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    # Strategy B — NOVA multi-factor dynamic overlay
    hedge_ratio = _compute_multi_factor_hedge_ratios(spot, ma5)
    effective_rate = (hedge_ratio * ma5) + ((1.0 - hedge_ratio) * spot)
    nova_cnh = daily_usd * effective_rate
    nova_returns = nova_cnh.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)

    benchmark = _compute_strategy_metrics(benchmark_cnh, benchmark_returns, rf_daily)
    nova = _compute_strategy_metrics(nova_cnh, nova_returns, rf_daily)

    timeline = df["date"].tolist() if "date" in df.columns else list(range(len(df)))
    data_source = df.attrs.get("data_source", "unknown")

    return {
        "pair": "USD/CNH",
        "data_source": data_source,
        "ticker": df.attrs.get("ticker", YFINANCE_PAIR_TICKERS.get("USD/CNH", "USDCNH=X")),
        "observation_days": len(df),
        "monthly_revenue_usd": monthly_revenue,
        "daily_usd_exposure": daily_usd,
        "timeline": timeline,
        "hedge_ratio_series": hedge_ratio.tolist(),
        "benchmark": {
            **benchmark,
            "label": "Unhedged Spot Benchmark",
            "daily_cnh": benchmark["daily_cnh"].tolist(),
            "daily_returns": benchmark["daily_returns"].tolist(),
            "cumulative_equity": benchmark["cumulative_equity"].tolist(),
        },
        "nova": {
            **nova,
            "label": "NOVA Dynamic Overlay",
            "daily_cnh": nova["daily_cnh"].tolist(),
            "daily_returns": nova["daily_returns"].tolist(),
            "cumulative_equity": nova["cumulative_equity"].tolist(),
        },
        "metrics_comparison": {
            "total_value": {
                "benchmark": benchmark["total_value"],
                "nova": nova["total_value"],
            },
            "annualized_volatility": {
                "benchmark": benchmark["annualized_volatility"],
                "nova": nova["annualized_volatility"],
            },
            "var_95": {
                "benchmark": benchmark["var_95"],
                "nova": nova["var_95"],
            },
            "max_drawdown": {
                "benchmark": benchmark["max_drawdown"],
                "nova": nova["max_drawdown"],
            },
            "sharpe_ratio": {
                "benchmark": benchmark["sharpe_ratio"],
                "nova": nova["sharpe_ratio"],
            },
        },
    }
