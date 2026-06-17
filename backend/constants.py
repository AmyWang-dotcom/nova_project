"""Static constants and configuration for NOVA FX backend."""
from __future__ import annotations

import re
from typing import Any

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
OFFSHORE_QUOTE_CCY = "CNH"
ONSHORE_QUOTE_CCY = "CNY"
CHINA_MACRO_ANCHOR = "China"
VAT_MAP: dict[str, float] = {
    "USD/CNH": 0.0,
    "EUR/CNH": 0.20,
    "GBP/CNH": 0.20,
    "HKD/CNH": 0.0,
}
VAT_STANDARD_DEDUCTION_RATE = VAT_MAP["EUR/CNH"]
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
NOVA_RISK_WEIGHTS: dict[str, float] = {
    "market": 0.35,
    "macro": 0.25,
    "policy": 0.25,
    "knowledge": 0.15,
}
_PAIR_RESERVE_CCY: dict[str, str] = {
    "USD/CNH": "USD",
    "EUR/CNH": "EUR",
    "GBP/CNH": "GBP",
    "HKD/CNH": "USD",
}
_KNOWLEDGE_DOMAIN_BASE_SCORES: dict[str, int] = {
    "Exposure Risk": 20,
    "Liquidity Risk": 15,
    "Compliance Risk": 20,
    "Macro Event Risk": 15,
    "Geopolitical / Supply Chain Risk": 15,
    "Treasury Operational Risk": 15,
}
_KNOWLEDGE_MITIGATION_FRAMEWORK_STRENGTHS: dict[str, int] = {
    "Natural Hedging": 10,
    "Forward Contract": 12,
    "Window Forward": 12,
    "Rolling Forward": 12,
    "Currency Option": 8,
    "Zero-Cost Collar": 10,
    "Government Guarantee": 15,
    "Multi-Currency Netting": 8,
}
_KNOWLEDGE_RISK_DOMAIN_RULES = _KNOWLEDGE_DOMAIN_BASE_SCORES
_KNOWLEDGE_MITIGATION_RULES = _KNOWLEDGE_MITIGATION_FRAMEWORK_STRENGTHS
_KNOWLEDGE_SIGNAL_WEIGHT = 3
_KNOWLEDGE_DOMAIN_REASONING: dict[str, str] = {
    "Exposure Risk": (
        "Large foreign-currency receivable or payout concentration increases "
        "earnings volatility when spot rates move against the enterprise."
    ),
    "Liquidity Risk": (
        "Receipts arrive later than payment obligations, squeezing working capital "
        "and forcing adverse spot conversions."
    ),
    "Compliance Risk": (
        "Regulatory settlement, AML, or trade-background gaps can freeze cash flow "
        "or invalidate planned FX conversions."
    ),
    "Macro Event Risk": (
        "Central-bank or macro data shocks widen spreads and create gap risk on "
        "unhedged settlement windows."
    ),
    "Geopolitical / Supply Chain Risk": (
        "Freight, logistics, or geopolitical shocks compress margins while raising "
        "immediate spot FX funding needs."
    ),
    "Treasury Operational Risk": (
        "Hedge sizing, delivery timing, or accounting mismatches can turn protection "
        "into delivery default or over-hedge loss."
    ),
}
_KNOWLEDGE_MITIGATION_REASONING: dict[str, str] = {
    "Natural Hedging": "Foreign-currency inflows partially offset foreign-currency outflows.",
    "Forward Contract": "Locks a future FX conversion rate against adverse spot moves.",
    "Window Forward": "Preserves delivery flexibility when payout timing is uncertain.",
    "Rolling Forward": "Layers staggered forward coverage across recurring payout cycles.",
    "Currency Option": "Establishes a downside floor while preserving upside participation.",
    "Zero-Cost Collar": "Creates a bounded conversion corridor with zero upfront premium.",
    "Government Guarantee": "Backstops bank margin requirements to preserve operating liquidity.",
    "Multi-Currency Netting": "Matches receivables and payables in the same currency to cut conversion friction.",
}
_LEGACY_DOMAIN_KEYWORD_PATTERNS: dict[str, tuple[str, ...]] = {
    "Exposure Risk": (
        "receivable",
        "exposure",
        "net payout",
        "gross sales",
        "payout funnel",
        "usd payout",
        "export enterprise",
    ),
    "Liquidity Risk": (
        "liquidity",
        "margin call",
        "working capital",
        "cash flow timing",
        "roll-over",
        "rollover",
    ),
    "Compliance Risk": (
        "safe",
        "compliance",
        "regulatory",
        "aml",
        "audit",
        "three-way matching",
    ),
    "Macro Event Risk": (
        "fomc",
        "nfp",
        "macro",
        "imf",
        "central bank",
        "cpi release",
    ),
    "Geopolitical / Supply Chain Risk": (
        "freight",
        "supply chain",
        "logistics",
        "geopolitical",
        "trade war",
    ),
    "Treasury Operational Risk": (
        "over-hedge",
        "settlement mismatch",
        "timing mismatch",
        "physical delivery",
        "hedge accounting",
    ),
}
_LEGACY_MITIGATION_KEYWORD_PATTERNS: dict[str, tuple[str, ...]] = {
    "Natural Hedging": ("natural hedge", "natural hedging", "natural offset"),
    "Forward Contract": ("forward contract", "rolling forward", "forward sleeve"),
    "Window Forward": ("window forward", "delivery window", "flexible delivery"),
    "Rolling Forward": ("rolling forward", "rolling hedge"),
    "Currency Option": ("currency option", "fx option", "put option", "one-yuan"),
    "Zero-Cost Collar": ("zero-cost collar", "collar structure", "range forward"),
    "Government Guarantee": ("government guarantee", "government-backed", "safe harbor"),
    "Multi-Currency Netting": ("multi-currency netting", "netting", "world account"),
}
_SCORE_AMPLIFIER_TIERS: tuple[tuple[float, float], ...] = (
    (85.0, 15.0),
    (75.0, 10.0),
    (65.0, 5.0),
)
_EXPORT_ENTITY_MARKERS: tuple[str, ...] = (
    "exporter",
    "export",
    "cross-border ecommerce",
    "e-commerce",
    "seller",
    "amazon",
    "micro-merchant",
)
_IMPORT_ENTITY_MARKERS: tuple[str, ...] = (
    "importer",
    "import",
    "payable",
    "supplier",
    "factory payroll",
    "supply chain liability",
)
