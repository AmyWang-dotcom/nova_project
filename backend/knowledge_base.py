"""FX knowledge base corpus and IMF macro metadata."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .constants import (
    CHINA_MACRO_ANCHOR,
    DEFAULT_PAIR,
    IMF_MACRO_KB_ID,
    ONSHORE_QUOTE_CCY,
    _KNOWLEDGE_DOMAIN_BASE_SCORES,
    _KNOWLEDGE_MITIGATION_FRAMEWORK_STRENGTHS,
)
from .utils import _cofer_reserve_pct, _fmt_cofer_reserve_label

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
        },
        "tags": "amazon,us,usd,payout,funnel,deduction,referral fee,fba,advertising,net payout,14-day,dd+7,forward,over-hedge"
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
        },
        "tags": "amazon,eu,euro,gbp,vat,marketplace facilitator,accs,currency conversion,multi-currency wallet,window forward,21-day,28-day,reserve"
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
        ),
        "tags": "shopify,independent store,stripe,paypal,rolling reserve,gateway,settlement,markup,natural hedge,forward,90-day"
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
        ),
        "tags": "hong kong,offshore,lers,peg,usd/hkd,cips,three-way matching,settlement,cnh,aggregation,multi-currency,compliance"
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
        },
        "tags": "furniture,heavy industry,long-cycle,supply chain,cash conversion,90-day,120-day,fixed forward,window forward,delivery delay,liquidity,3pl,fba,rollover"
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
        },
        "tags": "fast fashion,apparel,high return rate,return logistics,phantom exposure,eur,forward,over-hedge,decay coefficient,reverse logistics,consumer protection"
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
        },
        "tags": "amazon,uk,gbp,vat,hmrc,marketplace facilitator,14-day,dd+7,window forward,over-hedge,bank of england,boe,interest rate,sterling"
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
            "you must passively settle your forex at that capped price. When recommending this plan, the system must ensure the option's maturity date "
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
        },
        "tags": "zero-cost collar,option,put,call,premium neutral,floor strike,cap strike,range forward,otm,hedging,derivatives"
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
        },
        "tags": "fx swap,swap points,covered interest parity,cip,sofr,sonia,liquidity,multi-currency,spot,forward,intercompany,interest rate differential"
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
        },
        "tags": "roll-over,early delivery,forward contract,delivery window,swap points,liquidity,supply chain,3pl,customs,default risk,flexible treasury"
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
        },
        "tags": "fomc,nfp,federal reserve,volatility,event risk,hedge ratio,otm put option,dxy,interest rate,slippage"
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
        },
        "tags": "ecb,boe,monetary policy divergence,eur/gbp,interest rate,correlation,forward contract,otm collar"
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
        },
        "tags": "pboc,counter-cyclical factor,cnh,liquidity squeeze,cfets,fixing,hibor,forward swap,intervention"
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
        },
        "tags": "geopolitical,tail risk,freight,scfi,gpr,supply chain,liquidity buffer,window forward,otm put"
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
        },
        "tags": "boj,yen,carry trade unwind,ycc,liquidity shock,forward hedge,otm put,tonar"
    },
    {
        "id": "early_stage_seller",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH",
        "domain": "Seller Lifecycle",
        "title": "Bootstrapper Liquidity Constraint",
        "professional_definition": (
            "Liquidity Constraint Risk: Early-stage sellers prioritize operational cash flow "
            "over hedging complexity."
        ),
        "layman_translation": (
            "For newly established sellers, every dollar is critical for inventory replenishment. "
            "Entering complex hedging arrangements too early may lock up working capital and create "
            "inventory shortages caused by cash-flow constraints."
        ),
        "actionable_playbook": (
            "Platform Context: Marketplace payout cycles and reserve requirements may delay access "
            "to cash proceeds, reducing available liquidity for daily operations. "
            "Treasury Copilot Recommendation: Maintain a high-liquidity operating buffer, prioritize "
            "spot conversions for inventory purchases, and avoid long-dated hedging structures when "
            "working capital remains the primary operational constraint."
        ),
        "data_sources": {
            "payout_reserve": "https://sellercentral.amazon.com/help/hub/reference/G200185030"
        },
        "tags": "early stage,bootstrapper,liquidity,working capital,inventory,spot conversion"
    },
    {
        "id": "growth_seller",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH",
        "domain": "Seller Lifecycle",
        "title": "Growth Stage Ad-Spend Exposure",
        "professional_definition": (
            "Operational Margin Risk: Scaling advertising spend creates a net settlement mismatch "
            "that compounds with exchange-rate volatility."
        ),
        "layman_translation": (
            "As sellers scale, advertising becomes a major operating expense deducted before funds "
            "are paid out. Ignoring this deduction can cause a significant gap between expected "
            "profitability and realized FX-adjusted earnings."
        ),
        "actionable_playbook": (
            "Platform Context: Advertising expenses are often deducted directly from marketplace "
            "balances before payout calculations are finalized. "
            "Treasury Copilot Recommendation: Evaluate profitability based on net settlement value "
            "after advertising expenses and consider rolling hedge structures to smooth earnings volatility."
        ),
        "data_sources": {
            "ad_billing": "https://advertising.amazon.com/help"
        },
        "tags": "growth,advertising spend,net settlement,rolling hedge,margin"
    },
    {
        "id": "scaling_seller",
        "pairs": "USD/CNH,EUR/CNH,GBP/CNH",
        "domain": "Seller Lifecycle",
        "title": "Multi-Currency Settlement Complexity",
        "professional_definition": (
            "Treasury Coordination Risk: Scaling across multiple marketplaces requires orchestration "
            "of non-synchronized currency settlement balances."
        ),
        "layman_translation": (
            "Large-scale sellers often receive payments in multiple currencies. Because settlement "
            "timelines differ across marketplaces, immediately converting every payout may reduce "
            "flexibility and create inefficient treasury management."
        ),
        "actionable_playbook": (
            "Platform Context: Multi-marketplace sellers receive balances across different currencies "
            "and settlement cycles. Currency conversion timing therefore becomes a treasury decision "
            "rather than a simple operational process. "
            "Treasury Copilot Recommendation: Monitor regional settlement balances and coordinate "
            "conversion timing across marketplaces to improve overall FX efficiency."
        ),
        "data_sources": {
            "global_selling": "https://sellercentral.amazon.com/gp/help/G200336920"
        },
        "tags": "scaling,multi-currency,settlement cycle,currency conversion,treasury coordination"
    },
    {
        "id": "settlement_delay",
        "pairs": "USD/CNH",
        "domain": "Marketplace Payout Risk",
        "title": "Settlement Timing Risk",
        "professional_definition": (
            "Settlement Timing Risk: Compliance reviews or verification triggers create a temporal "
            "mismatch between delivery and cash realization."
        ),
        "layman_translation": (
            "When compliance reviews or verification procedures delay payouts, funds may not arrive "
            "when expected. If hedging contracts mature before payouts are received, liquidity pressure "
            "can emerge."
        ),
        "actionable_playbook": (
            "Platform Context: Marketplace payout processing may be delayed by account reviews, "
            "verification procedures, or other operational restrictions. "
            "Treasury Copilot Recommendation: Establish a settlement buffer window and execute "
            "hedging decisions only after funds enter a reasonably predictable settlement stage."
        ),
        "data_sources": {
            "unavailable_balance": "https://sellercentral.amazon.com/help/hub/reference/G200185030"
        },
        "tags": "settlement delay,payout timing,compliance review,liquidity,hedging mismatch"
    },
    {
        "id": "reserve_balance",
        "pairs": "USD/CNH",
        "domain": "Marketplace Payout Risk",
        "title": "Reserved Funds Risk",
        "professional_definition": (
            "Reserved Funds Risk: The discrepancy between Account Balance and Available Balance "
            "necessitates dynamic exposure calculation."
        ),
        "layman_translation": (
            "The balance displayed in a marketplace account may not represent cash that can actually "
            "be withdrawn. Hedging against reserved funds may result in overstated FX exposure."
        ),
        "actionable_playbook": (
            "Platform Context: Account Level Reserve policies may temporarily hold a portion of "
            "seller balances to cover refunds, chargebacks, or operational risks. "
            "Treasury Copilot Recommendation: Use Available Balance rather than total account balance "
            "as the effective hedging base and continuously monitor reserve ratios."
        ),
        "data_sources": {
            "reserve_policy": "https://sellercentral.amazon.com/help/hub/reference/G200185030"
        },
        "tags": "reserve,available balance,account balance,over-hedging,exposure calculation"
    },
    {
        "id": "tiktok_settlement",
        "pairs": "USD/CNH",
        "domain": "Marketplace Payout Risk",
        "title": "TikTok Shop Settlement Logic",
        "professional_definition": (
            "Settlement Logic Risk: TikTok Shop settlement protocols differ materially from Amazon, "
            "requiring dedicated cycle modeling."
        ),
        "layman_translation": (
            "TikTok Shop follows a different settlement process from Amazon. Reusing an Amazon-based "
            "cash-flow model may lead to inaccurate forecasts and poor hedging decisions."
        ),
        "actionable_playbook": (
            "Platform Context: TikTok Shop payout timing and settlement logic differ from traditional "
            "marketplace structures and may require separate cash-flow assumptions. "
            "Treasury Copilot Recommendation: Build a dedicated TikTok settlement model and incorporate "
            "delivery-to-payout adjustment factors when estimating FX exposure."
        ),
        "data_sources": {
            "tiktok_payout": "https://sellercentral.tiktok.com/help/article/settlement-and-payout"
        },
        "tags": "tiktok,settlement logic,cash flow model,dedicated cycle,exposure estimation"
    },
    {
        "id": "account_hold",
        "pairs": "USD/CNH",
        "domain": "Marketplace Payout Risk",
        "title": "Account Suspension Risk",
        "professional_definition": (
            "Account Suspension Risk: Total account holds lead to zero-liquidity scenarios, making "
            "active hedging contracts speculative liabilities."
        ),
        "layman_translation": (
            "When a marketplace account is suspended, incoming cash flows may stop entirely. Existing "
            "hedging positions can become disconnected from underlying business activity."
        ),
        "actionable_playbook": (
            "Platform Context: Marketplace enforcement actions, account health issues, or policy "
            "violations may suspend selling privileges and interrupt payouts. "
            "Treasury Copilot Recommendation: Monitor account-health signals and payout status. "
            "Activate emergency controls that pause or review hedging activity when payout channels "
            "become unavailable."
        ),
        "data_sources": {
            "account_health": "https://sellercentral.amazon.com/help/hub/reference/G200370560"
        },
        "tags": "account suspension,zero liquidity,hedging liability,emergency control,payout interruption"
    },
    {
        "id": "seasonal_gap",
        "pairs": "USD/CNH",
        "domain": "Working Capital Risk",
        "title": "Inventory Financing Risk",
        "professional_definition": (
            "Inventory Financing Risk: Peak sales seasons create significant capital allocation gaps "
            "between inventory investment and cash recovery."
        ),
        "layman_translation": (
            "During major promotional seasons, cash is often tied up in inventory long before sales "
            "proceeds are received. Liquidity management becomes more important than short-term FX optimization."
        ),
        "actionable_playbook": (
            "Platform Context: Seasonal inventory build-up and fulfillment requirements increase "
            "working-capital demands well before sales revenue is realized. "
            "Treasury Copilot Recommendation: Forecast seasonal funding gaps in advance and evaluate "
            "inventory financing or short-term liquidity facilities before increasing hedging activity."
        ),
        "data_sources": {
            "inventory_guide": "https://sellercentral.amazon.com/help/hub/reference/G201411300"
        },
        "tags": "seasonal,inventory financing,working capital,capital gap,liquidity management"
    },
    {
        "id": "shopify_stripe_double_conversion",
        "pairs": "USD/CNH",
        "domain": "Multi-Currency Settlement & Gateway Risk (Shopify + Stripe)",
        "title": "Bypassing Stripe's Double Conversion FX Markup in Cross-Border E-Commerce",
        "professional_definition": (
            "Stripe enforces a strict 'Double Conversion FX Markup' policy on non-US entities. It automatically converts collected USD "
            "into the merchant's local currency using an opaque spot rate prior to disbursement, tacking on a 1.5% to 2.9% FX markup. "
            "If merchants hedge using commercial bank forward contracts based on gross USD sales, the 3% funds intercepted by Stripe "
            "will cause a severe physical spot deficit on the delivery date, triggering a derivative default. Mitigation involves bypassing "
            "Stripe's localized routing by binding US-based virtual accounts (e.g., Wise, WorldFirst) to enforce zero-loss original currency disbursement."
        ),
        "layman_translation": (
            "A non-US e-commerce company sells products in USD using Shopify and Stripe. Instead of sending pure USD to the seller, "
            "Stripe automatically forces a bad currency conversion into the seller's home currency, secretly stealing up to 3% of the money "
            "in hidden exchange fees. If the seller buys a bank forward contract to lock in rates for their total sales, they will face a "
            "cash shortage on settlement day because Stripe already skimmed off that 3%. To fix this, the seller must link a US virtual bank account "
            "to Stripe, forcing the platform to pay out pure USD without any conversion loss."
        ),
        "data_source": "https://www.reddit.com/r/smallbusinessuk/comments/1pzkq5s/uk_business_with_us_clients_better_to_get_paid_in/",
        "tags": "shopify, stripe, double-conversion, fx-markup, virtual-bank, usd-cnh, aml-compliance"
    },
    {
        "id": "amazon_shopify_tax_misalignment",
        "pairs": "EUR/CNH",
        "domain": "Multi-Platform Multi-Currency Risk (Amazon + Shopify)",
        "title": "Over-Hedging Risk from Gross vs. Net Tax Misalignment Across Platforms",
        "professional_definition": (
            "A structural accounting misalignment exists between platform payout rules: Amazon disburses Net revenue (after automated EU OSS VAT deductions), "
            "while Shopify disburses Gross revenue (retaining a 19%-27% uncollected VAT liability). Aggregating GMV blindly across both platforms to lock in "
            "fixed forward contracts results in aggressive over-hedging. Upon settlement, the required VAT withdrawal from the Shopify gross pipeline "
            "will create a critical foreign currency liquidity shortage. Systems must block direct GMV aggregation, downgrade to defensive hedging, and mandate Window Forwards."
        ),
        "layman_translation": (
            "A merchant runs both an Amazon Europe storefront and a Shopify independent site. Amazon automatically subtracts European sales tax "
            "(VAT) before paying the seller, but Shopify pays out the raw gross sales, meaning the tax money is still temporarily sitting in the seller's account. "
            "If the seller blindly adds the total sales figures from both dashboards together to buy a fixed bank hedging contract, they will fall into an over-hedging trap. "
            "When tax day arrives and the Shopify tax money is pulled out to pay the government, the seller won't have enough foreign cash left to fulfill the bank contract. "
            "The system prevents this by calculating a post-tax risk baseline and recommending flexible 'Window Forwards'."
        ),
        "data_source": "Amazon Seller Central - Payout Policy: https://sellercentral.amazon.com/, Shopify Payments Payout Periods: https://help.shopify.com/en/manual/payments/shopify-payments/payouts",
        "tags": "amazon, shopify, vat-tax, misalignment, over-hedging, window-forward, eur-cnh"
    },
    {
        "id": "amazon_tiktok_refund_overhedging",
        "pairs": "USD/CNH, GBP/CNH",
        "domain": "Multi-Platform Multi-Currency Risk (Amazon + TikTok Shop)",
        "title": "Mitigating Derivative Defaults Caused by Social Commerce High Refund Rates",
        "professional_definition": (
            "Social commerce channels like TikTok Shop suffer from impulse-buying dynamics, driving refund rates up to 30%-50%, compared to Amazon's stable 5%-15%. "
            "Treating TikTok's virtual gross GMV as actionable foreign exchange exposure creates massive over-hedging risks. When massive refund outflows hit, "
            "the merchant is left with an absolute deficit of physical foreign currency on the derivative delivery date. Treasury engines must enforce a mandatory "
            "50% wind-control discount on TikTok data feeds and institute a 3-week Refund Cycle Buffer utilizing Window Forwards to match actual delayed settlements."
        ),
        "layman_translation": (
            "A merchant expands from Amazon into TikTok Shop. Unlike Amazon's steady sales, TikTok live-streaming triggers massive impulse buying, "
            "which leads to an extremely high return and refund rate of 30% to 50%. The seller looks at their massive TikTok dashboard sales and immediately "
            "locks in a matching long-term forward contract with their bank. On settlement day, because nearly half of those orders were cancelled and refunded, "
            "the actual cash arriving in the bank account is completely insufficient to cover the contract, triggering heavy default penalties. "
            "The system fixes this by slashing the visible TikTok exposure by 50% for safety and adding a 3-week buffer to let the refund wave pass."
        ),
        "data_source": "Internal Database / Social Commerce Merchant Forums",
        "tags": "amazon, tiktok-shop, social-commerce, refund-rate, over-hedging, window-forward, dynamic-buffer, usd-cnh, gbp-cnh"
    },
    {
        "id": "wenzhou_stainless_steel_usd_option_funnel",
        "pairs": "USD/CNH",
        "domain": "Cross-Border Manufacturing & USD Options Combination",
        "title": "Case 21: Wenzhou Stainless Steel Export Asymmetric Risk Management",
        "professional_definition": (
            "Asymmetric risk management via American-style options combination allows export SMEs to hedge downside currency depreciation while preserving upside participation. "
            "By setting structural protection barriers tailored to historical settlement velocities, firms eliminate the rigidity of traditional static forwards, optimizing net operating margins against volatile macro cash flows."
        ),
        "layman_translation": (
            "This case is about breaking free from rigid fixed-rate lock-ins. A stainless steel factory in Wenzhou was losing out when the exchange rate moved in their favor, or getting crushed when it dropped. "
            "By using an 'American-style options combination,' they put a safety net under their profits: if the USD drops, they are protected; if the USD climbs, they can still cash in on the better rate."
        ),
        "actionable_playbook": (
            "1) Audit historical USD settlement ledgers and available credit lines to map a precision risk profile.\n"
            "2) Deploy a structural $10,000,000 American-style options combination to bound extreme financial costs without losing market upside.\n"
            "3) Monitor real-time spot gradients to lock in definitive net operating margins."
        ),
        "data_sources": {
            "source_url": "https://m.cnfin.com/wx/share?url=//thinktank.cnfin.com/yxyyj-lb/wap/detail/20250915/4302709_1.html",
            "description": "SPD Bank Wenzhou Branch FX Options Combination Hedging Case"
        },
        "tags": "stainless-steel, american-option, options-combination, asymmetric-risk, pufa-bank"
    },
    {
        "id": "wuxi_lithium_battery_usd_short_call_funnel",
        "pairs": "USD/CNH",
        "domain": "Lithium-Ion Supply Chain & Short Call Options Program",
        "title": "Case 22: Wuxi Lithium Precursor Export Programmatic Yield Optimization",
        "professional_definition": (
            "A programmatic short-call options overlay leverages systemic corporate cash inflows to harvest variance premiums. "
            "By executing multi-tiered strike price gradients across a simultaneous 12-tranche contract annualized scope, the treasury monetizes time decay at inception to subsidize baseline manufacturing costs."
        ),
        "layman_translation": (
            "Instead of playing the guessing game and staring at exchange rate screens every day, this lithium exporter automated their strategy. "
            "They set up a 12-part 'short call option' program. The bank pays them cash premiums upfront just for setting up the contracts, which acts as an immediate discount on their operating costs, while smoothing out their monthly financial reports."
        ),
        "actionable_playbook": (
            "1) Benchmark baseline cost metrics to establish a multi-tiered strike price gradient.\n"
            "2) Execute a simultaneous 12-tranche structured short-call options contract overlay covering $240,000,000 annualized scope.\n"
            "3) Capture and book the 12-tranche premium income at inception as an immediate operational cost subsidy.\n"
            "4) Enforce an automated calendar settlement protocol on designated monthly dates to smooth translation variances."
        ),
        "data_sources": {
            "source_url": "https://www.dzzq.com.cn/finance/51440655.html",
            "description": "CIB Nanjing Branch Lithium Ternary Precursor Leading Enterprise Short USD Call Options Combination Case"
        },
        "tags": "lithium-battery, short-call, premium-income, industrial-supply-chain, cib-bank"
    },
    {
        "id": "yiwu_sme_cross_border_zero_cost_option_funnel",
        "pairs": "USD/CNH",
        "domain": "Yiwu Micro-SME Hub & Government-Backed Derivative Credit",
        "title": "Case 23: Yiwu Micro-Merchant Premium and Margin Free Capital Preservation",
        "professional_definition": (
            "A Zero-Cost Option structure paired with government-backed credit enhancement channels targets working capital constraints in micro-SMEs. "
            "By embedding offsetting option layers, the upfront premium is completely neutralized, while state credit guarantees fully waive standard bank margin deposits to preserve corporate operational liquidity."
        ),
        "layman_translation": (
            "Small e-commerce shops in Yiwu usually can't afford hedging because banks demand expensive upfront fees and cash deposits. "
            "This solution uses a 'Zero-Cost Option' combined with a government backing program. It allows small merchants to lock in a guaranteed worst-case exchange rate with zero upfront cost and zero margin deposits, managing everything right inside their mobile banking app."
        ),
        "actionable_playbook": (
            "1) Deploy an institutional Zero-Cost Option structure to lock in a guaranteed settlement floor with zero initial cash premium disbursement.\n"
            "2) Apply for state-backed credit enhancement channels to fully waive bank margin deposit layers, freeing up corporate working capital.\n"
            "3) Route execution workflows through digital banking mobile applications to compress end-to-end transaction latency from 1 hour to under 10 minutes."
        ),
        "data_sources": {
            "source_url": "https://m.cnfin.com/wx/share?url=//thinktank.cnfin.com/yxyyj-lb/wap/detail/20251103/4328196_1.html",
            "description": "ICBC Yiwu Branch Micro Foreign Trade Enterprise Zero-Cost Option & Government Credit Enhancement Hedging Case"
        },
        "tags": "yiwu-sme, zero-cost-option, margin-free, government-credit, icbc-bank"
    },
    {
        "id": "hubei_chemical_sme_platform_option_funnel",
        "pairs": "USD/CNH",
        "domain": "Hubei Chemical Export SME & Cross-Border Financial Service Platform",
        "title": "Case 24: Hubei Chemical SME Long-Cycle Ledger Verification via Digital Platform",
        "professional_definition": (
            "Integrating treasury operations with the State Administration of Foreign Exchange (SAFE) Cross-Border Financial Service Platform enables real-time automated ledger verification. "
            "By digitalizing trade, logistics, and collection audits, compliance latency is minimized, allowing private SMEs with long order-to-cash cycles to rapidly deploy targeted Put Options to secure downside revenue floors."
        ),
        "layman_translation": (
            "A chemical exporter in Hubei had to wait months to collect money, but they couldn't hedge because the bank's manual paper review took forever. "
            "By linking directly to the State Administration of Foreign Exchange's digital platform, the bank can now verify their trade records instantly online. This allowed them to quickly buy a 'Put Option' to guarantee their minimum revenue floor, cutting the approval wait time from an hour to under 20 minutes."
        ),
        "actionable_playbook": (
            "1) Integrate treasury endpoints directly into the State Administration of Foreign Exchange (SAFE) Cross-Border Financial Service Platform.\n"
            "2) Authorize instant digital data sharing to pull verified trade, logistics, and collection ledgers automatically.\n"
            "3) Execute a targeted 100,000 USD long-dated Put Option contract to securely lock in a baseline downside floor.\n"
            "4) Compress the entire bank risk clearance review pipeline latency from 1 hour down to under 20 minutes."
        ),
        "data_sources": {
            "source_url": "http://www.safe.gov.cn/hubei/2025/0107/2394.html",
            "description": "SAFE Hubei Branch Launches Pilot Application for Exchange Rate Risk Management on Cross-Border Financial Service Platform (Bank of Communications Huangshi Branch Case)"
        },
        "tags": "chemical-export, safe-platform, put-option, digital-verification, bank-of-communications"
    },
    {
        "id": "henan_electrical_sme_long_production_window_funnel",
        "pairs": "USD/CNH",
        "domain": "Henan Electrical Equipment SME & Long Production Cycle Treasury",
        "title": "Case 25: Henan Electrical SME Margin Neutrality under Elongated Shipment Horizons",
        "professional_definition": (
            "Implementing 'FX Neutrality' frameworks via cross-border logistics data integration addresses valuation shocks in long-cycle manufacturing pipelines. "
            "By auditing multi-quarter production dependencies and embedding customized asset derivative hedging envelopes, the treasury structurally immunizes core manufacturing costs against macro-driven exchange rate fluctuations prior to shipment delivery."
        ),
        "layman_translation": (
            "An electrical equipment factory in Henan takes months to manufacture and ship its products, leaving its profit margins completely exposed to massive currency swings during production. "
            "By pairing up with Bank of China and utilizing the state cross-border digital platform, they built a customized option shield based on 'FX Neutrality.' This completely locks in their manufacturing costs ahead of time, ensuring that sudden exchange rate swings won't eat away their factory's hard-earned profits before the goods even ship."
        ),
        "actionable_playbook": (
            "1) Partner with the local Bank of China (BOC) branch to audit multi-quarter manufacturing cycle dependencies.\n"
            "2) Deploy the SAFE Cross-Border Financial Service Platform to inject verified cross-border logistics data lines into the credit pipeline.\n"
            "3) Tailor a programmatic multi-currency asset derivative hedging envelope structured around strict 'FX Neutrality' principles.\n"
            "4) Secure the export revenue baseline via customized USD options to structurally lock in manufacturing costs."
        ),
        "data_sources": {
            "source_url": "https://finance.sina.com.cn/jjxw/2025-02-11/doc-inekcccy3194591.shtml",
            "description": "Bank of China Pingdingshan Branch Implements Henan Province's First Exchange Rate Risk Management Service Business via SAFE Cross-Border Financial Service Platform (A Henan Electrical Company Case)"
        },
        "tags": "electrical-equipment, long-production-cycle, fx-neutrality, customized-options, bank-of-china"
    },
    {
        "id": "irish_sme_procurement_window_forward_funnel",
        "pairs": "EUR/USD",
        "domain": "Irish SME Import Hub & Flexible Cash-Cycle Procurement",
        "title": "Case 26: Irish SME Cash Flow Mismatch under Rigid Forward Obligations",
        "professional_definition": (
            "A Window Forward Contract acts as an agile liquidity optimization derivative designed to mitigate cash-cycle disruptions caused by logistics volatility. "
            "Unlike traditional fixed-date forwards that enforce a rigid maturity date, a Window Forward establishes a multi-month execution horizon (e.g., 30-to-180 days). "
            "This structure grants corporate treasuries embedded drawdown flexibility to dynamically settle liabilities in tranches or fully as real-time invoice arrivals occur, completely eliminating forced extension bank penalties and temporal short-term liquidity squeezes."
        ),
        "layman_translation": (
            "This solution breaks the handcuffs of rigid fixed-date bank contracts. An importing company in Ireland needs to pay US suppliers regularly, but shipping delays mean their goods—and their invoices—constantly arrive late. "
            "Instead of locking into a single calendar day and getting penalized when the timeline slips, they use a 'Window Forward Contract.' This lets them lock in today's exchange rate but gives them a flexible 1-to-6 month window to pull the cash whenever the bills actually land on their desk."
        ),
        "actionable_playbook": (
            "1) Abandon rigid fixed-date derivatives and map the corporate treasury to an agile 30-to-180 day liquidity baseline.\n"
            "2) Deploy an institutional Window Forward Contract to lock in the target baseline cost today.\n"
            "3) Utilize the embedded 'Window drawdown flexibility' to dynamically execute and pull dollars anytime within the designated multi-month window.\n"
            "4) Synchronize treasury delivery seamlessly with real-time supplier invoice arrivals to secure total margin certainty and avoid constant firefighting."
        ),
        "data_sources": {
            "source_url": "https://isme.ie/how-forward-contracts-give-smes-control-in-volatile-currency-markets/",
            "description": "Irish Small and Medium Enterprises Association (ISME) Official SME Forward Risk Control and Window Application Guidelines"
        },
        "tags": "irish-sme, window-forward, procurement-mismatch, cash-flow-flexibility, eur-usd"
    },
    {
        "id": "ebury_style_sme_growth_overhead_mismatch_funnel",
        "pairs": "GBP/USD",
        "domain": "Hyper-Growth Scale-Up & Non-Linear Overhead Treasury Risk",
        "title": "Case 27: Hyper-Growth Scale-Up Treasury Lag under Asymmetric Overhead Expansion",
        "professional_definition": (
            "Programmatic deployment of Window Forward Contracts serves as a capital-preservation shield against asymmetric non-linear cost expansion during global scale-up phases. "
            "By benchmarking a rigid 70% threshold of projected multi-region revenues under flexible drawdown derivatives and clearing the remaining 30% via automated spot FX mechanisms, the enterprise insulates its localized overhead runway from translation shocks and spot market degradation."
        ),
        "layman_translation": (
            "When a high-growth tech company scales globally, its office rents and payroll obligations skyrocket instantly, but international sales revenues usually trickle in much slower. "
            "To stop volatile exchange rates from swallowing up their remaining funding, they automated their treasury. They locked in 70% of their expected regional revenue using flexible 'Window Forward Contracts' so they can draw down cash flexibly as erratic client payments arrive, ensuring they always have enough local currency to keep the lights on."
        ),
        "actionable_playbook": (
            "1) Restructure the treasury framework to benchmark a rigid 70% threshold of projected multi-region revenues into institutional forward contracts.\n"
            "2) Deploy programmatic Window Forward Contracts as the primary hedging pillar to grant total flexible drawdown control over erratic cash arrival cycles.\n"
            "3) Consolidate the remaining 30% of volatile residual flows under strict automated Spot FX clearing protocols.\n"
            "4) Secure tier-1 institutional banking backing to cross-guarantee global credit limits and successfully insulate the localized overhead cost engine from macro currency shocks."
        ),
        "data_sources": {
            "source_url": "https://www.fxcintel.com/research/analysis/inside-eburys-journey-to-100m-in-revenue-and-break-even",
            "description": "FXC Intelligence Report on Multinational Fintech Cross-Border Platform Ebury's Journey to 100M Revenue and Structural Scale Expansion"
        },
        "tags": "ebury-scale, hyper-growth, overhead-lag, window-forward, operational-ebitda, gbp-usd"
    },
    {
        "id": "melecs_manufacturing_rolling_hedge_funnel",
        "pairs": "EUR/USD",
        "domain": "European Electronics Manufacturing & Asymmetric Supply Chain",
        "title": "Case 28: Material-Intensive Manufacturer Margin Compression under Rigid Pricing Clauses",
        "professional_definition": (
            "A dual-track Rolling Hedging Strategy utilizes monthly expiring options buckets structured around a strict baseline budget rate to immunize material-intensive manufacturing margins. "
            "By segregating contracts without price-escalation clauses and applying a multi-period programmatic rolling layer, the treasury effectively compresses balance sheet volatility and restores complete cost predictability across an extended (up to 24-month) procurement runway."
        ),
        "layman_translation": (
            "A massive European electronics maker buys components in USD but sells finished products in Euros. They tried adding clauses to change their prices if currency rates shifted, but clients rejected them. "
            "To protect their paper-thin factory profit margins, they launched a 'Rolling Hedging Strategy.' Every month, they layer on a new set of currency option buckets based on their strict factory budget rate. This guarantees their costs for 24 months out, compressing their balance sheet exchange rate risk from a scary 5% down to a highly predictable 1%."
        ),
        "actionable_playbook": (
            "1) Enforce a dual-track hedging program separating price-adjusted customer contracts from non-adjusted ones.\n"
            "2) Deploy fixed forward contracts for non-adjusted pipelines to lock in budget costs for up to 24 months.\n"
            "3) Structure a programmatic Rolling Hedging Strategy utilizing monthly expiring options buckets based on a strict budget baseline rate.\n"
            "4) Secure tailored option drawdown flexibility to insulate the balance sheet, successfully compressing FX volatility from 5% down to a predictable 1% to 2% corridor."
        ),
        "data_sources": {
            "source_url": "https://convera.com/success-stories/rolling-hedging-strategy-helps-melecs-mitigate-calculation-risk/",
            "description": "Convera Success Story on European Electronics Giant Melecs' Rolling Hedging Strategy and Balance Sheet FX Risk Control"
        },
        "tags": "electronics-manufacturing, rolling-hedge, price-escalation-clause, balance-sheet-volatility, convera-case, eur-usd"
    },
    {
        "id": "uk_parent_subsidiary_currency_swap_funnel",
        "pairs": "GBP/USD",
        "domain": "Cross-Border Intercompany Financing & Net Investment Hedging",
        "title": "Case 29: UK Parent Net Investment Exposure under Long-Term Subsidiary Loan",
        "professional_definition": (
            "An institutional Currency Swap Contract with a fixed-to-fixed architecture ($1.6/£1) serves as a structural hedge for multi-year cross-border intercompany financing. "
            "By synchronizing semi-annual interest payment exchanges and executing a mandatory final re-exchange of principal at a predetermined terminal rate, the structure insulates long-term asset values from accounting translation deterioration and achieves strict Net Investment Hedge standard compliance."
        ),
        "layman_translation": (
            "A UK parent company lent $100 million to its US branch for an acquisition. Because it's a long-term 5-year loan, regular forward contracts won't work, leaving the parent completely exposed to a weakening US Dollar that would wreck their consolidated accounting reports. "
            "To solve this, they entered into a 5-year 'Currency Swap' with a bank. They traded their USD interest cash flows for steady British Pound interest cash flows every six months. Best of all, at the very end of the 5 years, the bank is legally locked into converting the $100 million principal back into a guaranteed £62.5 million, completely wiping out long-term currency devaluation risks."
        ),
        "actionable_playbook": (
            "1) Enter into a 5-year institutional Currency Swap Contract with a fixed-to-fixed exchange architecture ($1.6/£1).\n"
            "2) Synchronize half-yearly payment streams: paying out the subsidiary's 5.5% USD interest to the bank while receiving a stable 4.75% Sterling interest calculation baseline.\n"
            "3) Execute a structural terminal re-exchange at maturity, securing exactly £62.5 million from the bank to fully insulate the $100 million principal repayment.\n"
            "4) Designate the swap as a strict Net Investment Hedge under international financial reporting standards to achieve absolute corporate margin and tax optimization neutrality."
        ),
        "data_sources": {
            "source_url": "https://www.gov.uk/hmrc-internal-manuals/corporate-finance-manual/cfm13430",
            "description": "UK HM Revenue & Customs (HMRC) Internal Corporate Finance Manual: Regulatory Compliance Standards for Cross-Border Intercompany Foreign Currency Loans and Currency Swap Hedging Matching"
        },
        "tags": "intercompany-loan, currency-swap, net-investment-hedge, fixed-to-fixed, long-term-exposure, gbp-usd"
    },
    {
        "id": "year_end_bulk_conversion_risk",
        "pairs": "USD/CNH, EUR/CNH",
        "domain": "Hedging Strategy & Policy Inaction",
        "title": "Risks of Year-End Bulk FX Conversion and Exposure Concentration",
        "professional_definition": (
            "Amassing unhedged foreign exchange exposures over multiple quarters with the intention of executing a single, "
            "consolidated bulk spot conversion at year-end exposes the enterprise to severe tail-risk events and illiquidity cycles. "
            "This practice aggregates temporal exposure into a highly volatile window (typically late December), during which "
            "international banking institutions undergo annual books-closing and balance-sheet compression, leading to significant "
            "bid-ask spread widening and liquidity evaporation. Furthermore, it creates severe accounting vulnerabilities under IFRS 9 "
            "regarding unrecognized fair-value volatility and risks breaching year-end commercial credit lines."
        ),
        "layman_translation": (
            "A cross-border merchant decides not to do any foreign exchange hedging throughout the year, thinking they will just save "
            "time and convert all their accumulated multi-currency sales into CNH in one big chunk at the end of December. When the "
            "end of the year arrives, most global banks are closing their books for the holidays, which causes transaction fees and "
            "exchange spreads to skyrocket. If the exchange rate crashes right during those final weeks, the merchant's entire annual "
            "profit can vanish in days. Additionally, banks often freeze or reduce credit lines for new contracts at year-end, leaving "
            "the merchant stuck without options."
        ),
        "data_source": "https://www.pwc.com/hu/hu/szolgaltatasok/ifrs/ifrs_9/kerdesekvalaszok/ifrs_9_heding_in_practice_FAQ.pdf",
        "tags": "hedging-inaction, year-end-closing, exposure-concentration, spot-layering, liquidity-risk"
    },
    {
        "id": "immediate_spot_conversion_inefficiency",
        "pairs": "USD/CNH",
        "domain": "Transaction Cost & Natural Hedging Ineffectiveness",
        "title": "Inefficiencies of Immediate Full Spot Conversion vs. Natural Hedging",
        "professional_definition": (
            "Executing a rigid 100% immediate spot conversion of incoming foreign currency revenues (e.g., USD) into the domestic currency "
            "(CNH) out of extreme risk aversion effectively dismantles the enterprise's natural hedging mechanism. Cross-border SMEs "
            "routinely face unavoidable multi-currency outgoing operational liabilities such as digital advertising spend (Facebook/Google Ads), "
            "overseas warehousing, and logistics. Forcing immediate spot settlement requires subsequent reverse currency repurchases, "
            "incurring a double-conversion bid-ask spread penalty. This high-frequency circular conversion pattern triggers automated "
            "banking AML tracking for suspicious trade activity and invites regulatory scrutiny over payment misalignment."
        ),
        "layman_translation": (
            "A seller gets scared of exchange rate fluctuations and automatically converts 100% of their incoming US dollar payouts into "
            "CNH the minute it hits their account. However, a few days later, they have to pay their overseas warehouse, shipping fees, "
            "or Google and Facebook ad bills in US dollars. Because they already converted everything to CNH, they are forced to buy "
            "US dollars back at a high price. The bank charges a fee both times, wiping out their profit margins. Doing this repeatedly "
            "also signals red flags to automated anti-money laundering bank systems, which might temporarily lock down the account to demand "
            "proof of real trade logs."
        ),
        "data_source": "https://www.treasurers.org/ACTmedia/mowact_enterpriserisksurvey.pdf",
        "tags": "immediate-conversion, natural-hedging, double-conversion, spot-rate, aml-flag, transaction-costs"
    },
    {
        "id": "commingling_business_personal_accounts",
        "pairs": "USD/CNH, EUR/CNH",
        "domain": "Compliance & Entity Segregation Risk",
        "title": "Anti-Money Laundering and Freezing Risks from Commingling Corporate and Personal Accounts",
        "professional_definition": (
            "Commingling personal foreign exchange accounts with corporate accounts to receive e-commerce platform disbursements or "
            "settle supplier liabilities breaches basic principles of asset segregation and triggers aggressive compliance intervention. "
            "Under the FATF Recommendation 16 (Travel Rule) and institutional KYC frameworks, financial institutions are mandated to track "
            "the ultimate beneficial owner (UBO) and maintain strict underlying commercial trade consistency. High-frequency, high-volume "
            "commercial flows routing into individual accounts instantly trigger automated anti-money laundering (AML) fraud detection "
            "algorithms, leading to immediate account freezes without prior notice and exposing the entity to invasive tax audits."
        ),
        "layman_translation": (
            "An e-commerce founder uses their personal bank account or digital wallet instead of a proper corporate account to receive "
            "marketplace funds or pay global suppliers because it seems faster or saves corporate tax. This triggers strict automated "
            "anti-money laundering alarms at the bank. Under global travel rules, financial networks must verify that the business identity "
            "matches the account holder name perfectly. When a personal account acts like a massive commercial pipeline, the bank will freeze "
            "the funds instantly without warning, leaving the business with zero cash flow and inviting severe corporate tax fines."
        ),
        "data_source": "FATF Recommendation 16 (Travel Rule) : https://tazapay.com/blog/travel-rule-cross-border-payments-2026",
        "tags": "entity-segregation, travel-rule, aml-freeze, asset-protection, kyc-compliance, tax-audit"
    },
    {
        "id": "ignoring_eur_cross_currency_exposure",
        "pairs": "EUR/CNH",
        "domain": "Multi-Currency Portfolio & Cross-Currency Risk",
        "title": "Cross-Currency Slippage and Policy Divergence from Neglecting Non-USD Exposures",
        "professional_definition": (
            "Treating non-USD assets like the Euro (EUR) as secondary, correlation-dependent exposures to the US Dollar represents a "
            "severe treasury blind spot. Under asymmetric macroeconomic monetary cycles (e.g., policy divergence between the ECB and the "
            "Federal Reserve) or localized geopolitical events, the EUR/USD cross-rate can fluctuate wildly, completely eroding unhedged "
            "European profit margins. Operating without independent currency monitoring violates multi-currency portfolio management principles "
            "under Basel regulations, incurring compound bid-ask spread friction during mandatory EU VAT settlements and violating EMIR "
            "hedging compliance disclosure limits."
        ),
        "layman_translation": (
            "A global seller focuses entirely on managing their US Dollar risk, completely ignoring their Euro accounts because they "
            "wrongly assume the Euro always moves in sync with the Dollar. When the European Central Bank and the US Federal Reserve split "
            "on interest rates, the Euro plummets independently, wiping out all the profits from the seller's European stores. To make things "
            "worse, because they do not hold Euros safely or hedge them, they are forced to do expensive emergency exchanges when the "
            "fixed EU sales tax (VAT) deadline arrives, running the risk of heavy tax penalties and storefront bans."
        ),
        "data_source": "https://www.simmons-simmons.com/en/publications/cmmjgg2xn0010ufws2alycppm/eu-emir-3-the-new-clearing-thresholds-regime",
        "tags": "eur-cnh, policy-divergence, cross-currency, portfolio-risk, vat-settlement, emir-compliance"
    },
    {
        "id": "missing_rolling_fx_budget_planning",
        "pairs": "USD/CNH, EUR/CNH",
        "domain": "Budgeting & Liquidity Forecasting Risk",
        "title": "Credit Line Denials and Cash Flow Mismatch Due to Absence of Rolling FX Budgets",
        "professional_definition": (
            "Relying on reactive, ad-hoc spot trading rather than an established cash-flow forecasting framework leaves an enterprise "
            "highly vulnerable to structural currency mismatches. Under international banking and regulatory standards, commercial banks "
            "require objective documentation proving underlying trade authenticity (such as forward-looking corporate rolling budgets and "
            "pro-forma invoices) before approving or maintaining derivative credit lines. Operating without a 13-week rolling forecast leads "
            "to liquidity spikes, forced spot executions during disadvantageous market dips, and an inability to maintain defensive hedge "
            "ratios, which violates Basel III Liquidity Coverage Ratio (LCR) safety buffers."
        ),
        "layman_translation": (
            "A company operates its cross-border treasury day-to-day by reacting to whatever the exchange rate is this morning, completely "
            "lacking a 90-day rolling budget forecast for their cash flow. When an unexpected major invoice from a supplier or factory "
            "arrives, they are forced to exchange cash immediately at a terrible market rate, losing a massive chunk of change. Furthermore, "
            "when they try to ask a commercial bank for a proper forward contract line to lock in a safe rate for the future, the bank rejects "
            "them because the company cannot show documented, organized proof of their real future trade projections."
        ),
        "data_source": "CIMA Global Management Accounting Principle (Rolling Forecasts and Budget Rate) : https://www.onetribeadvisory.com/knowledge-hub/scenario-analysis-guide/, Basel III LCR - Liquidity Coverage Ratio",
        "tags": "rolling-forecast, cash-flow-mismatch, baseline-budget, credit-line, underlying-authenticity, liquidity-buffer"
    },
    {
        "id": "fx_microstructure_liquidity_squeeze",
        "pairs": "USD/CNH,EUR/CNH,HKD/CNH",
        "domain": "FX Microstructure & Central Bank Liquidity Management",
        "title": "CNH Order Flow Toxicity, CCS Dislocation, and Offshore Squeeze Mechanics",
        "professional_definition": (
            "Offshore CNH market operates as a friction-heavy parallel pool where liquidity can evaporate non-linearly due to clearing bank bottlenecks and cross-border settlement quotas. "
            "When CNH HIBOR spikes (via targeted central bank bill issuance or commercial bank hoarding), the cost of shorting CNH scales parabolically, forcing speculative short-covering. "
            "The deviation of CNH Cross-Currency Swap (CCS) from standard covered interest parity (CIP) measures the localized 'scarcity premium' of dollar liquidity vs. offshore CNH velocity."
        ),
        "layman_translation": (
            "Offshore CNH liquidity can suddenly dry up because Hong Kong clearing banks have quota limits and the central bank can issue bills to absorb liquidity. "
            "When CNH HIBOR spikes, the cost of shorting CNH explodes, forcing speculators to cover positions, which can cause a sharp, short-term rally in CNH. "
            "At the same time, the CNH-USD cross-currency swap (CCS) basis may deviate from interest rate parity, signalling a dollar scarcity premium. Watch for these signs of a liquidity squeeze."
        ),
        "actionable_playbook": (
            "1) Monitor CNH HIBOR curves daily; an inverted curve (O/N > 1M) indicates acute offshore pooling constraints or structural intervention.\n"
            "2) Track the basis spread between USD/CNY (Onshore) and USD/CNH (Offshore). If |CNY - CNH| > threshold, execute cross-border friction arbitrage or risk-reversal hedging strategy.\n"
            "3) Cross-reference implied volatility skew (25D Risk Reversal); a heavy bias towards USD call/CNH put coupled with dropping offshore deposits triggers the 'Liquidity Squeeze' alert.\n"
            "4) Evaluate central bank bill maturities in Hong Kong. If bill rolling decreases during depreciation cycles, anticipate direct forward-market or swap-market intervention."
        ),
        "tags": "cnh,offshore,hibor,ccs,liquidity squeeze,order flow toxicity,vpin,central bank bills,short squeeze,forward intervention",
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
        "professional_definition": (
            "State Administration of Foreign Exchange (SAFE) enforces a strict bona fides test requiring absolute synchronization "
            "between cross-border cash velocity, digital storefront invoices, and physical 3PL freight manifests. "
            "When forward contract maturities arrive without matching digital order flows, clearing banks freeze incoming onshore wires. "
            "Treasury automation must programmatically map transaction-level platform webhooks to derivative delivery manifests."
        ),
        "layman_translation": (
            "SAFE requires that every cross-border receipt be backed by matching e-commerce orders and shipping documents. "
            "If a forward contract matures without corresponding order and freight records, the clearing bank will freeze the funds. "
            "You must use automated tools to link platform orders, logistics information, and FX contracts; otherwise, money will be blocked from onshore settlement."
        ),
        "actionable_playbook": (
            "1) Extract transaction logs from Amazon/Shopify daily; compute cumulative standard deviations against active derivative delivery dates.\n"
            "2) If a mismatch between physical container dispatch and forward delivery date exceeds 5 business days, trigger automated compliance window adjustments.\n"
            "3) Strictly restrict long-term outbound cross-border intercompany liquid pooling transfers unless backed by verified invoice matrices.\n"
            "4) Keep all invoices, shipping bills, and storefront receipts archived for at least 2 years for SAFE audit."
        ),
        "tags": "safe,three-way matching,compliance,invoice,freight,audit,settlement freeze,derivative delivery,storefront,3pl",
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
        },
        "tags": "aml,gateway freeze,payment gateway,stripe,paypal,liquidity collapse,contagion,multi-account,forward commitment,operational risk"
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
        },
        "tags": "ifrs 9,cas 22,hedge accounting,hedge effectiveness,dollar-offset method,audit trail,cash flow hedge,corporate governance,derivatives,compliance"
    }
]
_KNOWLEDGE_RISK_PROFILE_BY_ID: dict[str, dict[str, Any]] = {
    "high_frequency_micro_payouts": {
        "risk_domains": ["Exposure Risk", "Liquidity Risk"],
        "severity": 3,
        "affected_entities": ["cross-border ecommerce", "exporters", "micro-merchants"],
        "mitigation_frameworks": ["Natural Hedging", "Multi-Currency Netting"],
    },
    "net_payout_rolling_hedge_eu_vat": {
        "risk_domains": ["Exposure Risk", "Treasury Operational Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "exporters", "eu vat sellers"],
        "mitigation_frameworks": ["Forward Contract", "Window Forward"],
    },
    "compliance_safe_harbor_allowance": {
        "risk_domains": ["Compliance Risk", "Liquidity Risk"],
        "severity": 3,
        "affected_entities": ["smes", "exporters", "cross-border enterprises"],
        "mitigation_frameworks": ["Government Guarantee"],
    },
    IMF_MACRO_KB_ID: {
        "risk_domains": ["Macro Event Risk"],
        "severity": 3,
        "affected_entities": ["treasury desks", "cross-border enterprises"],
        "mitigation_frameworks": ["Forward Contract"],
    },
    "amazon_us_usd_payout_funnel": {
        "risk_domains": ["Exposure Risk", "Treasury Operational Risk"],
        "severity": 4,
        "affected_entities": ["exporters", "cross-border ecommerce", "amazon sellers"],
        "mitigation_frameworks": ["Forward Contract", "Natural Hedging"],
    },
    "amazon_eu_eur_gbp_vat_funnel": {
        "risk_domains": ["Exposure Risk", "Liquidity Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "amazon sellers", "exporters"],
        "mitigation_frameworks": ["Window Forward", "Natural Hedging"],
    },
    "shopify_independent_gateway_funnel": {
        "risk_domains": ["Liquidity Risk", "Compliance Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "independent merchants"],
        "mitigation_frameworks": ["Natural Hedging", "Forward Contract"],
    },
    "hongkong_offshore_lers_cips_funnel": {
        "risk_domains": ["Compliance Risk", "Exposure Risk"],
        "severity": 3,
        "affected_entities": ["cross-border ecommerce", "hong kong offshore entities"],
        "mitigation_frameworks": ["Natural Hedging", "Multi-Currency Netting"],
    },
    "cross_border_payroll_window_netting": {
        "risk_domains": ["Liquidity Risk", "Exposure Risk"],
        "severity": 4,
        "affected_entities": ["exporters", "hong kong entities", "factory payroll"],
        "mitigation_frameworks": ["Window Forward"],
    },
    "government_backed_liquidity_enhancement": {
        "risk_domains": ["Liquidity Risk"],
        "severity": 4,
        "affected_entities": ["smes", "exporters"],
        "mitigation_frameworks": ["Government Guarantee", "Forward Contract"],
    },
    "cross_border_payment_cost_optimization": {
        "risk_domains": ["Treasury Operational Risk", "Compliance Risk"],
        "severity": 3,
        "affected_entities": ["smes", "importers", "exporters"],
        "mitigation_frameworks": ["Multi-Currency Netting", "Natural Hedging"],
    },
    "traditional_export_exposure_option_shield": {
        "risk_domains": ["Exposure Risk"],
        "severity": 4,
        "affected_entities": ["exporters", "low-margin export enterprises"],
        "mitigation_frameworks": ["Currency Option"],
    },
    "micro_merchant_one_yuan_option_shield": {
        "risk_domains": ["Exposure Risk"],
        "severity": 3,
        "affected_entities": ["micro-exporters", "individual merchants"],
        "mitigation_frameworks": ["Currency Option"],
    },
    "industry_furniture_usd_mismatch_trap": {
        "risk_domains": ["Treasury Operational Risk", "Liquidity Risk"],
        "severity": 5,
        "affected_entities": ["exporters", "furniture sellers", "long-cycle supply chain"],
        "mitigation_frameworks": ["Window Forward"],
    },
    "industry_apparel_eur_high_return_trap": {
        "risk_domains": ["Exposure Risk", "Treasury Operational Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "fast fashion sellers"],
        "mitigation_frameworks": ["Natural Hedging"],
    },
    "amazon_uk_gbp_vat_funnel": {
        "risk_domains": ["Exposure Risk", "Liquidity Risk"],
        "severity": 4,
        "affected_entities": ["exporters", "cross-border ecommerce", "amazon sellers"],
        "mitigation_frameworks": ["Window Forward"],
    },
    "derivative_zero_cost_collar_strategy": {
        "risk_domains": ["Treasury Operational Risk"],
        "severity": 3,
        "affected_entities": ["treasury desks", "exporters"],
        "mitigation_frameworks": ["Zero-Cost Collar"],
    },
    "derivative_fx_swap_liquidity_misalignment": {
        "risk_domains": ["Liquidity Risk", "Treasury Operational Risk"],
        "severity": 3,
        "affected_entities": ["multicurrency treasuries", "cross-border enterprises"],
        "mitigation_frameworks": ["Multi-Currency Netting"],
    },
    "derivative_forward_rollover_early_delivery": {
        "risk_domains": ["Liquidity Risk", "Treasury Operational Risk"],
        "severity": 3,
        "affected_entities": ["exporters", "supply chain operators"],
        "mitigation_frameworks": ["Forward Contract", "Window Forward"],
    },
    "macro_fed_fomc_nfp_volatility_shield": {
        "risk_domains": ["Macro Event Risk"],
        "severity": 5,
        "affected_entities": ["treasury desks", "exporters", "cross-border ecommerce"],
        "mitigation_frameworks": ["Forward Contract", "Currency Option"],
    },
    "macro_ecb_boe_joint_divergence_shield": {
        "risk_domains": ["Macro Event Risk", "Exposure Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "eur gbp treasuries"],
        "mitigation_frameworks": ["Forward Contract", "Zero-Cost Collar"],
    },
    "macro_pboc_counter_cyclical_factor_shield": {
        "risk_domains": ["Macro Event Risk", "Liquidity Risk"],
        "severity": 5,
        "affected_entities": ["treasury desks", "cross-border enterprises"],
        "mitigation_frameworks": ["Forward Contract"],
    },
    "macro_geopolitical_tail_risk_freight_shield": {
        "risk_domains": ["Geopolitical / Supply Chain Risk", "Liquidity Risk"],
        "severity": 5,
        "affected_entities": ["exporters", "supply chain operators", "importers"],
        "mitigation_frameworks": ["Window Forward", "Currency Option"],
    },
    "macro_boj_carry_trade_unwinding_shield": {
        "risk_domains": ["Macro Event Risk", "Liquidity Risk"],
        "severity": 4,
        "affected_entities": ["treasury desks", "exporters"],
        "mitigation_frameworks": ["Forward Contract", "Currency Option"],
    },
    "fx_microstructure_liquidity_squeeze": {
        "risk_domains": ["Liquidity Risk", "Macro Event Risk"],
        "severity": 5,
        "affected_entities": ["treasury desks", "offshore cnh participants"],
        "mitigation_frameworks": ["Forward Contract"],
    },
    "compliance_safe_three_way_matching": {
        "risk_domains": ["Compliance Risk", "Treasury Operational Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "exporters"],
        "mitigation_frameworks": ["Forward Contract"],
    },
    "compliance_aml_gateway_freeze_contagion": {
        "risk_domains": ["Compliance Risk", "Liquidity Risk"],
        "severity": 4,
        "affected_entities": ["cross-border ecommerce", "payment gateway merchants"],
        "mitigation_frameworks": ["Multi-Currency Netting"],
    },
    "accounting_ifrs9_hedge_effectiveness": {
        "risk_domains": ["Treasury Operational Risk", "Compliance Risk"],
        "severity": 3,
        "affected_entities": ["treasury desks", "cross-border enterprises"],
        "mitigation_frameworks": ["Forward Contract"],
    },
}
_KNOWLEDGE_CASE_METADATA_BY_ID: dict[str, dict[str, Any]] = {
    "high_frequency_micro_payouts": {
        "business_type": "Cross-Border E-Commerce",
        "pain_points": [
            "high-frequency micro payout fragmentation",
            "double-conversion spread bleed",
            "USD receivable concentration",
        ],
    },
    "net_payout_rolling_hedge_eu_vat": {
        "business_type": "EU VAT Multi-Factor Net Repatriation",
        "pain_points": [
            "gross-sales over-hedging trap",
            "VAT-adjusted net payout mismatch",
            "EUR receivable timing drift",
        ],
    },
    "compliance_safe_harbor_allowance": {
        "business_type": "SAFE Capital Settlement",
        "pain_points": [
            "post-settlement SAFE audit exposure",
            "trade background documentation gap",
            "petty cash settlement limit pressure",
        ],
    },
    IMF_MACRO_KB_ID: {
        "business_type": "IMF Macro Anchor Treasury",
        "pain_points": [
            "long-cycle CPI spread deviation",
            "structural inflation differential shock",
            "90-day receivable macro drift",
        ],
    },
    "amazon_us_usd_payout_funnel": {
        "business_type": "Amazon US Marketplace",
        "pain_points": [
            "USD payout concentration",
            "Amazon US settlement cycle lag",
            "referral and FBA deduction over-hedge",
        ],
    },
    "amazon_eu_eur_gbp_vat_funnel": {
        "business_type": "Amazon EU Marketplace",
        "pain_points": [
            "EUR VAT structural leakage",
            "ACCS FX markup premium",
            "account level reserve bottleneck",
        ],
    },
    "shopify_independent_gateway_funnel": {
        "business_type": "Shopify Independent Store",
        "pain_points": [
            "PayPal rolling reserve lock",
            "multi-gateway payout scattering",
            "cash flow timing mismatch",
        ],
    },
    "hongkong_offshore_lers_cips_funnel": {
        "business_type": "Hong Kong Offshore Banking",
        "pain_points": [
            "HKD/CNH floating leg volatility",
            "three-way matching compliance friction",
            "offshore aggregation settlement delay",
        ],
    },
    "cross_border_payroll_window_netting": {
        "business_type": "Cross-Border Payroll Matching",
        "pain_points": [
            "rigid CNH payroll deadline",
            "HKD weakness forced spot conversion",
            "factory payroll cash flow timing mismatch",
        ],
    },
    "government_backed_liquidity_enhancement": {
        "business_type": "SME Liquidity Preservation",
        "pain_points": [
            "derivative margin cash freeze",
            "working capital pressure",
            "naked FX exposure from credit constraint",
        ],
    },
    "cross_border_payment_cost_optimization": {
        "business_type": "Multi-Currency Payment Optimization",
        "pain_points": [
            "hidden SWIFT and bank spread leakage",
            "AML transit freeze risk",
            "payable-receivable denomination mismatch",
        ],
    },
    "traditional_export_exposure_option_shield": {
        "business_type": "Low-Margin Export Enterprise",
        "pain_points": [
            "USD receivable concentration",
            "unhedged export revenue volatility",
            "rapid export volume scaling exposure",
        ],
    },
    "micro_merchant_one_yuan_option_shield": {
        "business_type": "Micro-Exporter",
        "pain_points": [
            "irregular micro cash flow timing",
            "USD receivable concentration",
            "premium cost barrier to hedging",
        ],
    },
    "industry_furniture_usd_mismatch_trap": {
        "business_type": "Heavy Furniture Long-Cycle Supply Chain",
        "pain_points": [
            "90-120 day cash conversion elongation",
            "fixed forward delivery mismatch",
            "liquidity exhaustion at contract maturity",
        ],
    },
    "industry_apparel_eur_high_return_trap": {
        "business_type": "Fast Fashion E-Commerce",
        "pain_points": [
            "30-50% return rate phantom exposure",
            "EUR gross-checkout over-hedge",
            "reverse logistics cash shrinkage",
        ],
    },
    "amazon_uk_gbp_vat_funnel": {
        "business_type": "Amazon UK Marketplace",
        "pain_points": [
            "GBP VAT facilitator leakage",
            "14-day bi-weekly payout lag",
            "DD+7 reserve timing mismatch",
        ],
    },
    "derivative_zero_cost_collar_strategy": {
        "business_type": "FX Derivatives Treasury",
        "pain_points": [
            "upside truncation on receivable conversion",
            "contract maturity alignment risk",
            "breakeven floor breach exposure",
        ],
    },
    "derivative_fx_swap_liquidity_misalignment": {
        "business_type": "Multicurrency Cash Optimization",
        "pain_points": [
            "localized currency shortfall",
            "dual-layer conversion spread loss",
            "swap point mispricing risk",
        ],
    },
    "derivative_forward_rollover_early_delivery": {
        "business_type": "Flexible Treasury Operations",
        "pain_points": [
            "supply chain delay rollover pressure",
            "forward contract hard default risk",
            "cash flow timing mismatch",
        ],
    },
    "macro_fed_fomc_nfp_volatility_shield": {
        "business_type": "Macro Event Risk Management",
        "pain_points": [
            "FOMC intraday volatility spike",
            "NFP release slippage exposure",
            "naked spot settlement during macro shock",
        ],
    },
    "macro_ecb_boe_joint_divergence_shield": {
        "business_type": "European Macro Treasury",
        "pain_points": [
            "EUR/GBP policy divergence scissor spread",
            "unified hedge ratio distortion",
            "asymmetric central bank transmission",
        ],
    },
    "macro_pboc_counter_cyclical_factor_shield": {
        "business_type": "Sovereign Intervention Risk",
        "pain_points": [
            "CNH policy squeeze reversal",
            "fixing deviation vs spot gap",
            "offshore HIBOR liquidity spike",
        ],
    },
    "macro_geopolitical_tail_risk_freight_shield": {
        "business_type": "Geopolitical Supply Chain Risk",
        "pain_points": [
            "freight cost hyper-inflation impulse",
            "spot USD liquidity drain for logistics",
            "customs detention delivery failure",
        ],
    },
    "macro_boj_carry_trade_unwinding_shield": {
        "business_type": "Global Liquidity Shock",
        "pain_points": [
            "yen carry unwind margin liquidation",
            "passive CNH surge on USD liquidation",
            "global capital relocation shock",
        ],
    },
    "fx_microstructure_liquidity_squeeze": {
        "business_type": "CNH Microstructure Risk",
        "pain_points": [
            "CNH HIBOR squeeze acceleration",
            "CCS dislocation from CIP parity",
            "offshore pooling liquidity evaporation",
        ],
    },
    "compliance_safe_three_way_matching": {
        "business_type": "SAFE Regulatory Compliance",
        "pain_points": [
            "three-way matching deviation",
            "derivative delivery without order flow",
            "clearing bank settlement freeze",
        ],
    },
    "compliance_aml_gateway_freeze_contagion": {
        "business_type": "Payment Gateway AML Risk",
        "pain_points": [
            "gateway fund freeze contagion",
            "payout velocity breakdown",
            "rigid forward on frozen cash pool",
        ],
    },
    "accounting_ifrs9_hedge_effectiveness": {
        "business_type": "Hedge Accounting Governance",
        "pain_points": [
            "hedge effectiveness test failure",
            "P&L volatility from ineffective hedge",
            "audit trail documentation gap",
        ],
    },
}
def _attach_risk_profiles_to_knowledge_base() -> None:
    """Merge structured case metadata into each knowledge-base document."""
    for doc in _FX_KNOWLEDGE_BASE:
        doc_id = str(doc.get("id", ""))
        profile = dict(_KNOWLEDGE_RISK_PROFILE_BY_ID.get(doc_id, {}))
        profile.update(_KNOWLEDGE_CASE_METADATA_BY_ID.get(doc_id, {}))
        if not profile:
            continue

        hedging = profile.get("hedging_solutions") or profile.get("mitigation_frameworks", [])
        profile["hedging_solutions"] = list(hedging)
        profile.setdefault("business_type", str(doc.get("domain", "")))

        doc["risk_profile"] = profile
        doc["risk_domains"] = list(profile.get("risk_domains", []))
        doc["pain_points"] = list(profile.get("pain_points", []))
        doc["hedging_solutions"] = list(profile.get("hedging_solutions", []))
        doc["business_type"] = str(profile.get("business_type", doc.get("domain", "")))
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
def _get_doc_risk_profile(doc: dict[str, Any]) -> dict[str, Any]:
    profile = doc.get("risk_profile")
    return profile if isinstance(profile, dict) else {}
def _doc_case_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Return flattened structured case metadata for knowledge-driven reasoning."""
    profile = _get_doc_risk_profile(doc)
    hedging = doc.get("hedging_solutions") or profile.get("hedging_solutions") or profile.get(
        "mitigation_frameworks", []
    )
    return {
        "risk_domains": list(doc.get("risk_domains") or profile.get("risk_domains", [])),
        "pain_points": list(doc.get("pain_points") or profile.get("pain_points", [])),
        "hedging_solutions": list(hedging),
        "business_type": str(
            doc.get("business_type") or profile.get("business_type") or doc.get("domain", "")
        ),
        "severity": int(profile.get("severity", 3) or 3),
        "affected_entities": list(profile.get("affected_entities", [])),
    }
def _doc_retrieval_weight(doc: dict[str, Any]) -> float:
    """Normalize retrieval relevance to [0, 1] for weighted risk contributions."""
    raw = doc.get("similarity", doc.get("score", 1.0))
    try:
        return min(1.0, max(0.0, float(raw)))
    except (TypeError, ValueError):
        return 1.0
def _knowledge_domain_base_score(domain_name: str) -> int:
    return _KNOWLEDGE_DOMAIN_BASE_SCORES.get(domain_name, 10)
def _knowledge_mitigation_strength(framework_name: str) -> int:
    return _KNOWLEDGE_MITIGATION_FRAMEWORK_STRENGTHS.get(framework_name, 8)


_attach_risk_profiles_to_knowledge_base()
