Welcome to NOVA FX Risk Copilot – an executive dashboard built for cross border SMEs and treasury teams. Our tool helps you monitor FX exposure, quantify risk, and generate actionable hedge recommendations – all in one place.

The sidebar is your control center. Select any of the four supported pairs – USD/CNH, EUR/CNH, HKD/CNH, or GBP/CNH. You will immediately see a live snapshot: spot price, daily change, 5‑day moving average, and the 30‑day price range.
The traffic‑light indicator condenses our multi‑factor risk model into a simple green/yellow/red signal. This uses a weighted composite of market, macro, policy, and knowledge scores – a deterministic engine that never relies on black‑box AI for the final rating.
Below, we provide example questions in plain English – these trigger the RAG‑powered copilot that we’ll cover later.

The main tab is the Live Risk Desk.
At the top, we display four key metrics: spot, day change, MA5, and the desk mode – which switches from “Monitoring” to “Enterprise” once you submit your own exposure scenario.

Underneath, we break down the composite risk into four dimensions:
Market risk – derived from technical indicators like position relative to MA5, MA30, and Bollinger bands.
Macro risk – based on IMF inflation and GDP differentials between the destination country and China.
Policy risk – using the CFETS official USD/CNY parity and the US 10‑year Treasury yield as a FedWatch proxy.
Knowledge risk – from our internal treasury playbook database, weighted by semantic similarity to your query.
Each score is normalised to a 0‑100 scale and then combined with fixed weights (35% market, 25% macro, 25% policy, 15% knowledge).
This transparent formula produces the final composite score and hedge ratio.

Multi‑Factor Technical Engine
The candlestick chart shows 30 days of historical price action.
We overlay the 5‑day moving average – a short‑term trend trigger.
Below, we plot trading volume and daily volatility, both derived from the same OHLCV data.

All live data is fetched via yfinance or Alpha Vantage – we use a fallback chain: Alpha Vantage first, then yfinance, then a deterministic simulator if both are unavailable.
This ensures the dashboard always loads, even during network outages.

Quantitative Backtest Tab
Switch to the Strategy Backtest tab to compare two strategies over the last 180 days:

The Unhedged Spot Benchmark – simply converting monthly USD revenue at the daily spot rate.
The NOVA Dynamic Overlay – which applies a hedge ratio based on the same multi‑factor score:
Score ≥ 0.7 → 80% hedge
Score between 0.4 and 0.7 → 55% hedge
Score < 0.4 → 30% hedge

The effective rate is then calculated as hedge_ratio × MA5 + (1‑hedge_ratio) × spot.
We compute annualised volatility, 95% historical VaR, maximum drawdown, and Sharpe ratio (with a 2% risk‑free rate).
All calculations are done in pure NumPy and Pandas – no external black boxes.

Contract & Invoice Audit Agent
Narrator:
Below the chart, you’ll find the Contract Audit Agent.
Upload an invoice or contract – we support PDF, Excel, CSV, and text files.
Our parser extracts the exposure amount, settlement tenor, and gross margin using regular expressions and, for PDFs, the PyPDF library.
If the document contains clear labels like “total amount” or “net 30 days”, we use those.
Otherwise, we fall back to a deterministic simulation based on the filename and pair – this guarantees you always get a demo result.
The extracted data feeds into a recommended forward hedge ratio and, optionally, an AI‑generated risk review if you supply an OpenAI or DeepSeek API key.

Colloquial FX Risk Copilot (Chat)
The heart of the tool is the conversational copilot.
You can ask plain‑English questions like “I have a 1 million USD payout next month – what should I lock today?” The system first sanitizes your query against prompt‑injection patterns, then builds a dynamic prompt that includes:
Live desk metrics – spot, MA5, MA30, Bollinger, RSI, MACD.
Macro intelligence – IMF CPI/GDP spreads, CFETS parity, US 10Y yield and its death‑cross signal.
Retrieved knowledge – from our internal playbook database, indexed using sentence‑transformers and FAISS for fast vector similarity search.

The RAG (Retrieval‑Augmented Generation) pipeline retrieves the top‑3 most relevant playbooks and injects them into the LLM context. If you have an API key, we call OpenAI or DeepSeek (via the same chat‑completions endpoint) to generate a narrative. If no key is provided, the system falls back to a deterministic rule‑based response – which still includes a quantitative execution ticket. The execution ticket is computed entirely without AI: it parses your stated exposure amount, applies the hedge ratio, and shows exactly how much to forward‑lock and how much to keep unhedged, along with the implied CNH amounts.

Enterprise Risk Assessment & Treasury Copilot
When you submit a scenario – for example, “I’m an Amazon EU seller with 20% VAT” – the system moves from environment monitoring to enterprise assessment. It now computes a knowledge risk score by:
Retrieving related playbooks from the knowledge base.
Extracting structured metadata from those documents: risk domains, mitigation frameworks, and business type.
Aggregating a weighted score based on retrieval confidence, domain severity, and contextual amplifiers (e.g., market volatility or macro alignment).

This deterministic knowledge score is then blended into the composite. If an LLM is available, we also call a separate “knowledge analyst” prompt that suggests a bounded adjustment (between -15 and +15) to the knowledge score. That adjustment is transparently shown in the AI Knowledge Intelligence panel – including per‑domain contributions and confidence.

Additionally, the Treasury Copilot section generates a strategic report – business profile, exposure mapping, similar cases, and a 30‑day watchlist – all using either the LLM or a local rule‑based fallback. At the bottom of the risk desk, you will find the AI Board Report generator.
This is not just another chat response. With a single click on 'Generate AI Report', the system compiles a comprehensive, board-room-ready Markdown document that consolidates every layer of our analysis into one structured file. The report is divided into clear sections:
Executive Summary – a high-level narrative of the current risk posture.
Market, Macro, Policy, and Knowledge Outlooks – each enriched by the AI's interpretive layer.
Recommended Hedge Strategy – including suggested instruments, tenor, and rationale.
Deterministic Scores – our transparent, non-negotiable source-of-truth metrics.
30-Day Watchlist – actionable monitoring priorities for the treasury team.
From a technical standpoint, this feature is built with enterprise-grade robustness:

We enforce a strict session budget (capped at 20 AI calls) to prevent runaway OpenAI costs during demonstrations. All AI-generated reports are cached for one hour based on the scenario fingerprint, so repeated requests return instantly without redundant API charges.

Crucially, the AI enriches the interpretation but never alters the deterministic hedge ratios or composite scores – ensuring compliance with internal risk governance.

Once generated, you can preview the report instantly or download it as a .md file to share with your CFO, risk committee, or external auditors. This bridges the gap between real-time analytics and formal treasury documentation."

Hedge Preference Mapping
The hedge ratio is not fixed – you can customize the mapping between composite score and hedge percentage. The “Risk Preference Mapping” expander lets you adjust the low/high thresholds and the corresponding ratios. Your changes are previewed instantly via a segmented bar and then applied to the main recommendation when you click “Apply”. This empowers you to align the tool with your own risk appetite.

Institutional Data & Ingestion Inspect
Finally, the sidebar offers an “Institutional Data & Ingestion Inspect” panel. You can manually pull the latest IMF macro anchor, the CFETS central parity, the US Treasury yield curve, or a full live‑desk snapshot. This transparency is key – every external data source is cached and clearly labelled, so you always know where the numbers come from.

NOVA FX Risk Copilot combines real‑time market data, fundamental macro anchors, a curated knowledge base of treasury playbooks, and optional AI interpretation – all wrapped in a transparent, deterministic scoring framework.
You can run it entirely offline (local mode) or enhance it with your own OpenAI or DeepSeek key. We designed it for practicality: no black boxes, no hidden costs, and a clear audit trail for every decision. We hope that NOVA can help you navigate FX risk with confidence!
<img width="415" height="686" alt="image" src="https://github.com/user-attachments/assets/9c6526cc-0d69-4e7d-9ee2-610f74253030" />
