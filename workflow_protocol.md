# Operational Protocol: Nursing & Finance Management

Agency upgrade: use `agency_protocol.md` for bounded autonomy, trigger handling, causal reasoning, and action package requirements. Use `action-package-template.md` whenever a proactive recommendation affects money, taxes, benefits, school performance, clinical obligations, legal status, or long-term strategy.

## 1. The High-Signal Workflow (Partnership Model)
This protocol defines the division of labor between the User and the AI to prevent "data swamp" (too much low-quality data) and maximize learning/tracking efficiency.

### **User Role: The Curator (Human Intelligence)**
*   **Active Learning:** Perform primary study/research.
*   **Note Creation:** Create lightweight, high-signal Markdown notes in Obsidian when key concepts are identified.
*   **Raw Ingestion:** Paste raw lecture transcripts, large PDFs, or unorganized text into `.md` files within the `Nursing/` directory.
*   **Trigger Identification:** Manually flag significant financial changes or new strategic targets.

### **AI Role: The Analyst & Examiner (Artificial Intelligence)**
*   **Automated Auditing:** Periodically scan the `Nursing/` and `Finance/` directories to identify new content and cross-reference it with existing notes.
*   **Data Refinement:** Transform "low-signal" raw text (transcripts) into structured, metadata-rich Markdown (extracting interventions, rationales, and key terms).
*   **NCLEX Simulation:** Generate one-at-a-time, NCLEX-style practice questions derived *specifically* from the user's curated notes to ensure relevance.
*   **Gap Analysis:** Monitor the "Nursing" folder for thematic voids (e.g., "You have high coverage in Cardiac, but low coverage in Renal") and alert the User.
*   **Financial Watchdog:** Monitor price-based triggers (e.g., a fund reaching a pre-set re-entry price) and calendar deadlines.
*   **Action Packaging:** When a meaningful trigger appears, prepare a complete decision package with facts, assumptions, numbers, consequences, alternatives, and an approval gate rather than surfacing a vague suggestion.
*   **Cross-Domain Integration:** Treat nursing workload, benefits timing, tax-home preservation, cash flow, and portfolio-management capacity as one connected system.

## 2. Data Ingestion Pipeline
1.  **Input:** Raw text/transcripts $\rightarrow$ `.md` file in `Nursing/`.
2.  **Notification:** User notifies AI of new or updated content.
3.  **Processing:** AI performs extraction, structural formatting (YAML metadata), and integration into the existing Knowledge Map.

## 3. Communication Preferences
*   **NCLEX Training:** One question at a time; focus heavily on the **rationale**.
*   **Auditing:** The AI is authorized to audit the directory structure without explicit permission for every check.
*   **Priority:** Critical financial tasks and nursing interventions take highest priority in summaries.
*   **Agency:** The AI may prepare recommendations, drafts, and local file updates, but the user must explicitly approve external actions, trades, submissions, account changes, or messages.

***
