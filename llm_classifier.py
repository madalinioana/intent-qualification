import json
import pandas as pd
from groq import Groq

from hard_filter import parse_address, COUNTRY_CODE_TO_NAME


SYSTEM_PROMPT = """
You are a company qualification expert with deep knowledge of industries and business models.

User query: "{query}"

For each company below, decide if it truly satisfies every criterion in the query. Think step by step and check all conditions: industry, geography, size, public/private status, etc. A company that matches most but not all criteria should get a lower score.

CRITICAL RULES FOR EVALUATION:
1. STRICT EVIDENCE ("TRULY MATCH"): You must base your score ONLY on the provided data. If the query asks for a specific technology, software, or platform (e.g., 'Shopify', 'Salesforce'), and there is NO explicit evidence in the company's description or offerings that they use it or provide it, you MUST score them 0-39.
2. NO GUESSING: DO NOT assume a company uses a platform just because of their size or industry (e.g., do not assume a small e-commerce store uses Shopify without proof). 
3. ENTERPRISE COMMON SENSE: Massive global enterprises build proprietary software. They do not use off-the-shelf SMB platforms.

SCORING RUBRIC:
- 80-100: Perfect match. Definitively satisfies all criteria based on explicit evidence.
- 60-79: Good match. Meets core intent explicitly, with only minor deviations.
- 40-59: Poor match. Lacks explicit evidence for key requirements.
- 0-39: Irrelevant, contradicts the query, or relies on assumptions/guessing without data.

Companies with score >= {qualify_threshold} are qualified. Evaluate each company independently, position in list should not matter. Use industry_naics as the authoritative industry classification.

Return ONLY a JSON object in this exact format:
{{
  "results": [
    {{
      "name": "company name",
      "score": <integer 0-100>,
      "qualified": <true or false>,
      "reason": "one sentence with specific evidence"
    }}
  ]
}}

Companies to evaluate:
{companies}
"""


def _country_name(raw) -> str | None:
    code = (parse_address(raw).get("country_code") or "").lower()
    return COUNTRY_CODE_TO_NAME.get(code)


class LLMClassifier:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile"
        self.qualify_threshold = 60
        self.system_prompt = SYSTEM_PROMPT

    def _company_summary(self, row: pd.Series) -> dict:
        naics = row.get("primary_naics")
        if isinstance(naics, dict):
            naics_label = naics.get("label")
        elif isinstance(naics, str) and naics:
            naics_label = naics
        else:
            naics_label = None

        return {
            "name": row.get("operational_name", "Unknown"),
            "country": _country_name(row.get("address")),
            "description": row.get("description", ""),
            "industry_naics": naics_label,
            "core_offerings": row.get("core_offerings", []),
            "target_markets": row.get("target_markets", []),
            "business_model": row.get("business_model", []),
            "employee_count": row.get("employee_count"),
            "revenue_usd": row.get("revenue"),
            "year_founded": row.get("year_founded"),
            "is_public": row.get("is_public"),
        }

    def classify(self, query: str, candidates_df: pd.DataFrame) -> pd.DataFrame:
        if candidates_df.empty:
            return candidates_df

        companies = [self._company_summary(row) for _, row in candidates_df.iterrows()]

        prompt = self.system_prompt.format(
            query=query,
            qualify_threshold=self.qualify_threshold,
            companies=json.dumps(companies, indent=2, default=str),
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_id,
                response_format={"type": "json_object"},
                temperature=0,
            )

            results = json.loads(response.choices[0].message.content).get("results", [])
            score_map = {r["name"]: r for r in results}

            df = candidates_df.copy()
            df["llm_score"] = df["operational_name"].map(
                lambda n: score_map.get(n, {}).get("score", 0)
            )
            df["qualified"] = df["operational_name"].map(
                lambda n: score_map.get(n, {}).get("qualified", False)
            )
            df["llm_reason"] = df["operational_name"].map(
                lambda n: score_map.get(n, {}).get("reason", "")
            )

            return (
                df[df["qualified"]]
                .drop_duplicates(subset=["operational_name"])
                .sort_values("llm_score", ascending=False)
                .reset_index(drop=True)
            )

        except Exception as e:
            print(f"LLM classifier failed: {e}")
            df = candidates_df.copy()
            df["llm_score"] = 0
            df["qualified"] = True
            df["llm_reason"] = "LLM evaluation unavailable"
            return df
