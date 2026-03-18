import logging
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from hard_filter import parse_address, COUNTRY_CODE_TO_NAME

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model ({model_name})...")
        self.model = SentenceTransformer(model_name)

    def _address_text(self, raw) -> str | None:
        parsed = parse_address(raw)
        if not parsed:
            return None

        town = parsed.get("town") or parsed.get("city")
        region = parsed.get("region_name")
        code = (parsed.get("country_code") or "").lower()
        country = COUNTRY_CODE_TO_NAME.get(code)

        parts = []
        if town:
            parts.append(town)
        if region and region != town:
            parts.append(region)
        if country:
            parts.append(country)
        elif code:
            parts.append(code.upper())

        if not parts and "raw" in parsed:
            return parsed["raw"]

        return ", ".join(parts) if parts else None

    def _company_text(self, row: pd.Series) -> str:
        parts = []

        if pd.notna(row.get("operational_name")):
            parts.append(str(row["operational_name"]))

        if pd.notna(row.get("description")):
            parts.append(str(row["description"]))

        for field in ("core_offerings", "target_markets", "business_model"):
            val = row.get(field)
            if val:
                parts.append(
                    ", ".join(str(v) for v in val)
                    if isinstance(val, list)
                    else str(val)
                )

        naics = row.get("primary_naics")
        if isinstance(naics, dict) and "label" in naics:
            parts.append(naics["label"])

        for item in row.get("secondary_naics") or []:
            if isinstance(item, dict) and "label" in item:
                parts.append(item["label"])

        addr = self._address_text(row.get("address"))
        if addr:
            parts.append(addr)

        return ". ".join(filter(None, parts))

    def rank(self, df: pd.DataFrame, query: str, top_n: int = 20) -> pd.DataFrame:
        if df.empty:
            return df

        texts = df.apply(self._company_text, axis=1).tolist()
        query_emb = self.model.encode([query], show_progress_bar=False)
        company_embs = self.model.encode(texts, show_progress_bar=False, batch_size=64)

        scores = cosine_similarity(query_emb, company_embs)[0]

        result = df.copy()
        result["similarity_score"] = scores
        return result.nlargest(top_n, "similarity_score").reset_index(drop=True)
