import json
import os

import pandas as pd
from dotenv import load_dotenv
from intent_parser import IntentParser
from hard_filter import HardFilter
from embedder import Embedder
from llm_classifier import LLMClassifier

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

QUERIES = [
    "Clean energy startups founded after 2018 with fewer than 200 employees",
]

TOP_N_CONFIG = {
    "structured": 10,
    "semantic": 15,
    "ecosystem": 20,
}


def run_query(query, df, parser, hard_filter, embedder, classifier):
    print(f"\nQuery: {query}")

    parsed = parser.parse(query)
    query_type = parsed["query_type"]
    semantic_intent = parsed["semantic_intent"]
    filters = parsed["filters"]

    print(f"  type: {query_type}")
    print(f"  filters: {json.dumps(filters)}")
    print(f"  intent: {semantic_intent}")

    filtered_df = hard_filter.apply(df, filters)
    print(f"  {len(df)} -> {len(filtered_df)} after hard filter")

    if filtered_df.empty:
        print("  hard filter removed everything, falling back to full dataset")
        filtered_df = df.copy()

    top_n = TOP_N_CONFIG.get(query_type, 20)
    ranked_df = embedder.rank(filtered_df, semantic_intent, top_n=top_n)
    print(f"  {len(filtered_df)} -> top {len(ranked_df)} by similarity")

    qualified_df = classifier.classify(query, ranked_df)
    print(f"  {len(ranked_df)} -> {len(qualified_df)} qualified by LLM")

    return qualified_df


def display_results(query, results_df):
    if results_df.empty:
        print("\n  No qualified companies found.")
        return

    print(f"\n  {len(results_df)} qualified:")
    for _, row in results_df.iterrows():
        print(f"    {row['operational_name']} - {row['llm_score']} score")


def main():
    df = pd.read_json("companies.jsonl", lines=True)
    print(f"Loaded {len(df)} companies\n")

    parser = IntentParser(GROQ_API_KEY)
    hard_filter = HardFilter()
    embedder = Embedder()
    classifier = LLMClassifier(GROQ_API_KEY)

    all_results = {}
    for query in QUERIES:
        results = run_query(query, df, parser, hard_filter, embedder, classifier)
        all_results[query] = results
        display_results(query, results)

    output = {}
    for query, df_result in all_results.items():
        save_cols = [
            c
            for c in ["operational_name", "llm_score", "llm_reason"]
            if c in df_result.columns
        ]
        output[query] = (
            df_result[save_cols].to_dict(orient="records")
            if not df_result.empty
            else []
        )

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)

    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
