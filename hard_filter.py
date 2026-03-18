import ast
import pandas as pd

COUNTRY_TO_CODES = {
    "albania": "al",
    "andorra": "ad",
    "austria": "at",
    "belarus": "by",
    "belgium": "be",
    "bosnia": "ba",
    "bulgaria": "bg",
    "croatia": "hr",
    "cyprus": "cy",
    "czech republic": "cz",
    "czechia": "cz",
    "denmark": "dk",
    "estonia": "ee",
    "finland": "fi",
    "france": "fr",
    "germany": "de",
    "greece": "gr",
    "hungary": "hu",
    "iceland": "is",
    "ireland": "ie",
    "italy": "it",
    "latvia": "lv",
    "liechtenstein": "li",
    "lithuania": "lt",
    "luxembourg": "lu",
    "malta": "mt",
    "moldova": "md",
    "monaco": "mc",
    "montenegro": "me",
    "netherlands": "nl",
    "north macedonia": "mk",
    "norway": "no",
    "poland": "pl",
    "portugal": "pt",
    "romania": "ro",
    "russia": "ru",
    "san marino": "sm",
    "serbia": "rs",
    "slovakia": "sk",
    "slovenia": "si",
    "spain": "es",
    "sweden": "se",
    "switzerland": "ch",
    "ukraine": "ua",
    "united kingdom": "gb",
    "uk": "gb",
    "kosovo": "xk",
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "canada": "ca",
    "australia": "au",
    "new zealand": "nz",
    "japan": "jp",
    "china": "cn",
    "india": "in",
    "brazil": "br",
    "mexico": "mx",
    "south korea": "kr",
    "korea": "kr",
    "singapore": "sg",
    "israel": "il",
    "europe": [
        "al",
        "ad",
        "at",
        "by",
        "be",
        "ba",
        "bg",
        "hr",
        "cy",
        "cz",
        "dk",
        "ee",
        "fi",
        "fr",
        "de",
        "gr",
        "hu",
        "is",
        "ie",
        "it",
        "xk",
        "lv",
        "li",
        "lt",
        "lu",
        "mt",
        "md",
        "mc",
        "me",
        "nl",
        "mk",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sm",
        "rs",
        "sk",
        "si",
        "es",
        "se",
        "ch",
        "ua",
        "gb",
    ],
    "scandinavia": ["se", "no", "dk"],
    "nordic": ["se", "no", "dk", "fi", "is"],
    "benelux": ["be", "nl", "lu"],
    "dach": ["de", "at", "ch"],
    "baltics": ["ee", "lv", "lt"],
    "balkans": ["al", "ba", "bg", "hr", "gr", "xk", "me", "mk", "ro", "rs", "si"],
}

COUNTRY_CODE_TO_NAME = {
    "ro": "Romania",
    "fr": "France",
    "de": "Germany",
    "ch": "Switzerland",
    "us": "United States",
    "gb": "United Kingdom",
    "it": "Italy",
    "es": "Spain",
    "nl": "Netherlands",
    "pl": "Poland",
    "se": "Sweden",
    "no": "Norway",
    "dk": "Denmark",
    "fi": "Finland",
    "be": "Belgium",
    "at": "Austria",
    "pt": "Portugal",
    "cz": "Czech Republic",
    "hu": "Hungary",
    "bg": "Bulgaria",
    "hr": "Croatia",
    "sk": "Slovakia",
    "si": "Slovenia",
    "ee": "Estonia",
    "lv": "Latvia",
    "lt": "Lithuania",
    "lu": "Luxembourg",
    "ie": "Ireland",
    "gr": "Greece",
    "sg": "Singapore",
    "il": "Israel",
    "jp": "Japan",
    "cn": "China",
    "in": "India",
    "br": "Brazil",
    "ca": "Canada",
    "au": "Australia",
}


def parse_address(raw) -> dict:
    if raw is None or isinstance(raw, float):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                return ast.literal_eval(raw)
            except Exception:
                pass
        return {"raw": raw}
    return {}


def _address_matches(raw, query_value: str) -> bool:
    if raw is None or isinstance(raw, float):
        return True

    q = query_value.strip().lower()
    target_codes = COUNTRY_TO_CODES.get(q)
    code = (parse_address(raw).get("country_code") or "").lower()

    if target_codes is not None:
        if isinstance(target_codes, str):
            return code == target_codes
        return code in target_codes

    return query_value.lower() in str(raw).lower()


class HardFilter:
    def apply(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        if not filters:
            return df.copy()

        result = df.copy()

        for key, value in filters.items():
            if value is None:
                continue

            if key.startswith("min_"):
                col = key[4:]
                if col in result.columns:
                    result = result[(result[col] >= value) | result[col].isna()]

            elif key.startswith("max_"):
                col = key[4:]
                if col in result.columns:
                    result = result[(result[col] <= value) | result[col].isna()]

            elif key == "address" and isinstance(value, str):
                result = result[
                    result["address"].apply(lambda a: _address_matches(a, value))
                ]

            elif key in result.columns:
                if isinstance(value, bool):
                    result = result[(result[key] == value) | result[key].isna()]
                elif isinstance(value, str):
                    result = result[
                        result[key].str.contains(value, case=False, na=True)
                    ]
                elif isinstance(value, (int, float)):
                    result = result[(result[key] == value) | result[key].isna()]

        return result
