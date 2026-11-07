"""Crossref title search → candidate DOIs, for citations whose DOI must be found.

Prints the top hits with title/venue/year/DOI so the author can pick the real
record instead of guessing.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

UA = "EchoCLIP-TA-paper/1.0 (academic verification)"

QUERIES = [
    ("echoclip", "EchoCLIP echocardiography foundation model Nature Medicine"),
    ("echoclip2", "Christensen echocardiogram vision language model external validation"),
    ("echoprime", "EchoPrime foundation model echocardiography"),
    ("cardiacclip", "CardiacCLIP echocardiography contrastive"),
    ("nolan_ef", "echocardiographic assessment left ventricular ejection fraction guidelines"),
    ("oala", "machine learning echocardiography review"),
    ("ouyang_dd", "video-based artificial intelligence diastolic function echocardiography"),
    ("lvedl", "deep learning left ventricular ejection fraction echocardiography prognosis"),
    ("echonet_review", "EchoNet-Dynamic cardiac function deep learning validation"),
]


def search(q: str, rows: int = 6) -> list:
    url = (
        "https://api.crossref.org/works?"
        + urllib.parse.urlencode({"query.bibliographic": q, "rows": rows, "select": "DOI,title,container-title,issued,author,type"})
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    items = d.get("message", {}).get("items", [])
    out = []
    for it in items:
        out.append(
            {
                "doi": it.get("DOI"),
                "title": (it.get("title") or [""])[0],
                "venue": (it.get("container-title") or [""])[0],
                "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "type": it.get("type"),
                "first_author": (
                    f"{it['author'][0].get('given','')} {it['author'][0].get('family','')}".strip()
                    if it.get("author")
                    else ""
                ),
            }
        )
    return out


def main() -> int:
    for key, q in QUERIES:
        print(f"\n### {key}: {q}", flush=True)
        try:
            for it in search(q):
                print(
                    f"  {str(it['year']):5s} {str(it['doi']):40s} {it['venue'][:42]:42s} {it['title'][:80]}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
