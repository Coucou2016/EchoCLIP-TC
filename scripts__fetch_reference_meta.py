"""Fetch and print complete Crossref metadata for a fixed DOI list."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

UA = "EchoCLIP-TA-paper/1.0 (academic verification)"

DOIS = [
    "10.1038/s41586-020-2145-8",
    "10.1038/s41591-024-02959-y",
    "10.1007/978-3-032-04971-1_5",
    "10.1109/TMI.2019.2900516",
    "10.48550/arXiv.2103.00020",
    "10.48550/arXiv.1706.04599",
    "10.1561/2200000101",
    "10.48550/arXiv.2004.11362",
    "10.48550/arXiv.1706.03762",
    "10.1109/CVPR52688.2022.01167",
    "10.1038/s41746-018-0065-x",
    "10.1038/s41467-025-66340-4",
    "10.1038/s43856-024-00538-3",
]


def get_crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["message"]


def get_datacite(doi: str) -> dict:
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["data"]["attributes"]


def main() -> None:
    for doi in DOIS:
        print("=" * 100)
        print("DOI:", doi)
        try:
            if doi.startswith("10.48550/"):
                a = get_datacite(doi)
                print("  TITLE :", (a.get("titles") or [{}])[0].get("title", ""))
                print("  YEAR  :", a.get("publicationYear"), "| PUB:", a.get("publisher"))
                print("  AUTH  :", [c.get("name") for c in (a.get("creators") or [])[:4]])
                print("  TYPE  :", (a.get("types") or {}).get("resourceTypeGeneral"))
            else:
                m = get_crossref(doi)
                print("  TITLE :", (m.get("title") or [""])[0])
                print(
                    "  VENUE :",
                    (m.get("container-title") or [""])[0],
                    "|",
                    m.get("issued", {}).get("date-parts"),
                )
                print(
                    "  AUTH  :",
                    [
                        f"{a.get('given','')} {a.get('family','')}".strip()
                        for a in (m.get("author") or [])[:5]
                    ],
                )
                print(
                    "  VOL   :",
                    m.get("volume"),
                    "| ISSUE:",
                    m.get("issue"),
                    "| PAGE:",
                    m.get("page"),
                    "| ART:",
                    m.get("article-number"),
                )
                print("  TYPE  :", m.get("type"), "| refs:", m.get("is-referenced-by-count"))
                lic = [l.get("URL") for l in (m.get("license") or [])]
                print("  LICENSE:", lic[:2])
                print("  URL   :", m.get("URL"))
        except Exception as exc:  # noqa: BLE001
            print("  ERROR:", type(exc).__name__, exc)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
