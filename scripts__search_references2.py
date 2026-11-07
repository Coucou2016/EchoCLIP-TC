"""Targeted Crossref queries for the remaining citations."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

UA = "EchoCLIP-TA-paper/1.0 (academic verification)"


def search(q: str, rows: int = 8, filt: str = "") -> list:
    params = {"query.bibliographic": q, "rows": rows, "select": "DOI,title,container-title,issued,author,type"}
    if filt:
        params["filter"] = filt
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        d = json.loads(r.read().decode("utf-8", "replace"))
    out = []
    for it in d.get("message", {}).get("items", []):
        out.append(
            {
                "doi": it.get("DOI"),
                "title": (it.get("title") or [""])[0],
                "venue": (it.get("container-title") or [""])[0],
                "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
                "type": it.get("type"),
                "author": (it.get("author") or [{}])[0].get("family", ""),
            }
        )
    return out


QUERIES = [
    ("echoprime_title", "EchoPrime a foundation model for echocardiography", ""),
    ("echoprime_authors", "Vaid echocardiography foundation model", ""),
    ("cardiacclip_exact", "CardiacCLIP", ""),
    ("cardiacclip_miccai", "cardiac vision language pretraining echocardiography MICCAI", ""),
    ("echoclip_natmed", "Vision-language foundation model for echocardiogram interpretation", ""),
    ("ef_deep_1", "deep learning ejection fraction echocardiography mortality JACC", ""),
    ("ef_deep_2", "artificial intelligence echocardiography left ventricular systolic function Lancet Digital Health", ""),
    ("ef_deep_3", "video based deep learning echocardiography cardiac function npj Digital Medicine", ""),
    ("tts", "temperature scaling calibration neural network confidence", ""),
    ("conformal_med", "conformal prediction medical imaging uncertainty", ""),
    ("clip_med", "vision language model medical imaging contrastive pretraining", ""),
]

for key, q, filt in QUERIES:
    print(f"\n### {key}: {q}", flush=True)
    try:
        for it in search(q, filt=filt):
            print(
                f"  {str(it['year']):5s} {str(it['doi']):42s} {str(it['type'])[:12]:12s} "
                f"{it['venue'][:38]:38s} {it['title'][:78]}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
    time.sleep(0.4)
