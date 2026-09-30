"""Render paper/Prometheus_alpha_paper.md to HTML (python-markdown). For the PDF, print the HTML with a
headless browser, e.g. Playwright's page.pdf(format="A4").

    python tools/build_paper.py
"""
import pathlib

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
STYLE = ("body{font:10.5pt/1.5 Georgia,serif;max-width:46em;margin:2em auto;color:#111}h1{font-size:18pt;line-height:1.2}"
         "h2{font-size:13pt;margin-top:1.6em}h3{font-size:11pt}\n"
         "img{max-width:100%}table{border-collapse:collapse;font:8.5pt/1.3 sans-serif;margin:1em 0}"
         "td,th{border-bottom:1px solid #ccc;padding:3px 6px;text-align:left}\n"
         "blockquote{margin:0;padding:.4em 1em;background:#f4f4f2;font-size:9.5pt}em{color:#333}code{font-size:9pt}")

src = (ROOT / "paper" / "Prometheus_alpha_paper.md").read_text(encoding="utf-8")
body = markdown.markdown(src, extensions=["tables"])
html = f"<!doctype html><meta charset=utf-8><title>Prometheus-alpha paper</title><style>{STYLE}</style>{body}\n"
(ROOT / "paper" / "Prometheus_alpha_paper.html").write_text(html, encoding="utf-8")
print("wrote paper/Prometheus_alpha_paper.html")
