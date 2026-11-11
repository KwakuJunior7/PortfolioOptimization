"""Build the static, self-contained HTML preview (docs/preview.html) from reports/results.json.

    python run.py               # writes reports/results.json
    python scripts/build_preview.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(results=os.path.join(ROOT, "reports", "results.json"),
         template=os.path.join(ROOT, "docs", "preview_template.html"),
         out=os.path.join(ROOT, "docs", "preview.html")):
    with open(results) as fh:
        data = json.load(fh)
    with open(template, encoding="utf-8") as fh:
        html = fh.read()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__RESULTS__*/null", payload)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Preview written to {out} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main(*sys.argv[1:])
