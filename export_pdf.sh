#!/usr/bin/env bash
# Export a deck to slides/slides_<deck>.pdf with headless Chrome, using
# reveal.js's print mode (?print-pdf): one page per slide, every fragment
# shown, no speaker notes.
#   ./export_pdf.sh                     latest draft: re-render it, then export
#   ./export_pdf.sh draft1              re-render draft1, then export
#   ./export_pdf.sh draft1 --no-render  export the slides_draft1.html that's already there
# Re-rendering here never regenerates the figures; run ./render.sh for that.
#
# Works in WSL (uses the Windows Chrome or Edge install) or on Linux with
# google-chrome / chromium on the PATH.
set -euo pipefail
cd "$(dirname "$0")"

render=1
deck=""
for arg in "$@"; do
  case "$arg" in
    --no-render) render=0 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *) deck="$(basename "$arg" .qmd)"; deck="${deck#slides_}" ;;
  esac
done
if [[ -z "$deck" ]]; then
  # default: the highest-numbered draft
  latest="$(ls slides/slides_*.qmd | sort -V | tail -n 1)"
  deck="$(basename "$latest" .qmd)"; deck="${deck#slides_}"
fi
[[ -f "slides/slides_${deck}.qmd" ]] || { echo "no such deck: slides/slides_${deck}.qmd" >&2; exit 1; }

if (( render )); then
  ./render.sh "$deck" --no-figures
fi

html="$PWD/slides/slides_${deck}.html"
pdf="$PWD/slides/slides_${deck}.pdf"

find_browser() {
  local c
  for c in google-chrome google-chrome-stable chromium chromium-browser; do
    command -v "$c" >/dev/null && { echo "$c"; return; }
  done
  for c in "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
           "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
           "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
           "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"; do
    [[ -x "$c" ]] && { echo "$c"; return; }
  done
  echo "no Chrome/Chromium/Edge found" >&2
  exit 1
}
browser="$(find_browser)"

if [[ "$browser" == *.exe ]]; then
  # Windows browser: give it Windows-style paths into the WSL filesystem
  url="file://wsl.localhost/${WSL_DISTRO_NAME}${html}?print-pdf"
  out="$(wslpath -w "$pdf")"
else
  url="file://${html}?print-pdf"
  out="$pdf"
fi

echo "== pdf ($(basename "$browser"))"
# virtual-time-budget: give MathJax and the images time to finish first.
# Headless Chrome now and then prints a blank page; retry if the PDF is tiny.
for attempt in 1 2 3; do
  "$browser" --headless=new --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=20000 --print-to-pdf="$out" "$url" 2>/dev/null || true
  size=$(stat -c %s "$pdf" 2>/dev/null || echo 0)
  (( size > 100000 )) && { echo "done: slides/slides_${deck}.pdf"; exit 0; }
  echo "blank PDF (attempt $attempt), retrying..."
done
echo "PDF export failed" >&2
exit 1
