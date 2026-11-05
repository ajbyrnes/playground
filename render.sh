#!/usr/bin/env bash
# Rebuild the slide figures and re-render the deck(s).
#   ./render.sh                          figures + every deck
#   ./render.sh draft2                   figures + slides/slides_draft2.qmd
#   ./render.sh draft1 draft2            figures + both drafts
#   ./render.sh draft2 --no-figures      slides only (skip regenerating figures)
#   ./render.sh --figures-only           figures only, no decks
# A deck can be named as draft2, slides_draft2, or slides/slides_draft2.qmd.
# --slides is kept as an alias for --no-figures.
set -euo pipefail
cd "$(dirname "$0")"

figures=1
slides=1
decks=()
for arg in "$@"; do
  case "$arg" in
    --no-figures|--slides) figures=0 ;;
    --figures-only) slides=0 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option: $arg" >&2; exit 1 ;;
    *)
      name="$(basename "$arg" .qmd)"
      name="${name#slides_}"
      decks+=("slides/slides_${name}.qmd")
      ;;
  esac
done

# no deck named: render all of them
if (( ${#decks[@]} == 0 )); then
  decks=(slides/slides_*.qmd)
fi

for deck in "${decks[@]}"; do
  [[ -f "$deck" ]] || { echo "no such deck: $deck" >&2; exit 1; }
done

if (( figures )); then
  echo "== figures"
  .venv/bin/python figures/scripts/foundations.py
  .venv/bin/python figures/scripts/adp.py
  .venv/bin/python figures/scripts/applications.py
  # boxed copies of Ozaki et al.'s tables (needs the extracted figures in figures/papers/)
  .venv/bin/python figures/scripts/highlight_tables.py
fi

if (( slides )); then
  for deck in "${decks[@]}"; do
    echo "== $deck"
    quarto render "$deck"
    echo "done: ${deck%.qmd}.html"
  done
fi
