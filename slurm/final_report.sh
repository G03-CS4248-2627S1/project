#!/bin/bash
# The tables of final scoring (#115): each system's mean ± std over its seeds and the contrasts' paired bootstrap
# intervals, on each final set, and on dev against the first reference alone. CPU only, from results/, so run it
# anywhere that has the runs' predictions and data/external. Writes reports/final_scoring/<set>.json and tables.md.
#   bash slurm/final_report.sh            # PYTHON=<interpreter> to use another one
#
# Systems: R0 and R0mbr, the standard head; R1 and R3, the joint head (rows 1 and 3); R4map and R4, the separate
# verifier (row 4); R5map and R5, the row 5 reader. Row 4 stands on reader seed 0 alone, so its contrasts use R1s0
# and R3s0. Row 0* (R0s) joins when its runs have been scored: set ROW0STAR to its run ids, comma-separated.
set -euo pipefail
python=${PYTHON:-python}
row0=2026-10-05-row0-baseline-s0-j910132,2026-10-05-row0-baseline-s1-j910133,2026-10-05-row0-baseline-s2-j910134
row1=2026-10-23-row1-joint-s0-j998373,2026-10-23-row1-joint-s1-j998384,2026-10-23-row1-joint-s2-j998397
row1s0=2026-10-23-row1-joint-s0-j998373
row4=2026-10-29-verifier-separate-s0-j1032972,2026-10-30-verifier-separate-s1-j1041306,2026-10-30-verifier-separate-s2-j1041319
row5=2026-10-30-row1-joint-aqa-s0-j1045742,2026-10-30-row1-joint-aqa-s1-j1045747,2026-10-30-row1-joint-aqa-s2-j1045751
systems=(R0=map:$row0 R0mbr=mbr:$row0 R1=map:$row1 R3=mbr:$row1 R1s0=map:$row1s0 R3s0=mbr:$row1s0
         R4map=combined_map:$row4 R4=combined_mbr:$row4 R5map=map:$row5 R5=mbr:$row5)
contrasts=(--contrast R1-R0 --contrast R3-R1 --contrast R4map-R1s0 --contrast R4-R3s0 --contrast R5-R3)
if [ -n "${ROW0STAR:-}" ]; then
  systems+=(R0s=map:$ROW0STAR)
  contrasts+=(--contrast R1-R0s)
fi
out=reports/final_scoring
mkdir -p "$out"
: > "$out/tables.md"
score() {  # score <title> <json name> <compare options...>
  local title=$1 name=$2
  shift 2
  echo "== $title" >&2
  { printf '## %s\n\n' "$title"
    PYTHONIOENCODING=utf-8 "$python" -m squadqa.report compare "${systems[@]}" "${contrasts[@]}" --final -o "$out/$name.json" "$@"
    printf '\n'
  } >> "$out/tables.md"
}
score "dev, all references" dev --split dev
score "dev, first reference only" dev-first-reference --split dev --first-reference
score "AddSent, worst case over each question's versions" addsent --split addsent
score "AddOneSent, worst case over each question's versions" addonesent --split addonesent
for s in new_wiki nyt reddit amazon_reviews; do
  score "SQuAD-Shifts $s" "squadshifts-$s" --split "squadshifts-$s"
done
