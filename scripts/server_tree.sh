#!/usr/bin/env bash
# Read-only listing of what is on the server.
#
#   bash ~/code/peptidefusion/scripts/server_tree.sh
#
# Writes $DATA/results/server_tree.txt (a few hundred lines; the du part takes a minute or
# two).
#
# Not listed, counted instead: miniconda3, snap, hidden entries in home, .git, __pycache__,
# cache/, the contents of *_shards/ and the part_*.parquet downloads.
source "$(dirname "$0")/env.sh"
OUT=$DATA/results/server_tree.txt
echo "listing the server, du takes a minute or two..."
{
  echo "## HOME $HOME"
  echo "## DATA $DATA"
  echo "## DATE $(date -Is)"
  echo "## FIND type<TAB>bytes<TAB>mtime<TAB>path"
  find "$HOME" "$DATA" \( -path "$HOME/miniconda3" -o -path "$HOME/snap" -o -path "$HOME/.*" \
       -o -path "$DATA/cache" -o -name .git -o -name __pycache__ -o -path '*_shards/*' \
       -o -name 'part_*.parquet' \) -prune -o -printf '%y\t%s\t%TY-%Tm-%Td %TH:%TM\t%p\n'
  echo "## COUNTS entries in directories that were not listed"
  for d in "$DATA"/data/sources/* "$DATA"/data/corpus/sources/*_shards; do
    [ -d "$d" ] && printf '%s\t%s\n' "$(ls "$d" | wc -l)" "$d"
  done
  printf '%s\t%s\n' "$(ls "$DATA/cache" | wc -l)" "$DATA/cache: $(ls "$DATA/cache" | tr '\n' ' ')"
  echo "## DU"
  du -sh "$DATA" "$DATA"/* "$DATA"/data/* "$DATA"/features/* "$HOME" "$HOME"/* "$HOME"/.[!.]* 2>/dev/null
  echo "## DF"
  df -h "$HOME" "$DATA"
} > "$OUT"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
