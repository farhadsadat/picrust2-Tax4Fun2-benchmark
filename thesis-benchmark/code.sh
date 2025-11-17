# PICRUSt KO header (after PICRUSt2 finishes)
zcat "$OUT/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz" | head -n 1

# Shotgun KO header
head -n 1 "$HOME/thesis-benchmark/shotgun/rhizo_wgs_p.txt"

cat > "$HOME/thesis-benchmark/sample_map.csv" << 'CSV'
picrust_id,shotgun_id
ERR1456815,<SHOTGUN_NAME>
ERR1456817,<SHOTGUN_NAME>
ERR1456792,<SHOTGUN_NAME>