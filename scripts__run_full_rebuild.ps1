# Full experiment matrix, rebuilt after the F: build tree was lost.
# Runs only after every cache exists. Sequential: 4 GB VRAM, no parallelism.
Set-Location "E:\Projects\20260522-EchoCLIP"

$base = "F:\EchoCLIP_paper_build"
$log  = "$base\logs\rebuild_experiments.log"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $m
    $line | Tee-Object -FilePath $log -Append
}
function Need($file) {
    if (-not (Test-Path $file)) { throw "missing cache: $file" }
    Say "cache ready: $file"
}
function Run($label, $cmdline, $logfile) {
    Say $label
    cmd /c "$cmdline > `"$logfile`" 2>&1"
    Say ("exit=" + $LASTEXITCODE + "  " + $label)
}

Need "$base\cache\test500_32_bgr.npz"
Need "$base\cache\test500_stride_bgr.npz"
Need "$base\cache\val_32_bgr.npz"
Need "$base\cache\train_32_bgr.npz"
Need "$base\cache\text_ef.npz"

$T = "python scripts\train_heads_from_cache.py"

Say "=== zero-shot rows ==="
Run "R0" "$T --experiment R0 --seeds 42 --frames 32" "$base\logs\run_R0.log"
Run "R1" "$T --experiment R1 --seeds 42 --frames 32" "$base\logs\run_R1.log"
Run "R1U16" "$T --experiment R1U16 --seeds 42 --frames 16" "$base\logs\run_R1U16.log"
Run "ORACLE_EDES" "$T --experiment ORACLE_EDES --seeds 42 --frames 32" "$base\logs\run_ORACLE_EDES.log"

# top-k ablation, zero-shot only
Run "R1 top 10" "$T --experiment R1 --seeds 42 --frames 32 --top-frac 0.1 --tag abtop10" "$base\logs\run_abtop10.log"
Run "R1 top 30" "$T --experiment R1 --seeds 42 --frames 32 --top-frac 0.3 --tag abtop30" "$base\logs\run_abtop30.log"

Say "=== supervised heads, seeds 42 43 44 ==="
foreach ($exp in @("R2", "R3", "R4", "R5", "R6")) {
    Run "$exp seeds 42 43 44" "$T --experiment $exp --seeds 42 43 44 --frames 32 --epochs 40 --pretrain-epochs 20" "$base\logs\run_$exp.log"
}

Say "=== frame-count ablation on R4 ==="
foreach ($nf in @(8, 16)) {
    foreach ($s in @(42, 43, 44)) {
        Run "R4 frames $nf seed $s" "$T --experiment R4 --seeds $s --frames $nf --cache-tag 32_bgr --tag abframes$nf" "$base\logs\run_abframes${nf}_s$s.log"
    }
}

Say "=== pooling ablation ==="
foreach ($s in @(42, 43, 44)) {
    Run "R4_MEAN seed $s" "$T --experiment R4_MEAN --seeds $s --frames 32 --tag abpoolmean" "$base\logs\run_abpoolmean_s$s.log"
}

Say "=== training-size ablation ==="
foreach ($n in @(500, 1000)) {
    foreach ($s in @(42, 43, 44)) {
        Run "R4 scale $n seed $s" "$T --experiment R4 --seeds $s --frames 32 --train-limit $n --tag abscale$n" "$base\logs\run_abscale${n}_s$s.log"
    }
}

Say "=== contrastive pretraining with annotated ED/ES frames ==="
foreach ($s in @(42, 43, 44)) {
    Run "R5_EDES seed $s" "$T --experiment R5_EDES --seeds $s --frames 32 --tag edes" "$base\logs\run_edes_s$s.log"
}

Say "=== statistics ==="
Run "paired bootstrap" "python scripts\run_ablation_matrix.py --group stat" "$base\logs\run_stat.log"
Run "data audit" "python scripts\audit_data.py" "$base\logs\audit_data.log"

Say "=== figures and documents ==="
Run "figures" "python scripts\make_figures.py" "$base\logs\make_figures.log"
Run "manuscript" "python scripts\build_manuscript.py" "$base\logs\build_manuscript.log"
Run "review doc" "python scripts\build_review_doc.py" "$base\logs\build_review_doc.log"
Run "docx validation" "python scripts\validate_docx.py" "$base\logs\validate_docx.log"
Run "number crosscheck" "python scripts\crosscheck_numbers.py" "$base\logs\crosscheck.log"

Say "=== all done ==="
Get-ChildItem "$base\results\zh\*.json" | Measure-Object | ForEach-Object { Say ("result files: " + $_.Count) }
