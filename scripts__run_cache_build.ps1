# Rebuild the frame-embedding caches on the internal disk.
# The external F: volume dropped off the USB bus on 2026-11-07 14:41 and NTFS
# repaired it at 15:23, which emptied the previous build tree. Everything now
# lives under the repository via scripts/_paths.py.
Set-Location "E:\Projects\20260522-EchoCLIP"

$base = "E:\Projects\20260522-EchoCLIP\_paper_build_zh"
$log  = "$base\logs\rebuild_cache.log"

function Say($m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ss"), $m
    $line | Tee-Object -FilePath $log -Append
}
function Run($step, $cmdline, $logfile) {
    Say $step
    cmd /c "$cmdline > `"$logfile`" 2>&1"
    Say ("exit=" + $LASTEXITCODE + "  " + $step)
}

Say "cache rebuild start (internal disk)"

Run "step 1/4 test500 uniform 32 frames (resume if present)" `
    "python scripts\extract_frames_embeddings.py --split test500 --limit 500 --frames 32 --batch-size 8" `
    "$base\logs\extract_test500.log"

Run "step 2/4 test500 official stride (resume from 210-video checkpoint)" `
    "python scripts\extract_frames_embeddings.py --split test500 --sample official_stride --batch-size 8" `
    "$base\logs\extract_test500_stride.log"

Run "step 3/4 val uniform 32 frames" `
    "python scripts\extract_frames_embeddings.py --split val --limit 1288 --frames 32 --batch-size 8" `
    "$base\logs\extract_val.log"

Run "step 4/4 train uniform 32 frames" `
    "python scripts\extract_frames_embeddings.py --split train --limit 2000 --frames 32 --batch-size 8" `
    "$base\logs\extract_train.log"

Say "cache rebuild done"
Get-ChildItem "$base\cache\*.npz" |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize |
    Tee-Object -FilePath $log -Append
