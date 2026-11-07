# Full artefact rebuild after the F: volume was lost (drive letter remapped and
# the old build volume disappeared). Commands mirror Section 4.4 of the
# manuscript exactly, including --limit, because the frame caches are shared by
# every downstream run. Artefacts land under _paper_build_zh (scripts/_paths.py).

$ErrorActionPreference = 'Continue'
$root = 'E:\Projects\20260522-EchoCLIP'
Set-Location $root
$logs = Join-Path $root '_paper_build_zh\logs'
$cache = Join-Path $root '_paper_build_zh\cache'
$results = Join-Path $root '_paper_build_zh\results_zh'
New-Item -ItemType Directory -Force -Path $logs, $cache, $results | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runlog = Join-Path $logs "rebuild_$stamp.log"

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    Write-Host $line
    Add-Content -Path $runlog -Value $line -Encoding UTF8
}

function Step {
    param([string]$Name, [string[]]$PyArgs)
    $t0 = Get-Date
    $out = Join-Path $logs ("{0}_{1}.out" -f $Name, $stamp)
    $err = Join-Path $logs ("{0}_{1}.err" -f $Name, $stamp)
    Say "START $Name"
    $p = Start-Process -FilePath python -ArgumentList $PyArgs `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait -PassThru
    $dt = ((Get-Date) - $t0).TotalMinutes
    $code = [int]$p.ExitCode
    Say ("END   {0}  exit={1}  {2:N1} min" -f $Name, $code, $dt)
    if ($code -ne 0) {
        Add-Content -Path $runlog -Value "---- stdout tail ----" -Encoding UTF8
        Get-Content $out -Tail 20 -ErrorAction SilentlyContinue |
            ForEach-Object { Add-Content -Path $runlog -Value $_ -Encoding UTF8 }
        Add-Content -Path $runlog -Value "---- stderr tail ----" -Encoding UTF8
        Get-Content $err -Tail 20 -ErrorAction SilentlyContinue |
            ForEach-Object { Add-Content -Path $runlog -Value $_ -Encoding UTF8 }
    }
    return $code
}

Say "REBUILD START  root=$root"

# ---- 1. frame embeddings, exactly as Section 4.4 -------------------------
$extract = @(
    @{ n='c_test500';        f='test500_32_bgr.npz';   a=@('-u','scripts\extract_frames_embeddings.py','--split','test500','--limit','500','--frames','32','--batch-size','4') },
    @{ n='c_val';            f='val_32_bgr.npz';       a=@('-u','scripts\extract_frames_embeddings.py','--split','val','--limit','1288','--frames','32','--batch-size','4') },
    @{ n='c_train';          f='train_32_bgr.npz';     a=@('-u','scripts\extract_frames_embeddings.py','--split','train','--limit','2000','--frames','32','--batch-size','4') },
    @{ n='c_test500_stride'; f='test500_stride_bgr.npz'; a=@('-u','scripts\extract_frames_embeddings.py','--split','test500','--sample','official_stride','--batch-size','4') }
)
foreach ($s in $extract) {
    $fp = Join-Path $cache $s.f
    if (Test-Path $fp) { Say "SKIP  $($s.n) (cache present)"; continue }
    if ((Step -Name $s.n -PyArgs $s.a) -ne 0) { Say "ABORT at $($s.n)"; exit 1 }
}

# ---- 2. zero-shot rows, exactly as Section 4.4 ---------------------------
$zs = @(
    @{ n='R0';          a=@('-u','scripts\train_heads_from_cache.py','--experiment','R0','--seeds','42','--frames','32') },
    @{ n='R1';          a=@('-u','scripts\train_heads_from_cache.py','--experiment','R1','--seeds','42','--frames','32') },
    @{ n='R1U16';       a=@('-u','scripts\train_heads_from_cache.py','--experiment','R1U16','--seeds','42','--frames','16') },
    @{ n='ORACLE_EDES'; a=@('-u','scripts\train_heads_from_cache.py','--experiment','ORACLE_EDES','--seeds','42','--frames','32') }
)
foreach ($s in $zs) {
    if ((Step -Name $s.n -PyArgs $s.a) -ne 0) { Say "ABORT at $($s.n)"; exit 1 }
}

# ---- 3. supervised heads, three seeds each -------------------------------
foreach ($exp in @('R2','R3','R4','R5','R6')) {
    $a = @('-u','scripts\train_heads_from_cache.py','--experiment',$exp,
           '--seeds','42','43','44','--frames','32','--epochs','40','--pretrain-epochs','20')
    if ((Step -Name "${exp}_3seed" -PyArgs $a) -ne 0) { Say "ABORT at $exp"; exit 1 }
}

# ---- 4. ablations --------------------------------------------------------
$ab = Step -Name 'ablation' -PyArgs @('-u','scripts\run_ablation_matrix.py','--group','all')
if ($ab -ne 0) { Say "WARN ablation exit=$ab" }

# ---- 5. audit, figures, crosscheck --------------------------------------
Step -Name 'audit'      -PyArgs @('-u','scripts\audit_data.py')         | Out-Null
Step -Name 'figures'    -PyArgs @('-u','scripts\make_figures.py')       | Out-Null
Step -Name 'crosscheck' -PyArgs @('-u','scripts\crosscheck_numbers.py') | Out-Null
Step -Name 'verify'     -PyArgs @('-u','scripts\verify_rebuild_vs_paper.py') | Out-Null

Say "REBUILD END"
