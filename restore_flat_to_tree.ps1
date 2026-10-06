<#
    restore_flat_to_tree.ps1

    Reconstructs the original directory tree from this flat repository.

    Primary source of truth: FLAT_LAYOUT_MANIFEST.json, which maps every `flat_name` to its
    `original_path`. This is unambiguous even for filenames that begin with underscores
    (e.g. `echoclip/__init__.py` -> `echoclip____init__.py`).

    Fallback (if the manifest is unavailable): split the flat name on the `__` separator.

    Usage:
        pwsh -File restore_flat_to_tree.ps1 -Destination ..\EchoCLIP-TC-tree
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Destination
)

$ErrorActionPreference = 'Stop'

$source   = Split-Path -Parent $MyInvocation.MyCommand.Path
$manifest = Join-Path $source 'FLAT_LAYOUT_MANIFEST.json'
$selfFiles = @(
    'README.md', 'README_project_original.md', 'FLAT_LAYOUT.md',
    'FLAT_LAYOUT_MANIFEST.json', '_flat_map.json', 'restore_flat_to_tree.ps1'
)

if (Test-Path -LiteralPath $Destination) {
    try {
        cmd /c "rmdir /s /q `"$Destination`"" | Out-Null
    } catch {
        Write-Warning "could not fully clear $Destination (file in use?); continuing"
    }
}
New-Item -ItemType Directory -Path $Destination -Force | Out-Null

$count = 0

if (Test-Path -LiteralPath $manifest) {
    Write-Output "using $manifest"
    $index = Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($entry in $index.files) {
        $src = Join-Path $source $entry.flat_name
        if (-not (Test-Path -LiteralPath $src)) { Write-Warning "missing: $($entry.flat_name)"; continue }

        $rel    = ($entry.original_path -replace '/', '\')
        $target = Join-Path $Destination $rel
        $parent = Split-Path $target -Parent
        if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }

        Copy-Item -LiteralPath $src -Destination $target -Force
        $count++
    }
} else {
    Write-Warning 'manifest not found; falling back to __ splitting'
    Get-ChildItem -LiteralPath $source -File -Force |
        Where-Object { $selfFiles -notcontains $_.Name } |
        ForEach-Object {
            $name = $_.Name
            if ($name -eq 'README_project_original.md') { $name = 'README.md' }

            if ($name -like '*__*') {
                $parts = $name -split '__'
                $file  = $parts[-1]
                $dirs  = @($parts[0..($parts.Length - 2)] | Where-Object { $_ -ne '' })
                $rel   = if ($dirs.Count -gt 0) { Join-Path ($dirs -join '\') $file } else { $file }
            } else {
                $rel = $name
            }

            $target = Join-Path $Destination $rel
            $parent = Split-Path $target -Parent
            if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }

            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
            $count++
        }
}

Write-Output "restored $count files to $Destination"
Write-Output "next:  cd `"$Destination`"; python -m pytest tests -q"
