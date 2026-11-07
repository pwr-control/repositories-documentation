param(
  [string]$Dir = ".",
  [switch]$Recurse
)

$failed = @()
$files = Get-ChildItem -Path $Dir -Filter *.tex -File -Recurse:$Recurse |
  Where-Object { Select-String -Path $_.FullName -Pattern '\\documentclass' -Quiet }

foreach ($f in $files) {
  Write-Host "=== $($f.Name) ===" -ForegroundColor Cyan
  Push-Location $f.DirectoryName
  # cambia -pdf con -lualatex / -xelatex se serve; aggiungi -shell-escape se lo usi in TeXstudio
  latexmk -pdf -interaction=nonstopmode -halt-on-error $f.Name | Out-Null
  if ($LASTEXITCODE -ne 0) { $failed += $f.FullName }
  Pop-Location
}

Write-Host "`nCompilati: $($files.Count - $failed.Count) / $($files.Count)"
if ($failed) { Write-Host "Falliti:" -ForegroundColor Red; $failed }