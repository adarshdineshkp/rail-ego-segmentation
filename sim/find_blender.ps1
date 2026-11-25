# Prints the full path of blender.exe, or nothing. Used by run_sim.bat.
$ErrorActionPreference = 'SilentlyContinue'
# Store installs live in a protected WindowsApps package folder whose exe cannot be started
# directly ("Access is denied"); only their execution alias in %LOCALAPPDATA% works.
function Done($p) {
  if ($p -and (Test-Path $p) -and ($p -notmatch 'WindowsApps\\BlenderFoundation\.Blender_')) {
    Write-Output (Resolve-Path $p).Path; exit 0
  }
}

# 0. Store execution alias, if enabled
foreach ($a in (Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WindowsApps" -Filter 'blender*.exe')) { Done $a.FullName }

# 1. on PATH
Done (Get-Command blender.exe).Source

# 2. the program registered to open .blend files (installer, Store and Steam installs set this)
$progId = (Get-ItemProperty 'Registry::HKEY_CLASSES_ROOT\.blend').'(default)'
foreach ($id in @($progId, 'blendfile')) {
  if ($id) {
    $cmd = (Get-ItemProperty "Registry::HKEY_CLASSES_ROOT\$id\shell\open\command").'(default)'
    if ($cmd -match '"([^"]*blender[^"]*\.exe)"') { Done $Matches[1] }
    if ($cmd -match '^(\S*blender\S*\.exe)') { Done $Matches[1] }
  }
}

# 3. uninstall entries
$keys = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
foreach ($e in (Get-ItemProperty $keys | Where-Object { $_.DisplayName -like '*Blender*' })) {
  if ($e.InstallLocation) { Done (Join-Path $e.InstallLocation 'blender.exe') }
  if ($e.DisplayIcon -match '([^",]*blender\.exe)') { Done $Matches[1] }
}

# 4. (Microsoft Store package folders are skipped: not launchable from scripts)

# 5. search the usual places and every drive's top levels
$roots = @("$env:ProgramFiles", "${env:ProgramFiles(x86)}", "$env:LOCALAPPDATA\Programs", "$env:USERPROFILE\Downloads",
           "$env:USERPROFILE\Desktop", "$env:USERPROFILE\Documents")
$roots += (Get-PSDrive -PSProvider FileSystem | ForEach-Object { $_.Root })
foreach ($r in $roots) {
  if (Test-Path $r) {
    $hit = Get-ChildItem -Path $r -Filter blender.exe -Recurse -Depth 4 -File | Select-Object -First 1
    if ($hit) { Done $hit.FullName }
  }
}
