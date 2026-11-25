# Prints the full path of blender.exe, or nothing. Used by run_sim.bat.
$ErrorActionPreference = 'SilentlyContinue'
function Done($p) { if ($p -and (Test-Path $p)) { Write-Output (Resolve-Path $p).Path; exit 0 } }

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

# 4. Microsoft Store app
$pkg = Get-AppxPackage -Name '*Blender*' | Select-Object -First 1
if ($pkg) { Done (Join-Path $pkg.InstallLocation 'Blender\blender.exe'); Done (Join-Path $pkg.InstallLocation 'blender.exe') }

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
