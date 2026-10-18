# Reuse the installed official Microsoft MCP and dedicated FYP profile.
# Native windows must not inherit a fixed automated-test viewport.
param([string]$ConfigPath)
$ErrorActionPreference='Stop'
$workspace=Split-Path $PSScriptRoot -Parent
$toolRoot=Join-Path $env:LOCALAPPDATA 'CrimeScopeAI'
$entry=Join-Path $toolRoot 'browser-tools\node_modules\@playwright\mcp\cli.js'
$config=if($ConfigPath){(Resolve-Path -LiteralPath $ConfigPath).Path}else{Join-Path $PSScriptRoot 'playwright-native.json'}
if(-not (Test-Path -LiteralPath $entry)){throw 'Existing official Playwright MCP installation not found.'}
if(Get-NetTCPConnection -LocalPort 8931 -State Listen -ErrorAction SilentlyContinue){
    throw 'FYP MCP port 8931 is already in use. Do not launch a duplicate browser service.'
}
$serverArgs=@('"'+$entry+'"','--config','"'+$config+'"','--host','127.0.0.1','--port','8931',
    '--browser','chrome','--user-data-dir','"'+(Join-Path $toolRoot 'browser-profile')+'"',
    '--output-dir','"'+(Join-Path $toolRoot 'browser-output')+'"',
    '--shared-browser-context','--no-webmcp','--codegen','none')
Start-Process -FilePath 'C:\Program Files\nodejs\node.exe' -ArgumentList $serverArgs -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $toolRoot 'mcp-server.stdout.log') -RedirectStandardError (Join-Path $toolRoot 'mcp-server.stderr.log')
