$ErrorActionPreference = 'Continue'

$outFile = 'F:\hello\status_after_600s.txt'

Start-Sleep -Seconds 600

$sb = New-Object System.Text.StringBuilder

$null = $sb.AppendLine('ckpt exists:')
$ckptPath = 'F:\hello\project\backend\ckpt_2.pt'
$exists = Test-Path $ckptPath
$null = $sb.AppendLine($exists.ToString())
if ($exists) {
    $fi = Get-Item $ckptPath
    $null = $sb.AppendLine(('Name: {0}' -f $fi.Name))
    $null = $sb.AppendLine(('MB: {0}' -f [math]::Round($fi.Length / 1MB, 1)))
    $null = $sb.AppendLine(('LastWriteTime: {0}' -f $fi.LastWriteTime))
}

$null = $sb.AppendLine('')
$null = $sb.AppendLine('log:')
$logPath = 'F:\hello\project\backend\train_single_2.log'
try {
    $null = $sb.AppendLine([IO.File]::ReadAllText($logPath))
} catch {
    $null = $sb.AppendLine(('ERROR reading log: {0}' -f $_.Exception.Message))
}

$null = $sb.AppendLine('')
$null = $sb.AppendLine('procs:')
try {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'train_single_mlp\.py 2' } |
        Select-Object ProcessId, CommandLine

    if ($null -eq $procs -or $procs.Count -eq 0) {
        $null = $sb.AppendLine('(none)')
    } else {
        $null = $sb.AppendLine(($procs | Format-Table -AutoSize | Out-String))
    }
} catch {
    $null = $sb.AppendLine(('ERROR querying procs: {0}' -f $_.Exception.Message))
}

$sb.ToString() | Out-File -FilePath $outFile -Encoding utf8
