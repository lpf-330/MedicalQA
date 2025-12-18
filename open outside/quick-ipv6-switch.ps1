# quick-ipv6-switch.ps1
# 用法：.\quick-ipv6-switch.ps1 -Port 8001
# 或直接运行：.\quick-ipv6-switch.ps1

param([int]$Port = 8001)

# 以管理员运行
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Start-Process PowerShell -Verb RunAs "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit
}

# 自动获取公网IPv6
$ipv6 = Get-NetIPAddress -AddressFamily IPv6 | 
        Where-Object { $_.PrefixOrigin -eq 'RouterAdvertisement' -and 
                       -not $_.IPAddress.StartsWith('fe80') -and
                       $_.IPAddress -match '^2[0-9a-f]{3}:|^240[89abcdef]' } |
        Select-Object -First 1 -ExpandProperty IPAddress

if (-not $ipv6) {
    # 备选方案：从WSL2获取
    $ipv6 = (wsl hostname -I).Split() | Where-Object { $_ -match '^2409:' } | Select-Object -First 1
}

if (-not $ipv6) {
    Write-Host "无法自动获取IPv6地址" -ForegroundColor Red
    $ipv6 = Read-Host "请手动输入IPv6地址"
}

# 获取WSL2 IP
$wslIp = (wsl hostname -I).Split()[0].Trim()

# 更新转发
netsh interface portproxy reset
netsh interface portproxy add v6tov4 listenport=$Port listenaddress=$ipv6 connectport=$Port connectaddress=$wslIp
netsh interface portproxy add v6tov4 listenport=$Port listenaddress=:: connectport=$Port connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIp

# 输出结果
$configUrl = "http://[$ipv6]:$Port/v1/medical_rag_stream"
Write-Host "? 配置完成!" -ForegroundColor Green
Write-Host "IPv6地址: $ipv6" -ForegroundColor Cyan
Write-Host "WSL2 IP: $wslIp" -ForegroundColor Cyan
Write-Host "端口: $Port" -ForegroundColor Cyan
Write-Host "`nSpringBoot配置:" -ForegroundColor Yellow
Write-Host $configUrl -ForegroundColor Magenta

Set-Clipboard -Value $configUrl
Write-Host "（地址已复制到剪贴板）" -ForegroundColor Gray

# 保持窗口打开
Write-Host "`n按任意键退出..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")