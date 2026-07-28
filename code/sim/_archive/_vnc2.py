import paramiko, time
import io

host = "59.66.22.145"
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=22, username="KobayashiChiba", password="170516", timeout=10, look_for_keys=False, allow_agent=False)

def run(cmd, t=15):
    i, o, e = client.exec_command(cmd, timeout=t)
    return o.read().decode("gbk", errors="replace"), e.read().decode("gbk", errors="replace")

# Step 1: Stop everything
run("sc stop uvnc_service"); time.sleep(2)
run("taskkill /f /im winvnc.exe"); time.sleep(1)

# Step 2: Write config using PowerShell here-strings (avoids path issues)
ps1 = '''
$dir = [Environment]::GetFolderPath('CommonApplicationData') + '\\UltraVNC'
New-Item -Path $dir -ItemType Directory -Force | Out-Null
@'
[admin]
passwd=69fd3cce30cfcd4b
AuthRequired=1
'@ | Set-Content -Path (Join-Path $dir 'ultravnc.ini')
Get-Content (Join-Path $dir 'ultravnc.ini')
'''
cmd1 = 'powershell -Command ' + "'" + ps1.replace("'", "''") + "'"
o, e = run(cmd1)
print("Config write:", o[:200])

# Step 3: Start service
run("sc start uvnc_service"); time.sleep(5)

# Step 4: Check
o, e = run("netstat -ano | findstr 5900")
print("Port 5900:", o)
o, e = run("tasklist | findstr winvnc")
print("WinVNC:", o)

client.close()
