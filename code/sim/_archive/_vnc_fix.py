import paramiko, time

host = "59.66.22.145"
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, port=22, username="KobayashiChiba", password="170516", timeout=10, look_for_keys=False, allow_agent=False)

def run(cmd, t=10):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=t)
    return stdout.read().decode("gbk", errors="replace")

# 停
run("sc stop uvnc_service"); time.sleep(2)
run("taskkill /f /im winvnc.exe 2>nul"); time.sleep(1)

# 用 echo 直接写文件
run('mkdir C:\\ProgramData\\UltraVNC 2>nul')
run('echo [admin]> C:\\ProgramData\\UltraVNC\\ultravnc.ini')
run('echo passwd=69fd3cce30cfcd4b>> C:\\ProgramData\\UltraVNC\\ultravnc.ini')
run('echo AuthRequired=1>> C:\\ProgramData\\UltraVNC\\ultravnc.ini')

print(">>> 配置内容:")
print(run("type C:\\ProgramData\\UltraVNC\\ultravnc.ini"))

# 清理 portable
run("del C:\\Users\\KobayashiChiba\\Downloads\\ultravnc.portable 2>nul")

# 启动
print("\n>>> 启动服务...")
print(run("sc start uvnc_service"))
time.sleep(5)

print("\n=== 端口 ===")
print(run("netstat -ano | findstr 5900"))

print("\n=== 服务 ===")
print(run("sc query uvnc_service"))

client.close()
