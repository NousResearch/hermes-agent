@echo off
echo Hermes Agent 更新工具
echo ======================
cd /d "D:\hermes-agent"
python scripts\update_hermes.py %*
pause
