@echo off
echo ========================================
echo   整晚自動訓練啟動腳本
echo ========================================
echo.
echo 這將執行 5 個實驗，預計 11-12 小時
echo.
echo 開始時間: %date% %time%
echo.
pause

python run_all_experiments.py

echo.
echo ========================================
echo 所有實驗完成！
echo ========================================
echo.
echo 完成時間: %date% %time%
echo.
echo 下一步: 執行 ensemble.py 合併預測
echo.
pause
