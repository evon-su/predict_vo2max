### 測試方式
(不須安裝第三方套件)
```commandline
git clone https://github.com/evon-su/predict_vo2max.git
cd predict_vo2max
cd submit
python predict.py
```
### 說明
1. api 接口為 submit/predict  `predictVO2max`
   - Arguments：
     - user_info: ex. {'gender': 0, 'hr_rest': 63} 
     - speed_ls: speed list
     - hr_ls: heart rate list
     - model: 目前有 modeling.model1() & modeling.model2() 兩種模型，可於括號內輸入模型參數。

2. 測試範例見predict下方 ```if __name__ == '__main__'```。其測試資料在 submit/sample_data
3. api 呼叫之模型運算在 submit/modeling
3. 模型訓練流程紀錄在 submit/pipeline (需安裝第三方套件)
4. model1 為簡單線性模型(單一`ratio`參數)，model2 為多元線性模型(`ratio`參數+`性別`&`靜止心率`)

