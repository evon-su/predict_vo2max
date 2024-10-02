### 測試方式
(不須安裝第三方套件)
```commandline
git clone https://github.com/evon-su/predict_vo2max.git
cd submit
python predict.py
```
### 說明
1. api 接口為 predict.py / `predictVO2max`
   - Arguments：
     - user_info: ex. {'gender': 0, 'hr_rest': 63} 
     - speed_ls: speed list
     - hr_ls: heart rate list
     - model: 目前有 modeling.model1() & modeling.model2() 兩種模型，可於括號內加入模型參數。

2. 測試資料在 sample_data.py
3. api 呼叫之模型運算在 modeling.py
3. 模型訓練流程紀錄在 pipeline.py (需安裝第三方套件)
4. model1 為簡單線性回歸(單一`ratio`參數)，model2 為多元線性模型(`ratio`參數+`性別`&`靜止心率`)

