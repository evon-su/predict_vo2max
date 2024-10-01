### 測試方式
(不須下載第三套件)
```commandline
git clone https://github.com/evon-su/predict_vo2max.git
python -m venv venv
cd submit
python predict.py
```
### 說明
1. api 接口為 predict.py / `predictVO2max`
   - Arguments：
     - user_info: ex. {'gender': 0, 'hr_rest': 63} 
     - speed_ls: speed list
     - hr_ls: heart rate list
     - model: 目前有 modeling.model1 & modeling.model2 兩種模型

2. 測試資料在 sample_data.py
3. api 呼叫之模型運算在 modeling.py
3. 模型訓練流程在 pipeline.py

