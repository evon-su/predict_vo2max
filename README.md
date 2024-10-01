### 使用方式
(不須下載第三套件)
```commandline
git clone https://github.com/evon-su/predict_vo2max.git
python -m venv venv
cd submit
python predict.py
```
### 說明
1. api 接口為 predict.py / `predictVO2max`
2. api 呼叫之模型運算在 modeling.py
3. 模型訓練在 pipeline.py
