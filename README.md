# TFM sobre recomendación
## Librerias a instalar
```bash
python -m venv .venv && source .venv/bin/activate
pip install "pyspark==<versión del clúster>"
pip install lenskit
pip install xgboost
pip install pandas numpy scipy
pip install pyarrow 
pip install matplotlib seaborn plotly
pip freeze > requirements.txt
```