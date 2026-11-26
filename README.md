# TFM sobre recomendación
## Librerias a instalar
- Por tema de compatibilidad, se va a usar python 3.11 en principio, ya que para Spark 3.5.x y la 4.x sirve y para Lenskit tambien
```bash
python -m pip install --upgrade pip
py -3.11 -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pip install matplotlib seaborn plotly
pip freeze > requirements.txt
python prepare_data.py ml-100k
```