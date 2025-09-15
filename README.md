# Smart Price Optimizer

Smart Price Optimizer is a demo Flask application with a **Machine Learning model** that predicts optimized prices based on competitor data.  
It provides a simple web interface to:

- 📥 **Download a sample CSV** to use as template  
- ⬆️ **Upload your own CSV of competitors** (`company`, `product_model`, `feature`, `price`)  
- 🧠 **Train a regression model** on the uploaded dataset  
- 🔮 **Predict a new price** by entering company, model and feature  

---

## 📂 Project Structure

```text
smart-price-optimizer/
├── .env.example              # Example env file
├── .gitignore
├── README.md
├── data/
│   ├── .gitkeep
│   ├── train.csv             # Default dataset
│   └── uploads/              # Uploaded CSVs
│       └── 20250914_123939__sample_competitors.csv
├── models/
│   ├── .gitkeep
│   └── model.pkl             # Trained model
├── out/
│   └── metrics.txt           # Training metrics/logs
├── pyproject.toml            # Dependencies & metadata
└── src/
    ├── __init__.py
    ├── price_app/
    │   ├── __init__.py
    │   ├── app.py            # Flask routes
    │   ├── model.py          # Training & persistence
    │   ├── static/
    │   │   └── style.css     # Custom styles
    │   └── templates/        # HTML templates
    │       ├── base.html     # Layout base
    │       ├── index.html    # Home
    │       ├── upload.html   # Upload page
    │       ├── predict.html  # Prediction form
    │       └── result.html   # Prediction result
    └── smart_price_optimizer.egg-info/
        ├── PKG-INFO
        ├── SOURCES.txt
        ├── dependency_links.txt
        ├── requires.txt
        └── top_level.txt
```

## 🚀 Getting Started
### 1. Clone the repo
```bash
git clone https://github.com/EngLucaPisani/smart-price-optimizer.git
cd smart-price-optimizer
```
---

### 2. Create & activate a virtual environment (Linux/WSL)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
---

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -e .

```
## 🧩 How it Works

### Step 1: Download sample CSV

Go to: [http://127.0.0.1:5000/download-sample](http://127.0.0.1:5000/download-sample)

The CSV looks like this:
```csv
company,product_model,feature,price
BrandA,Alpha,base,99.9
BrandA,Alpha,pro,129.0
BrandB,Beta,base,109.0
BrandB,Beta,max,159.0
BrandC,Gamma,pro,139.0

```

Replace with your own competitor data.



### Step 2: Upload CSV & Train

Go to: [http://127.0.0.1:5000/upload](http://127.0.0.1:5000/upload)

Upload your CSV → model is saved in `models/model.pkl`.  
Logs/metrics are saved in `out/metrics.txt`.


### Step 3: Predict

Go to: [http://127.0.0.1:5000/predict-form](http://127.0.0.1:5000/predict-form)

Fill the form:
- Company  
- Product model  
- Feature  

→ Get the predicted price.


## 🌐 Run the Flask app
```bash
flask --app price_app.app run --debug
```

Open http://127.0.0.1:5000


## 🛠️ Tech Stack

- Python 3.10+

- Flask 3

- scikit-learn

- Pandas & NumPy

- Joblib


## 🔮 Next Steps

- Improve UI/UX styling with base.html + CSS

- Add dropdowns auto-populated from CSV values

- Extend ML model with tree-based / time series methods

- Deploy on Render/Heroku with Gunicorn

## 📜 License

MIT License. Feel free to use and adapt.


---


```bash
head -20 README.md