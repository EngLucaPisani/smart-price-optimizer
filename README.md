# Smart Price Optimizer

Smart Price Optimizer is a demo Flask application with a **Machine Learning model** to predict optimized prices (or any continuous target) based on input features.  
It is structured with a clean `src/` layout, making it easy to extend, deploy, and integrate with other systems.

---

## 📂 Project Structure

```text
smart-price-optimizer/
├── data/                     # Training datasets (optional)
├── models/                   # Saved ML models (joblib)
├── src/
│   └── price_app/
│       ├── app.py            # Flask app entrypoint
│       ├── model.py          # Model training and persistence
│       └── templates/        # HTML templates
│           ├── index.html
│           └── result.html
├── .venv/                    # Virtual environment (local)
├── pyproject.toml             # Project dependencies and metadata
├── .gitignore
└── README.md
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
## 🧠 Train the model

Run the demo training script:
```bash
python -m price_app.model
```

```text
This generates models/model.pkl.
```
## 🌐 Run the Flask app

 Option A:
```bash
flask --app price_app.app run --debug
```

Option B:

```bash
python -m price_app.app
```

```text
Visit: http://127.0.0.1:5000
```
## 🛠️ Tech Stack

- Python 3.10+

- Flask 3

- Jinja2

- scikit-learn

- Pandas & NumPy

- Joblib

## 🔮 Next Steps

- Add a real pricing dataset

- Extend model features (time series, categorical variables)

- Deploy with Gunicorn on Render/Heroku

- Add REST API endpoints for external integrations

## 📜 License

MIT License. Feel free to use and adapt.