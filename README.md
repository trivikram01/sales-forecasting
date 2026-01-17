# Product Sales Forecasting: End-to-End Data Science Project

## 📌 Problem Statement
In the competitive retail industry, accurate sales forecasting is crucial for inventory management, financial planning, and strategic decision-making. Overstocking leads to waste, while understocking leads to lost revenue. This project aims to build a machine learning model to predict daily store sales based on historical data, store type, location, and seasonal factors like holidays and discounts.

## 🎯 Project Goals
1.  **Analyze** historical sales data to uncover patterns (Seasonality, Holiday impacts).
2.  **Validate** assumptions using statistical Hypothesis Testing.
3.  **Build** a predictive Machine Learning model (Random Forest).
4.  **Deploy** the solution as an interactive web application for end-users.

## 📊 Block 1 & 2: EDA and Hypothesis Testing
**Key Insights:**
* **Discounts:** Statistically significant impact (P-Value < 0.05). Stores with discounts sell considerably more.
* **Holidays:** Sales spike during holidays, confirmed by T-Tests.
* **Store Types:** Different store types (S1 vs S4) show distinct performance tiers.
* **Correlation:** A 94% strong positive correlation exists between `Order_Count` and `Sales`.

## 🤖 Block 3: Machine Learning Modeling
* **Model Used:** Random Forest Regressor (n_estimators=100)
* **Feature Engineering:** extracted Day/Month/Year from Date; Label Encoded categorical variables.
* **Performance Metrics:**
    * **RMSE:** ~9851 (Root Mean Squared Error)
    * **R2 Score:** 0.71 (The model explains ~71% of the variance in sales data).

## 🚀 Block 4: Deployment
The model was deployed using a **Flask API** with a simple HTML frontend.
* **Backend:** Python (Flask) loads the `.pkl` model and processes user inputs.
* **Frontend:** HTML/CSS form for Store Managers to input date and store details.

### How to Run Locally
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the app: `python app.py`
4.  Open `http://127.0.0.1:5000` in your browser.

## 📂 Repository Structure
* `app.py`: Main Flask application.
* `sales_model.pkl`: Pre-trained Random Forest model.
* `encoders.pkl`: Saved LabelEncoders for preprocessing.
* `templates/`: Contains `index.html`.
* `Product_Sales_Analysis.ipynb`: Jupyter Notebook with full code.