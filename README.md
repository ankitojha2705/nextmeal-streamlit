# 🍽️ SVD-based Restaurant Recommendation System

This is a **Streamlit web application** that provides restaurant recommendations using the **SVD (Singular Value Decomposition)** collaborative filtering algorithm. The app predicts ratings for unrated restaurants and recommends the top restaurants based on user input.

Youtube Link - https://www.youtube.com/watch?v=CTSQ4x0QPfY

---

## 🚀 Features
- **User-Specific Recommendations**: Input a User ID and get personalized restaurant recommendations.
- **SVD Model**: Utilizes a pre-trained SVD model for collaborative filtering.
- **Dynamic Controls**: Adjust the number of recommendations with an interactive slider.
- **Interactive UI**: Built with Streamlit for a clean and easy-to-use interface.
- **Debugging Option**: View combined dataset preview for debugging purposes.

---

## 📋 Prerequisites
Ensure you have the following installed:

- Python 3.8 or higher
- Streamlit
- Pandas
- Scikit-surprise
- Joblib

---

## 🔧 Installation
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Sample `requirements.txt`:
   ```
   streamlit
   pandas
   scikit-surprise
   joblib
   ```

---

## ▶️ Run the Application
To launch the Streamlit app, run:
```bash
streamlit run app.py
```

The app will open in your default browser at:
```
http://localhost:8501/
```

---

## 📝 Usage
1. **Input User ID**: Enter a valid User ID present in the dataset.
2. **Select Number of Recommendations**: Use the slider to choose between 1 and 10 recommendations.
3. **Get Recommendations**: Click the **"Get Recommendations"** button to view the results.
4. **Optional Debugging**: Check the box to preview the combined dataset.

---

## 📊 Data Overview
The application uses a dataset (`combined_df.json`) containing the following columns:
- **user_id**: ID of the user who rated a restaurant.
- **business_id**: ID of the restaurant.
- **review_rating**: Rating given by the user (scale: 1-5).
- **name**: Name of the restaurant.

---

## ⚙️ Model Details
- **Algorithm**: Singular Value Decomposition (SVD) from the `surprise` library.
- **Training**: The model is pre-trained on user-item rating data.
- **Prediction**: The app predicts ratings for unrated restaurants for a specific user and recommends the top-N restaurants based on predicted scores.

---

