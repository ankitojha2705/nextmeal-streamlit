import streamlit as st
import pandas as pd
import joblib
from surprise import SVD, Dataset, Reader

# =================== Load Files =================== #
@st.cache_resource
def load_files():
    svd_model = joblib.load('model/svd_model.pkl')  # Ensure correct path
    combined_df = pd.read_json('data/combined_df.json')
    return svd_model, combined_df

svd_model, combined_df = load_files()

# =================== Preprocess Data =================== #
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(combined_df[['user_id', 'business_id', 'review_rating']], reader)

# =================== Recommendation Function =================== #
def svd_recommendations(user_id, svd_model, combined_df, top_n=5):
    """
    Generate Top-N recommendations for a specific user.
    """
    # Filter items the user has already rated
    user_rated_items = combined_df[combined_df['user_id'] == user_id]['business_id'].tolist()
    all_items = combined_df['business_id'].unique()

    # Predict ratings for unrated items
    predictions = []
    for item in all_items:
        if item not in user_rated_items:
            pred = svd_model.predict(user_id, item)
            predictions.append((item, pred.est))

    # Sort predictions by estimated rating
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_n]

    # Retrieve restaurant details and predicted scores
    recommended_items = pd.DataFrame(predictions, columns=['business_id', 'predicted_rating'])
    final_recommendations = pd.merge(
        recommended_items, 
        combined_df[['business_id', 'name']].drop_duplicates(), 
        on='business_id', 
        how='left'
    )
    return final_recommendations

# =================== Streamlit UI =================== #
st.title("🍽️ SVD-based Restaurant Recommendation System")

# Input User ID and Number of Recommendations
user_id = st.text_input("Enter User ID:")
top_n = st.slider("Number of Recommendations:", 1, 10, 5)

if st.button("Get Recommendations"):
    if user_id:
        # Check if the user exists in the dataset
        if user_id in combined_df['user_id'].unique():
            recommendations = svd_recommendations(user_id, svd_model, combined_df, top_n)
            if not recommendations.empty:
                st.success("Top Recommended Restaurants for User:")
                st.dataframe(recommendations[['name', 'business_id', 'predicted_rating']])
            else:
                st.warning("No recommendations found for this user!")
        else:
            st.error("Invalid User ID. Please enter a valid User ID!")
    else:
        st.warning("Please enter a User ID to get recommendations.")

# Debugging Section (Optional)
if st.checkbox("Show Data (Debugging)"):
    st.write("Combined Data Preview:")
    st.dataframe(combined_df.head())
