import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

# --- Dummy Classes for local development without the full training script ---
# These are included to allow the app to run on a local machine without needing the original training script.
# They are simplified versions of the classes from your full pipeline.
try:
    import xgboost as xgb
except ImportError:
    st.error("XGBoost library not found. Please install it with `pip install xgboost`.")
    xgb = None

# A simplified DietRecommender class for the app's output.
class DietRecommender:
    def __init__(self):
        self.dietplans = {
            'DIABETES_OBESITY': """
--- Personalized Diet Plan for Diabetes & Obesity ---

**Goal:** Control both blood sugar and caloric intake for healthy weight loss.
Your estimated daily calorie goal for a healthy weight loss is **{calories}** calories.

* **Low Glycemic Index Carbs:** Prioritize complex carbs that don't spike blood sugar, such as whole grains, legumes, and most vegetables.
* **Lean Protein & Healthy Fats:** These promote fullness and support weight management without negatively impacting blood sugar.
* **No Sugary Drinks:** Eliminate all sugary beverages and refined snacks, as they contribute to both weight gain and blood sugar fluctuations.
""",
            'DIABETES': """
--- Your Personalized Diabetes Management Plan ---

**Goal:** Our primary goal is to help you stabilize your blood sugar levels and feel more energized throughout the day. This plan focuses on balancing your meals to prevent sharp glucose spikes and crashes.

* **Carbohydrates:** Think "complex, not simple." Focus on high-fiber carbs like whole grains (quinoa, brown rice, oats), legumes, and vegetables. These are digested slowly, providing steady energy. Let's minimize refined grains (white bread, pasta) and all sugary drinks.
* **Protein:** Let's make protein a star in every meal. Lean protein from sources like chicken, fish, tofu, and beans will help you feel full and manage your blood sugar effectively.
* **Healthy Fats:** Good fats are your friends! Incorporate healthy fats from avocados, nuts, seeds, and olive oil to support heart health and satiety.
* **Meal Timing:** Let's aim for consistent meal times. Regular meals help regulate insulin and prevent extreme blood sugar fluctuations.
""",
            'HYPERTENSION': """
--- Your Personalized Hypertension Management Plan ---

**Goal:** Our main focus here is to lower your blood pressure. The key to this is managing your sodium intake and boosting foods that naturally support healthy blood flow.

* **Sodium Control:** Let's become "salt detectives." The goal is to reduce processed foods, fast food, and canned goods, which are often hidden sources of sodium. Instead of salt, let's explore a world of flavor with herbs, spices, lemon juice, and vinegar.
* **Potassium Power:** Potassium is a crucial mineral for balancing sodium levels. We'll increase your intake of potassium-rich foods like bananas, sweet potatoes, spinach, and tomatoes.
* **Fiber:** Fiber is your ally! Foods rich in fiber, such as whole grains and vegetables, help support a healthy circulatory system.
* **Hydration:** Staying well-hydrated is essential. Let's focus on drinking plenty of water and unsweetened beverages.
""",
            'OBESITY': """
--- Your Personalized Weight Management Plan ---

**Goal:** Our plan is to create a healthy caloric deficit that leads to sustainable weight loss. This isn't about restriction; it's about nourishing your body with nutrient-dense foods to feel full and energized.
Your estimated daily calorie goal for a healthy weight loss is **{calories}** calories.

* **Calorie-Density:** Let's choose foods that are big in volume but low in calories. Think leafy greens, high-fiber vegetables, and lean proteins. These foods fill you up without adding unnecessary calories.
* **Protein:** We'll ensure every meal contains a source of lean protein. Protein helps curb cravings and preserves muscle mass as you lose weight.
* **Fiber:** Fiber is key to feeling full for longer. We'll add plenty of high-fiber foods to your meals to aid digestion and manage hunger.
* **Sugars:** Let's replace sugary drinks, sweets, and refined snacks with healthier options like whole fruits, which provide natural sweetness and fiber.
""",
            'NORMAL': """
--- General Healthy Diet Plan ---

**Goal:** This plan is your guide to maintaining good health and preventing future disease. It's about balance, variety, and mindful eating.

* **Balanced Meals:** Eat a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats.
* **Portion Control:** Eat mindfully and listen to your body's hunger and fullness cues.
* **Hydration:** Drink plenty of water.
"""
        }
        self.meal_plans = {
            'DIABETES_OBESITY': {'NORMAL': {'Breakfast': 'Vegetable omelet with low-fat cheese.', 'Lunch': 'Chicken and vegetable skewers with a small portion of brown rice.', 'Dinner': 'Grilled fish with a large serving of steamed broccoli and cauliflower.'}},
            'DIABETES': {'NORMAL': {'Breakfast': 'Scrambled eggs with spinach and whole-wheat toast.', 'Lunch': 'Grilled chicken salad with a light vinaigrette.', 'Dinner': 'Baked salmon with steamed broccoli and quinoa.'}},
            'HYPERTENSION': {'NORMAL': {'Breakfast': 'Unsalted oatmeal with sliced banana.', 'Lunch': 'Large green salad with grilled fish and no-salt added dressing.', 'Dinner': 'Baked chicken with steamed vegetables seasoned with herbs.'}},
            'OBESITY': {'NORMAL': {'Breakfast': 'Greek yogurt with a handful of berries.', 'Lunch': 'A large bowl of mixed greens with grilled lean protein.', 'Dinner': 'Baked cod with a side of steamed asparagus and a sprinkle of lemon juice.'}},
            'NORMAL': {'NORMAL': {'Breakfast': 'Avocado toast on whole-wheat bread with a boiled egg.', 'Lunch': 'Turkey and veggie wrap on a whole-wheat tortilla.', 'Dinner': 'Lean steak with roasted potatoes and a side salad.'}},
        }
    def calculate_user_tdee(self, user_data):
        activity_multipliers = {'sedentary': 1.2, 'lightly active': 1.375, 'moderately active': 1.55, 'very active': 1.725}
        activity_level = user_data.get('Activity_Level', 'sedentary').lower()
        if activity_level not in activity_multipliers: activity_level = 'sedentary'
        age, height_cm, weight_kg, gender = user_data.get('Age', 30), user_data.get('Height_cm', 170), user_data.get('Weight_kg', 70), user_data.get('Gender', 'male').lower()
        if gender == 'female': bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        else: bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        return bmr * activity_multipliers[activity_level]
    def get_diet_recommendation(self, diseases, user_data):
        main_condition = 'NORMAL'
        has_diabetes, has_obesity, has_hypertension = 'Diabetes' in diseases, 'Obesity' in diseases, 'Hypertension' in diseases
        if has_diabetes and has_obesity: main_condition = 'DIABETES_OBESITY'
        elif has_diabetes and has_hypertension: main_condition = 'HYPERTENSION_DIABETES'
        elif has_diabetes: main_condition = 'DIABETES'
        elif has_obesity: main_condition = 'OBESITY'
        elif has_hypertension: main_condition = 'HYPERTENSION'
        
        plan = self.dietplans[main_condition]
        if main_condition in ['OBESITY', 'DIABETES_OBESITY']:
            tdee = self.calculate_user_tdee(user_data)
            calorie_goal = tdee - 500
            plan = plan.format(calories=int(calorie_goal))
        meal_plan = self.meal_plans.get(main_condition, {}).get('NORMAL', self.meal_plans['NORMAL']['NORMAL'])
        final_output = plan
        final_output += f"\n\n--- Sample Meal Plan ---\n**Breakfast:** {meal_plan['Breakfast']}\n**Lunch:** {meal_plan['Lunch']}\n**Dinner:** {meal_plan['Dinner']}"
        final_output += "\n\n--- Personal Dietary Notes ---\n"
        cuisine = user_data.get('Preferred_Cuisine', '').strip()
        final_output += f"**Preferred Cuisine:** The provided plan can be adapted to a **{cuisine}**-style diet.\n" if cuisine else "**Preferred Cuisine:** No specific preference noted.\n"
        restrictions = user_data.get('Dietary_Restrictions', '').strip()
        final_output += f"**Dietary Restrictions:** The plan has been designed to honor your restriction(s) of **{restrictions}**.\n" if restrictions else "**Dietary Restrictions:** None noted.\n"
        allergies = user_data.get('Allergies', '').strip()
        final_output += f"**Allergies:** Please ensure all meals and ingredients are free of **{allergies}**.\n" if allergies else "**Allergies:** None noted.\n"
        final_output += "\n**⚠️ CRITICAL SAFETY WARNING:** Always read food labels and consult a healthcare professional. This plan is not a substitute for professional medical advice.\n"
        return final_output

# --- 1. Load the Trained Model and Preprocessing Objects ---
MODEL_PATH = ' C:\Users\lenovo\OneDrive\Desktop\internship1\models\trained_model.pkl'

# Use st.cache_data to load the model only once.
@st.cache_data
def load_model_and_objects():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}! Please ensure 'trained_model.pkl' is in the specified path.")
        return None, None, None, None
    
    with open(MODEL_PATH, 'rb') as f:
        model_info = pickle.load(f)
    return model_info['model'], model_info['scaler'], model_info['mlb'], model_info['feature_names']

model, scaler, mlb, feature_names = load_model_and_objects()

# Initialize DietRecommender class
recommender = DietRecommender()

# --- 2. Build the Streamlit App User Interface ---
st.title("👨‍⚕️ Personalized Health & Diet Recommender")
st.markdown("Enter your health metrics to receive a comprehensive health report and a personalized diet plan.")


# --- Input Sections ---
st.header("1. Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age (years)", 18, 100, 45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height_cm = st.slider("Height (cm)", 100, 250, 175)
with col2:
    weight_kg = st.slider("Weight (kg)", 30, 200, 80)
    activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
    daily_caloric_intake = st.number_input("Daily Caloric Intake (kcal)", 500, 5000, 2500)

st.header("2. Health Metrics")
col3, col4, col5 = st.columns(3)
with col3:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
with col4:
    blood_pressure = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
with col5:
    glucose = st.number_input("Glucose (mg/dL)", 50, 300, 95)
    
st.header("3. Dietary Preferences & Restrictions")
col6, col7 = st.columns(2)
with col6:
    dietary_preference = st.selectbox("Dietary Preference", ["Normal", "Vegetarian"])
    preferred_cuisine = st.text_input("Preferred Cuisine (e.g., Indian, Mediterranean)", "")
with col7:
    dietary_restrictions = st.text_input("Dietary Restrictions (e.g., Gluten-Free, Low-Carb)", "")
    allergies = st.text_input("Allergies (e.g., Peanuts, Dairy)", "")

# --- Prediction Button ---
if st.button("Get My Health Report", type="primary"):
    if model is None:
        st.error("Cannot proceed. Model file is missing or invalid.")
    else:
        # --- 3. Process User Input and Make Prediction ---
        user_data = {
            'Age': float(age),
            'Gender': gender,
            'Height_cm': float(height_cm),
            'Weight_kg': float(weight_kg),
            'Daily_Caloric_Intake': float(daily_caloric_intake),
            'Cholesterol_mg/dL': float(cholesterol),
            'Blood_Pressure_mmHg': float(blood_pressure),
            'Glucose_mg/dL': float(glucose),
            'Calories_per_Exercise_Hour': 300,  # A placeholder value for now
            'Activity_Level': activity_level,
            'Dietary_Preference': dietary_preference,
            'Preferred_Cuisine': preferred_cuisine,
            'Dietary_Restrictions': dietary_restrictions,
            'Allergies': allergies,
        }

        # Calculate engineered features
        user_data['BMI'] = user_data['Weight_kg'] / (user_data['Height_cm'] / 100)**2
        user_data['Cardiovascular_Risk_Score'] = (user_data['Age'] * 0.005) + (user_data['Cholesterol_mg/dL'] * 0.001) + (user_data['Blood_Pressure_mmHg'] * 0.002)
        user_data['Metabolic_Health_Score'] = 1 / (1 + np.exp(-(130 - user_data['Glucose_mg/dL']) / 20))
        user_data['Lifestyle_Health_Score'] = 0.8 - (0.1 * (user_data['BMI'] > 25))
        user_data['Multiple_Risk_Factors'] = int((user_data['Blood_Pressure_mmHg'] >= 130) or (user_data['Glucose_mg/dL'] > 100))
        user_data['BMI_Age_Interaction'] = user_data['Age'] * user_data['BMI']
        user_data['Cholesterol_Age_Ratio'] = user_data['Cholesterol_mg/dL'] / user_data['Age']
        user_data['BMI_Squared'] = user_data['BMI']**2
        user_data['Glucose_Squared'] = user_data['Glucose_mg/dL']**2
        user_data['Glucose_over_BMI'] = user_data['Glucose_mg/dL'] / user_data['BMI']
        user_data['Chol_over_BMI'] = user_data['Cholesterol_mg/dL'] / user_data['BMI']
        user_data['BP_over_Age'] = user_data['Blood_Pressure_mmHg'] / (user_data['Age'] + 1)
        user_data['BPxBMI'] = user_data['Blood_Pressure_mmHg'] * user_data['BMI']
        user_data['BP_flag'] = int(user_data['Blood_Pressure_mmHg'] >= 130)
        user_data['BP_stage1_flag'] = int((user_data['Blood_Pressure_mmHg'] >= 130) and (user_data['Blood_Pressure_mmHg'] < 140))
        user_data['BP_stage2_flag'] = int(user_data['Blood_Pressure_mmHg'] >= 140)
        user_data['BP_hinge_130'] = max(0, user_data['Blood_Pressure_mmHg'] - 130)
        user_data['BP_hinge_140'] = max(0, user_data['Blood_Pressure_mmHg'] - 140)
        user_data['BP_per_BMI'] = user_data['Blood_Pressure_mmHg'] / (user_data['BMI'] + 1)
        user_data['BPx_Metabolic'] = user_data['Blood_Pressure_mmHg'] * user_data['Metabolic_Health_Score']
        user_data['BPx_CardioRisk'] = user_data['Blood_Pressure_mmHg'] * user_data['Cardiovascular_Risk_Score']
        user_data['Calories_per_BMI'] = user_data['Daily_Caloric_Intake'] / (user_data['BMI'] + 1)
        user_data['Log_BP'] = np.log1p(user_data['Blood_Pressure_mmHg'])

        # Prepare data for the model
        user_df = pd.DataFrame([user_data])
        user_df = pd.get_dummies(user_df, columns=['Gender', 'Activity_Level'], prefix=['Gender', 'Activity'])
        
        missing_cols = set(feature_names) - set(user_df.columns)
        for c in missing_cols:
            user_df[c] = 0
        user_df = user_df[feature_names]

        # Scale the data
        user_scaled = scaler.transform(user_df)

        # Make prediction
        if 'xgboost' in model_info.get('model_type', '').lower():
            y_pred_probs = []
            for label in mlb.classes_:
                y_pred_probs.append(model[label].predict_proba(user_scaled)[:, 1])
            y_pred_probs = np.array(y_pred_probs).T
        else: # TensorFlow
            y_pred_probs = model.predict(user_scaled)

        predicted_labels = (y_pred_probs > 0.5).astype(int)
        predicted_diseases = mlb.inverse_transform(predicted_labels)[0]

        # --- 4. Display Results ---
        st.subheader("Your Health Analysis & Recommendations")
        st.markdown("---")
        
        # Display Predictions
        st.subheader("Risk Profile 📉")
        if predicted_diseases:
            for i, disease in enumerate(mlb.classes_):
                confidence = y_pred_probs[0][i] * 100
                if disease in predicted_diseases:
                    st.success(f"**{disease}**: {confidence:.2f}% confidence")
                else:
                    st.info(f"**{disease}**: {confidence:.2f}% confidence")
        else:
            st.success("You appear to be at low risk for the analyzed diseases. Keep up the good work! 🎉")

        # Display Diet Plan
        st.subheader("Personalized Diet Plan 🍎")
        report = recommender.get_diet_recommendation(predicted_diseases, user_data)
        st.markdown(report, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Disclaimer: This is for educational purposes only and not a substitute for medical advice. Consult a healthcare professional.")
