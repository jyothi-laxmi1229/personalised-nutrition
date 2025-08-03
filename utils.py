import pandas as pd
from datetime import datetime, timedelta

def get_weekly_macros(username):
    import pandas as pd
    from datetime import datetime, timedelta

    # Load meal logs
    df = pd.read_csv("user_meal_logs.csv")
    df.columns = df.columns.str.strip()  # Clean up column names

    # Debug: Show available columns
    print("Meal log columns:", df.columns.tolist())

    # Check for required columns
    if 'Username' not in df.columns or 'Date' not in df.columns or 'Dish' not in df.columns:
        raise KeyError("Required columns ('Username', 'Date', 'Dish') not found in user_meal_logs.csv")

    # Parse date and filter by last 7 days
    df['Date'] = pd.to_datetime(df['Date'])
    one_week_ago = datetime.now() - timedelta(days=7)
    user_data = df[(df['Username'] == username) & (df['Date'] >= one_week_ago)]

    # Load nutrition data
    nutrition = pd.read_csv("Indian_Food_Nutrition_Filled.csv")
    nutrition.columns = nutrition.columns.str.strip()  # Clean up column names

    # Rename columns to match expected names
    nutrition.rename(columns={
        'Carbohydrates (g)': 'Carbohydrates',
        'Protein (g)': 'Proteins',
        'Fats (g)': 'Fats'
    }, inplace=True)

    # Debug: Show nutrition file columns after renaming
    print("Nutrition columns (renamed):", nutrition.columns.tolist())

    # Check if required columns are present
    required_columns = ['Dish Name', 'Carbohydrates', 'Proteins', 'Fats']
    for col in required_columns:
        if col not in nutrition.columns:
            raise KeyError(f"Required column '{col}' not found in Indian_Food_Nutrition_Filled.csv")

    # Merge user meals with nutrition info
    merged = pd.merge(user_data, nutrition, left_on='Dish', right_on='Dish Name', how='left')

    # Handle missing macro values
    merged[['Carbohydrates', 'Proteins', 'Fats']] = merged[['Carbohydrates', 'Proteins', 'Fats']].fillna(0)

    # Sum up macros
    total_carbs = merged['Carbohydrates'].sum()
    total_protein = merged['Proteins'].sum()
    total_fats = merged['Fats'].sum()

    return {
        'Carbohydrates': round(total_carbs, 2),
        'Proteins': round(total_protein, 2),
        'Fats': round(total_fats, 2)
    }


def analyze_macro_distribution(username):
    macros = get_weekly_macros(username)

    carbs = macros.get("Carbohydrates", 0)
    protein = macros.get("Proteins", 0)
    fats = macros.get("Fats", 0)
    total = carbs + protein + fats

    if total == 0:
        return "No data available to analyze. Please log your meals."

    # Calculate percentage
    perc_carbs = (carbs / total) * 100
    perc_protein = (protein / total) * 100
    perc_fats = (fats / total) * 100

    insights = []

    # Based on dietary guidelines (can be refined later)
    if perc_carbs > 60:
        insights.append("High carbohydrate intake detected. Reduce sugary and refined carbs to avoid blood sugar spikes.")
    elif perc_carbs < 40:
        insights.append("Low carb intake. You may feel tired or low on energy. Consider including more whole grains or fruits.")

    if perc_protein < 15:
        insights.append("Low protein intake. May cause muscle loss or fatigue. Add more legumes, dairy, or lean meat.")
    elif perc_protein > 30:
        insights.append("High protein intake. Ensure adequate hydration and kidney health.")

    if perc_fats > 35:
        insights.append("High fat intake. May lead to weight gain or heart issues. Use healthier fats like nuts and seeds.")
    elif perc_fats < 15:
        insights.append("Too little fat in the diet. Healthy fats are needed for hormones and brain function.")

    if not insights:
        insights.append("Your macronutrient intake seems balanced. Keep up the good work!")

    return "<br>".join(insights)

