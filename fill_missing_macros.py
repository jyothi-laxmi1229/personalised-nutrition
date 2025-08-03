import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

def fill_missing_macros(nutrition_file="Indian_Food_Nutrition_Processing.csv"):
    df = pd.read_csv(nutrition_file)
    df.columns = df.columns.str.strip()

    # Drop rows with missing dish name or calories
    df = df.dropna(subset=['Dish Name', 'Calories (kcal)'])
    df['Calories (kcal)'] = pd.to_numeric(df['Calories (kcal)'], errors='coerce')
    df.dropna(subset=['Calories (kcal)'], inplace=True)

    # TF-IDF + Calories as input
    tfidf = TfidfVectorizer()

    def train_predict_macro(target_col):
        # Split: Train on rows where macro is not missing
        train_data = df.dropna(subset=[target_col])
        predict_data = df[df[target_col].isna()]

        if train_data.empty or predict_data.empty:
            return

        # Pipeline: TF-IDF on Dish + Calories
        X_train_text = train_data['Dish Name']
        X_train_cal = train_data['Calories (kcal)'].values.reshape(-1, 1)
        X_train_text_tfidf = tfidf.fit_transform(X_train_text)
        X_train_combined = pd.concat([
            pd.DataFrame(X_train_text_tfidf.toarray()),
            pd.DataFrame(X_train_cal)
        ], axis=1)

        y_train = train_data[target_col]

        # Train
        model = LinearRegression()
        model.fit(X_train_combined, y_train)

        # Predict
        X_pred_text = predict_data['Dish Name']
        X_pred_cal = predict_data['Calories (kcal)'].values.reshape(-1, 1)
        X_pred_text_tfidf = tfidf.transform(X_pred_text)
        X_pred_combined = pd.concat([
            pd.DataFrame(X_pred_text_tfidf.toarray()),
            pd.DataFrame(X_pred_cal)
        ], axis=1)

        preds = model.predict(X_pred_combined)

        # Fill back predicted values
        df.loc[df[target_col].isna(), target_col] = preds

    # Predict all 3 macros
    for macro in ['Carbohydrates (g)', 'Protein (g)', 'Fats (g)']:
        train_predict_macro(macro)

    # Save the updated data
    df.to_csv("Indian_Food_Nutrition_Filled.csv", index=False)
    print("Missing macros filled and saved to Indian_Food_Nutrition_Filled.csv")
