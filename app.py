import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset', 'salary_prediction_data.csv')
METRICS_PATH = os.path.join(BASE_DIR, 'Dataset', 'metrics.json')
IMAGES_DIR = os.path.join(BASE_DIR, 'Images')

st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E3A5F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A7A9A;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f4f8, #e2eaf3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #d0dce8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6B8BAE;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .highlight-green {
        color: #0B8754;
        font-weight: 700;
    }
    .highlight-blue {
        color: #1E6BB8;
        font-weight: 700;
    }
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        border: 1px solid #eaeef2;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1E3A5F, #2B5A87);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 24px rgba(30,58,95,0.2);
    }
    .prediction-amount {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'page' not in st.session_state:
    st.session_state.page = 'predict'

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)

@st.cache_resource
def bucket_experience_level(years):
    if years < 2:
        return 'Entry'
    elif years < 5:
        return 'Junior'
    elif years < 10:
        return 'Mid'
    elif years < 15:
        return 'Senior'
    else:
        return 'Expert'

def train_models():
    df = load_data()

    df = df.drop_duplicates().reset_index(drop=True)
    edu_map = {"Bachelor's Degree": "Bachelor's", "Masters": "Master's",
               "Master's Degree": "Master's", "phd": "PhD", "Highschool": "High School"}
    df['Education Level'] = df['Education Level'].map(edu_map).fillna(df['Education Level'])
    job_counts = df['Job Title'].value_counts()
    rare = job_counts[job_counts < 30].index
    df['Job Title'] = df['Job Title'].apply(lambda x: 'Other' if x in rare else x)

    df_fe = df.copy()
    df_fe['experience_level'] = df_fe['Years of Experience'].apply(bucket_experience_level)
    df_fe['age_experience_ratio'] = np.where(
        df_fe['Years of Experience'] > 0,
        df_fe['Age'] / df_fe['Years of Experience'],
        df_fe['Age'] * 2
    )

    y = df['Salary']

    X_base = pd.get_dummies(df[['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']],
                            columns=['Gender', 'Education Level', 'Job Title'], drop_first=True, dtype=int)

    X_eng = pd.get_dummies(df_fe[['Age', 'Gender', 'Education Level', 'Job Title',
                                   'Years of Experience', 'experience_level', 'age_experience_ratio']],
                           columns=['Gender', 'Education Level', 'Job Title', 'experience_level'],
                           drop_first=True, dtype=int)

    X_train_e, X_test_e, y_train, y_test = train_test_split(X_eng, y, test_size=0.2, random_state=42)
    X_train_b, X_test_b, _, _ = train_test_split(X_base, y, test_size=0.2, random_state=42)

    lr_b = LinearRegression().fit(X_train_b, y_train)
    lr_e = LinearRegression().fit(X_train_e, y_train)

    rf_best = RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_split=5,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    rf_best.fit(X_train_e, y_train)

    xgb_best = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.2,
        subsample=0.8, colsample_bytree=1.0, random_state=42, verbosity=0
    )
    xgb_best.fit(X_train_e, y_train)

    base_columns = X_train_b.columns.tolist()
    eng_columns = X_train_e.columns.tolist()

    return {
        'lr_baseline': lr_b,
        'lr_engineered': lr_e,
        'rf': rf_best,
        'xgb': xgb_best,
        'base_columns': base_columns,
        'eng_columns': eng_columns,
        'X_test_e': X_test_e,
        'y_test': y_test
    }

def preprocess_input_baseline(age, gender, education, job_title, experience, base_columns):
    data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience
    }
    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input, columns=['Gender', 'Education Level', 'Job Title'],
                              drop_first=True, dtype=int)
    for col in base_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[base_columns]
    return df_input

def preprocess_input_engineered(age, gender, education, job_title, experience, eng_columns):
    exp_level = bucket_experience_level(experience)
    age_exp_ratio = age / experience if experience > 0 else age * 2

    data = {
        'Age': age,
        'Gender': gender,
        'Education Level': education,
        'Job Title': job_title,
        'Years of Experience': experience,
        'experience_level': exp_level,
        'age_experience_ratio': age_exp_ratio
    }
    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input, columns=['Gender', 'Education Level', 'Job Title', 'experience_level'],
                              drop_first=True, dtype=int)
    for col in eng_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[eng_columns]
    return df_input

st.sidebar.markdown("""
<div style="text-align:center; padding: 1rem 0;">
    <span style="font-size:3rem;">💰</span>
    <h2 style="color:#1E3A5F; margin:0;">Salary<br>Predictor</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🎯 Predict Salary", "📊 EDA & Insights", "📈 Model Performance", "ℹ️ About Project"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#8A9BB0; text-align:center;">
    <p><strong>AIML Mini Project</strong><br>
    Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)

if page.startswith("🎯"):
    st.markdown('<div class="main-header">🎯 Salary Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter candidate details to get a data-driven salary prediction</div>', unsafe_allow_html=True)

    df = load_data()
    metrics = load_metrics()

    edu_levels = sorted(['High School', "Bachelor's", "Master's", 'PhD'])
    job_titles = sorted(df['Job Title'].unique().tolist())
    genders = ['Male', 'Female']

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("🎂 Age", min_value=18, max_value=75, value=30, step=1)
            gender = st.selectbox("⚧️ Gender", genders)

        with col2:
            education = st.selectbox("🎓 Education Level", edu_levels)
            job_title = st.selectbox("💼 Job Title", job_titles)

        with col3:
            experience = st.number_input("📅 Years of Experience", min_value=0, max_value=50, value=5, step=1)
            model_choice = st.selectbox(
                "🤖 Select Model",
                ["XGBoost (Best - R² 0.9655)", "Random Forest (R² 0.9645)",
                 "Linear Regression (Engineered)", "Linear Regression (Baseline)"]
            )

        st.markdown('</div>', unsafe_allow_html=True)

    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_btn = st.button("💫 Predict Salary", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("🤖 Running prediction models..."):
            models = train_models()

            X_base_input = preprocess_input_baseline(age, gender, education, job_title, experience, models['base_columns'])
            X_eng_input = preprocess_input_engineered(age, gender, education, job_title, experience, models['eng_columns'])

            pred_lr_b = models['lr_baseline'].predict(X_base_input)[0]
            pred_lr_e = models['lr_engineered'].predict(X_eng_input)[0]
            pred_rf = models['rf'].predict(X_eng_input)[0]
            pred_xgb = models['xgb'].predict(X_eng_input)[0]

            model_map = {
                "XGBoost (Best - R² 0.9655)": ("XGBoost (Tuned)", pred_xgb, "#1E6BB8"),
                "Random Forest (R² 0.9645)": ("Random Forest (Tuned)", pred_rf, "#0B8754"),
                "Linear Regression (Engineered)": ("Linear Regression (Eng)", pred_lr_e, "#D4783C"),
                "Linear Regression (Baseline)": ("Linear Regression (Base)", pred_lr_b, "#8A9BB0")
            }
            selected_name, selected_pred, selected_color = model_map[model_choice]

        st.markdown("""
        <div style="height: 1rem;"></div>
        """, unsafe_allow_html=True)

        col_pred, col_detail = st.columns([3, 2])

        with col_pred:
            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:1rem; opacity:0.85; margin-bottom:0.3rem;">{selected_name} Prediction</div>
                <div class="prediction-amount">₹{selected_pred:,.0f}</div>
                <div style="font-size:0.9rem; opacity:0.75;">Annual Salary (₹)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📋 Candidate Profile")
            profile_df = pd.DataFrame({
                'Attribute': ['Age', 'Gender', 'Education', 'Job Title', 'Experience'],
                'Value': [f"{age} years", gender, education, job_title, f"{experience} years"]
            })
            st.dataframe(profile_df, hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div style="height: 1rem;"></div>
        <h3>📊 All Model Predictions</h3>
        """, unsafe_allow_html=True)

        pred_data = pd.DataFrame({
            'Model': ['Linear Regression (Baseline)', 'Linear Regression (Engineered)',
                      'Random Forest (Tuned)', 'XGBoost (Tuned)'],
            'Predicted Salary': [f"₹{pred_lr_b:,.0f}", f"₹{pred_lr_e:,.0f}",
                                 f"₹{pred_rf:,.0f}", f"₹{pred_xgb:,.0f}"],
            'R² Score': [f"{metrics['linear_regression']['baseline']['r2']:.4f}",
                         f"{metrics['linear_regression']['engineered']['r2']:.4f}",
                         f"{metrics['random_forest']['r2']:.4f}",
                         f"{metrics['xgboost']['r2']:.4f}"],
            'MAE': [f"₹{metrics['linear_regression']['baseline']['mae']:,.0f}",
                    f"₹{metrics['linear_regression']['engineered']['mae']:,.0f}",
                    f"₹{metrics['random_forest']['mae']:,.0f}",
                    f"₹{metrics['xgboost']['mae']:,.0f}"]
        })

        st.dataframe(pred_data, hide_index=True, use_container_width=True)

        st.markdown("### 📈 Prediction Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        models_names = ['LR\nBaseline', 'LR\nEngineered', 'RF\nTuned', 'XGBoost\nTuned']
        preds = [pred_lr_b, pred_lr_e, pred_rf, pred_xgb]
        colors_bar = ['#8A9BB0', '#D4783C', '#0B8754', '#1E6BB8']
        bars = ax.bar(models_names, preds, color=colors_bar, edgecolor='white', width=0.6)
        ax.set_title('Salary Predictions by Model', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Predicted Salary (₹)', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for bar, val in zip(bars, preds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800,
                    f'₹{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div style="height: 0.5rem;"></div>
        """, unsafe_allow_html=True)
        with st.expander("💡 What does this salary mean?"):
            salary_mean = metrics['dataset']['salary_mean']
            salary_median = metrics['dataset']['salary_median']
            salary_min = metrics['dataset']['salary_min']
            salary_max = metrics['dataset']['salary_max']

            st.markdown(f"""
            | Range | Amount |
            |-------|--------|
            | 📉 Dataset Minimum | ₹{salary_min:,} |
            | 📊 Dataset Median | ₹{salary_median:,} |
            | 📈 Dataset Average | ₹{salary_mean:,} |
            | 📈 Dataset Maximum | ₹{salary_max:,} |

            **Predicted Salary: ₹{selected_pred:,.0f}**

            This prediction is based on **{model_choice}** — our best performing model.
            The model uses **Years of Experience** as the strongest predictor, alongside
            job title, education level, age, and engineered features.
            """)

elif page.startswith("📊"):
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Visual insights from the salary prediction dataset</div>', unsafe_allow_html=True)

    df = load_data()
    metrics = load_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['dataset']['samples']:,}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['dataset']['job_titles']}</div>
            <div class="metric-label">Job Titles</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{metrics['dataset']['salary_mean']:,}</div>
            <div class="metric-label">Mean Salary</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{metrics['dataset']['salary_median']:,}</div>
            <div class="metric-label">Median Salary</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Salary Distribution", "🎓 Education & Gender", "📅 Experience vs Salary",
         "💼 Top Job Titles", "🔗 Correlations"]
    )

    with tab1:
        st.markdown("### Salary Distribution")
        st.markdown("The salary data is right-skewed — most salaries fall between **₹40 Lakhs–₹1.2 Crores**.")

        img_path = os.path.join(IMAGES_DIR, 'salary_distribution.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(df['Salary'], bins=50, kde=True, color='steelblue', ax=axes[0])
            axes[0].axvline(df['Salary'].mean(), color='r', ls='--', label=f'Mean: ₹{df["Salary"].mean():,.0f}')
            axes[0].axvline(df['Salary'].median(), color='g', ls='--', label=f'Median: ₹{df["Salary"].median():,.0f}')
            axes[0].legend()
            sns.boxplot(y=df['Salary'], color='lightcoral', ax=axes[1])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            st.info(f"📌 **Key Insight:** The mean (₹{df['Salary'].mean():,.0f}) is higher than the median (₹{df['Salary'].median():,.0f}), indicating high-salary executive outliers pulling the average up.")
        with insight_col2:
            st.info("📌 **Business Impact:** HR should use median rather than mean as a reference for typical salary ranges.")

    with tab2:
        st.markdown("### Salary by Education Level & Gender")
        img_path = os.path.join(IMAGES_DIR, 'salary_by_education_gender.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            edu_order = ['High School', "Bachelor's", "Master's", 'PhD']
            sns.barplot(data=df, x='Education Level', y='Salary', order=edu_order, palette='Blues_d', ax=axes[0], ci=None)
            sns.barplot(data=df, x='Gender', y='Salary', palette='Set2', ax=axes[1], ci=None)
            for ax_ in axes:
                for p in ax_.patches:
                    ax_.annotate(f'₹{p.get_height():,.0f}', (p.get_x()+p.get_width()/2, p.get_height()),
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        insight_col1, insight_col2 = st.columns(2)
        with insight_col1:
            edu_stats = df.groupby('Education Level')['Salary'].mean().round(0).astype(int)
            master_salary = edu_stats["Master's"]
            edu_msg = f"📌 **Education Impact:** PhD holders earn the highest (₹{edu_stats['PhD']:,}), followed by Master's (₹{master_salary:,})."
            st.info(edu_msg)
        with insight_col2:
            gender_stats = df.groupby('Gender')['Salary'].mean().round(0).astype(int)
            gap_pct = round((1 - gender_stats['Female'] / gender_stats['Male']) * 100, 1)
            st.info(f"📌 **Gender Insight:** A {gap_pct}% pay gap exists. This supports the business need for bias-aware compensation tools.")

    with tab3:
        st.markdown("### Years of Experience vs Salary")
        img_path = os.path.join(IMAGES_DIR, 'experience_vs_salary.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.regplot(data=df, x='Years of Experience', y='Salary',
                       scatter_kws={'alpha': 0.3, 's': 10}, line_kws={'color': 'r'}, ax=axes[0])
            axes[1].hexbin(df['Years of Experience'], df['Salary'], gridsize=25, cmap='Blues', mincnt=1)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        corr_val = df['Years of Experience'].corr(df['Salary'])
        st.info(f"📌 **Strongest Correlation:** Experience has a **{corr_val:.2f}** correlation with Salary — the highest of any feature. This confirms experience is the strongest predictor.")

    with tab4:
        st.markdown("### Highest Paying Job Titles")
        img_path = os.path.join(IMAGES_DIR, 'top_paying_jobs.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            avg = df.groupby('Job Title')['Salary'].mean().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(12, 7))
            colors_bar = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 15))
            bars = ax.barh(range(len(avg)), avg.values, color=colors_bar)
            ax.set_yticks(range(len(avg)))
            ax.set_yticklabels(avg.index)
            ax.invert_yaxis()
            for bar, val in zip(bars, avg.values):
                ax.text(val + 500, bar.get_y() + bar.get_height()/2, f'₹{val:,.0f}',
                       va='center', fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.info("📌 **Key Insight:** Executive roles (CEO, CTO, CFO, VP) and senior technical roles (Director of Engineering, Cloud Architect) command the highest salaries — 3-5× more than entry-level positions.")

    with tab5:
        st.markdown("### Correlation Heatmap")
        img_path = os.path.join(IMAGES_DIR, 'correlation_heatmap.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            corr = df[['Age', 'Years of Experience', 'Salary']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt='.4f', cmap='coolwarm', square=True,
                       mask=np.triu(np.ones_like(corr, dtype=bool)), ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("""
        | Feature Pair | Correlation | Interpretation |
        |-------------|-------------|---------------|
        | Experience ↔ Salary | **0.9098** | Very strong positive — more experience → higher salary |
        | Age ↔ Salary | **0.5975** | Moderate positive — partially explained by experience |
        | Age ↔ Experience | **0.6688** | Moderate — older candidates tend to have more experience |
        """)

elif page.startswith("📈"):
    st.markdown('<div class="main-header">📈 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed comparison of all models with evaluation metrics</div>', unsafe_allow_html=True)

    metrics = load_metrics()

    st.markdown("### 🏆 Best Model: XGBoost (Tuned)")
    st.markdown(f"""
    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;">
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metrics['xgboost']['r2']:.4f}</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-label">MAE</div>
            <div class="metric-value">₹{metrics['xgboost']['mae']:,.0f}</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">₹{metrics['xgboost']['rmse']:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Model Comparison Table")
    comp_data = pd.DataFrame({
        'Model': ['Linear Regression (Baseline)', 'Linear Regression (Engineered)',
                  'Random Forest (Tuned)', 'XGBoost (Tuned)'],
        'R² Score': [f"{metrics['linear_regression']['baseline']['r2']:.4f}",
                     f"{metrics['linear_regression']['engineered']['r2']:.4f}",
                     f"{metrics['random_forest']['r2']:.4f}",
                     f"{metrics['xgboost']['r2']:.4f}"],
        'MAE': [f"₹{metrics['linear_regression']['baseline']['mae']:,.2f}",
                f"₹{metrics['linear_regression']['engineered']['mae']:,.2f}",
                f"₹{metrics['random_forest']['mae']:,.2f}",
                f"₹{metrics['xgboost']['mae']:,.2f}"],
        'RMSE': [f"₹{metrics['linear_regression']['baseline']['rmse']:,.2f}",
                 f"₹{metrics['linear_regression']['engineered']['rmse']:,.2f}",
                 f"₹{metrics['random_forest']['rmse']:,.2f}",
                 f"₹{metrics['xgboost']['rmse']:,.2f}"]
    })
    st.dataframe(comp_data, hide_index=True, use_container_width=True)

    st.markdown("### 📈 Visual Comparison")

    img_path = os.path.join(IMAGES_DIR, 'model_comparison.png')
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)

    img_path2 = os.path.join(IMAGES_DIR, 'predicted_vs_actual.png')
    if os.path.exists(img_path2):
        st.image(img_path2, use_container_width=True)

    with st.expander("🔧 Hyperparameter Tuning Details (GridSearchCV)"):
        st.markdown(f"""
        **Random Forest Grid Search:**
        - Parameters searched: `n_estimators` (100/200/300), `max_depth` (10/15/20/None),
          `min_samples_split` (2/5/10), `min_samples_leaf` (1/2/4)
        - Total combinations: **108** × 3-fold CV = **324 fits**
        - **Best Params:** n_estimators=300, max_depth=None, min_samples_split=5, min_samples_leaf=1

        **XGBoost Grid Search:**
        - Parameters searched: `n_estimators` (100/200/300), `max_depth` (3/6/9),
          `learning_rate` (0.05/0.1/0.2), `subsample` (0.7/0.8/1.0), `colsample_bytree` (0.7/0.8/1.0)
        - Total combinations: **243** × 3-fold CV = **729 fits**
        - **Best Params:** n_estimators=300, max_depth=3, learning_rate=0.2, subsample=0.8, colsample_bytree=1.0

        > ⚡ A random subset of **3,000 training samples** was used for faster cross-validation.
        """)

    with st.expander("📐 Feature Engineering Impact"):
        st.markdown(f"""
        **Feature 1: `experience_level`** (Categorical Bucket)
        - Discretizes Years of Experience into: Entry, Junior, Mid, Senior, Expert
        - `experience_level_Expert` had an XGBoost importance score of **0.545**
        - Captures non-linear salary jumps between career stages

        **Feature 2: `age_experience_ratio`** (Numeric Ratio)
        - Formula: Age / Years of Experience (or Age × 2 for zero experience)
        - Captures career efficiency — lower ratios = more established careers

        **Impact on Linear Regression:**
        - R² improved by +{metrics['linear_regression']['improvement']['r2_pct']}%
        - MAE decreased by {metrics['linear_regression']['improvement']['mae_pct']}%
        """)

elif page.startswith("ℹ️"):
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Salary Prediction with Multiple Models — AIML Mini Project</div>', unsafe_allow_html=True)

    metrics = load_metrics()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="card">
        <h3>📋 Problem Statement</h3>
        <p>An HR analytics team wants to estimate a <strong>fair salary</strong> for a candidate based on
        age, education level, job title, and years of experience. The goal is to build a predictive model
        that can recommend compensation offers that are <strong>data-driven, consistent, and bias-aware</strong>.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
        <h3>🔬 Project Workflow</h3>
        <ol>
            <li><strong>Data Collection</strong> — 6,700 synthetic records matching Kaggle schema</li>
            <li><strong>Data Cleaning</strong> — Remove duplicates, standardize education, group rare job titles</li>
            <li><strong>EDA</strong> — 5 visualizations uncovering key relationships</li>
            <li><strong>Feature Engineering</strong> — Created `experience_level` and `age_experience_ratio</li>
            <li><strong>Model Building</strong> — 4 models trained with GridSearchCV hyperparameter tuning</li>
            <li><strong>Evaluation</strong> — Compared R², MAE, RMSE across all models</li>
            <li><strong>Deployment</strong> — Interactive Streamlit web app</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        r2_improvement = round((metrics['xgboost']['r2'] - metrics['linear_regression']['baseline']['r2'])
                              / abs(metrics['linear_regression']['baseline']['r2']) * 100, 2)

        st.markdown(f"""
        <div class="card">
        <h3>🏆 Key Results</h3>
        <ul>
            <li><strong>Best Model:</strong> XGBoost (Tuned)</li>
            <li><strong>R² Score:</strong> {metrics['xgboost']['r2']:.4f}</li>
            <li><strong>MAE:</strong> ₹{metrics['xgboost']['mae']:,.2f}</li>
            <li><strong>Improvement:</strong> +{r2_improvement}% over baseline</li>
            <li><strong>Strongest Predictor:</strong> Years of Experience</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
        <h3>📊 Dataset Summary</h3>
        <ul>
            <li><strong>Records:</strong> {metrics['dataset']['samples']:,}</li>
            <li><strong>Features:</strong> {metrics['dataset']['features_original']} original + {metrics['dataset']['features_engineered']} engineered</li>
            <li><strong>Job Titles:</strong> {metrics['dataset']['job_titles']} unique</li>
            <li><strong>Salary Range:</strong> ₹{metrics['dataset']['salary_min']:,} – ₹{metrics['dataset']['salary_max']:,}</li>
            <li><strong>Average Salary:</strong> ₹{metrics['dataset']['salary_mean']:,}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Technologies Used")
    tech_cols = st.columns(5)
    techs = [
        ("🐍", "Python 3.13"),
        ("📊", "Pandas, NumPy"),
        ("📈", "Matplotlib, Seaborn"),
        ("🤖", "scikit-learn, XGBoost"),
        ("🚀", "Streamlit")
    ]
    for i, (emoji, name) in enumerate(techs):
        with tech_cols[i]:
            st.markdown(f"""
            <div style="text-align:center; padding:1rem; background:#f8fafc; border-radius:10px;">
                <div style="font-size:2rem;">{emoji}</div>
                <div style="font-weight:600; color:#1E3A5F;">{name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📁 Repository Structure")
    st.code("""
AIML-PROJECT-2302221530034/
├── Dataset/
│   ├── salary_prediction_data.csv    # Original dataset (6,700 records)
│   ├── generate_salary_data.py       # Data generation script
│   └── metrics.json                  # Saved model metrics
├── Notebook/
│   └── Salary_Prediction.ipynb       # Jupyter notebook (all cells executed)
├── Images/                           # EDA & model visualizations
├── app.py                           # Streamlit web application
├── run_all.py                        # Complete pipeline script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Project documentation
    """, language="text")

    st.markdown("### 🚀 How to Run Locally")
    st.code("""
# 1. Clone the repository
git clone https://github.com/er-kumardeepak/AIML-PROJECT-2302221530034.git
cd AIML-PROJECT-2302221530034

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python run_all.py

# 4. Launch Streamlit app
streamlit run app.py
    """, language="bash")
