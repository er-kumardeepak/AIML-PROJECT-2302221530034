import numpy as np
import pandas as pd

np.random.seed(42)

NUM_SAMPLES = 6700

job_titles = [
    "Software Engineer", "Data Scientist", "Data Analyst", "Machine Learning Engineer",
    "Senior Software Engineer", "Senior Data Scientist", "Junior Developer",
    "DevOps Engineer", "Product Manager", "Project Manager", "Business Analyst",
    "Marketing Manager", "Sales Manager", "HR Manager", "Financial Analyst",
    "Accountant", "Chief Executive Officer", "Chief Technology Officer",
    "Chief Financial Officer", "Operations Manager", "Graphic Designer",
    "UX Designer", "UI Designer", "IT Support Specialist", "Network Engineer",
    "Database Administrator", "Systems Administrator", "Security Engineer",
    "Cloud Architect", "Technical Writer", "Research Scientist", "Statistician",
    "Economist", "Consultant", "Director of Engineering", "Vice President",
    "Analyst", "Associate", "Intern", "Customer Support Specialist",
    "Healthcare Specialist", "Teacher", "Professor", "Legal Assistant",
    "Lawyer", "Architect", "Civil Engineer", "Mechanical Engineer",
    "Electrical Engineer", "Biomedical Engineer"
]

job_salary_base = {
    "Software Engineer": 85, "Data Scientist": 95, "Data Analyst": 65,
    "Machine Learning Engineer": 110, "Senior Software Engineer": 130,
    "Senior Data Scientist": 140, "Junior Developer": 55,
    "DevOps Engineer": 100, "Product Manager": 105, "Project Manager": 85,
    "Business Analyst": 70, "Marketing Manager": 75, "Sales Manager": 80,
    "HR Manager": 70, "Financial Analyst": 75, "Accountant": 60,
    "Chief Executive Officer": 180, "Chief Technology Officer": 170,
    "Chief Financial Officer": 165, "Operations Manager": 85,
    "Graphic Designer": 50, "UX Designer": 80, "UI Designer": 75,
    "IT Support Specialist": 45, "Network Engineer": 80,
    "Database Administrator": 85, "Systems Administrator": 70,
    "Security Engineer": 105, "Cloud Architect": 135, "Technical Writer": 60,
    "Research Scientist": 90, "Statistician": 80, "Economist": 85,
    "Consultant": 90, "Director of Engineering": 155, "Vice President": 160,
    "Analyst": 60, "Associate": 55, "Intern": 35,
    "Customer Support Specialist": 40, "Healthcare Specialist": 65,
    "Teacher": 50, "Professor": 85, "Legal Assistant": 45,
    "Lawyer": 120, "Architect": 85, "Civil Engineer": 75,
    "Mechanical Engineer": 78, "Electrical Engineer": 82,
    "Biomedical Engineer": 80
}

education_levels = ["High School", "Bachelor's", "Master's", "PhD"]
education_salary_multiplier = {
    "High School": 0.85,
    "Bachelor's": 1.0,
    "Master's": 1.15,
    "PhD": 1.30
}

genders = ["Male", "Female"]

ages = np.random.randint(21, 66, size=NUM_SAMPLES)

genders_arr = np.random.choice(genders, size=NUM_SAMPLES, p=[0.52, 0.48])

education_arr = []
for age in ages:
    if age < 25:
        p = [0.20, 0.55, 0.20, 0.05]
    elif age < 35:
        p = [0.08, 0.40, 0.35, 0.17]
    elif age < 50:
        p = [0.05, 0.30, 0.40, 0.25]
    else:
        p = [0.10, 0.35, 0.35, 0.20]
    education_arr.append(np.random.choice(education_levels, p=p))

job_titles_arr = []
years_exp_arr = []

for i in range(NUM_SAMPLES):
    age = ages[i]
    edu = education_arr[i]

    if edu == "High School":
        min_exp = max(0, age - 18)
    elif edu == "Bachelor's":
        min_exp = max(0, age - 22)
    elif edu == "Master's":
        min_exp = max(0, age - 24)
    else:
        min_exp = max(0, age - 28)

    years_exp = int(np.random.uniform(0, max(1, min_exp + 1)))
    years_exp = min(years_exp, 45)
    years_exp_arr.append(years_exp)

    if years_exp < 2:
        pool = ["Intern", "Junior Developer", "Analyst", "Associate",
                "Customer Support Specialist", "IT Support Specialist"]
    elif years_exp < 5:
        pool = ["Junior Developer", "Data Analyst", "Business Analyst",
                "Software Engineer", "Graphic Designer", "UX Designer",
                "IT Support Specialist", "Analyst", "Associate",
                "Customer Support Specialist", "Teacher", "Financial Analyst",
                "Accountant", "Healthcare Specialist"]
    elif years_exp < 10:
        pool = ["Software Engineer", "Data Scientist", "Data Analyst",
                "DevOps Engineer", "Product Manager", "Project Manager",
                "Marketing Manager", "Sales Manager", "HR Manager",
                "Financial Analyst", "Accountant", "Network Engineer",
                "Systems Administrator", "Security Engineer", "UX Designer",
                "Consultant", "Statistician", "Civil Engineer",
                "Mechanical Engineer", "Electrical Engineer", "Architect",
                "Database Administrator", "Biomedical Engineer"]
    elif years_exp < 15:
        pool = ["Software Engineer", "Data Scientist", "Machine Learning Engineer",
                "Senior Software Engineer", "DevOps Engineer", "Product Manager",
                "Project Manager", "Operations Manager", "Marketing Manager",
                "Sales Manager", "Consultant", "Research Scientist",
                "Statistician", "Economist", "Database Administrator",
                "Security Engineer", "Cloud Architect", "Professor",
                "Lawyer", "Architect", "Civil Engineer", "Electrical Engineer",
                "Mechanical Engineer", "Technical Writer"]
    else:
        pool = ["Senior Software Engineer", "Senior Data Scientist",
                "Machine Learning Engineer", "Director of Engineering",
                "Vice President", "Chief Technology Officer",
                "Chief Executive Officer", "Chief Financial Officer",
                "Product Manager", "Operations Manager", "Cloud Architect",
                "Professor", "Lawyer", "Consultant", "Research Scientist",
                "Security Engineer", "Database Administrator", "Economist"]

    job_titles_arr.append(np.random.choice(pool))

salaries = []
for i in range(NUM_SAMPLES):
    job = job_titles_arr[i]
    edu = education_arr[i]
    exp = years_exp_arr[i]
    age = ages[i]
    gender = genders_arr[i]

    base = job_salary_base.get(job, 70)
    edu_mult = education_salary_multiplier[edu]
    exp_factor = 1 + 0.03 * exp - 0.0003 * exp ** 2
    exp_factor = max(exp_factor, 0.5)
    age_factor = 1 - 0.002 * (age - 40) ** 2 / 100
    age_factor = max(age_factor, 0.85)
    gender_factor = 1.0 if gender == "Male" else 0.97
    noise = np.random.normal(1, 0.08)

    # Realistic domestic Indian salary scaling (in ₹ / LPA)
    # Multiplier of 10,000 yields realistic domestic Indian compensation:
    # Entry level (0-2 yrs): ₹2.5L - ₹7.5L, Mid level: ₹8L - ₹20L, Senior: ₹20L - ₹45L, Executive: ₹50L - ₹1.6 Cr
    INR_SCALE_FACTOR = 10000
    salary = base * edu_mult * exp_factor * age_factor * gender_factor * noise * INR_SCALE_FACTOR
    salary = max(salary, 220000)
    salaries.append(int(round(salary)))

df = pd.DataFrame({
    "Age": ages,
    "Gender": genders_arr,
    "Education Level": education_arr,
    "Job Title": job_titles_arr,
    "Years of Experience": years_exp_arr,
    "Salary": salaries
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = r"D:\\Salary_Prediction\\Dataset\\salary_prediction_data.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nSalary range: ₹{df['Salary'].min():,.0f} - ₹{df['Salary'].max():,.0f}")
print(f"Average salary: ₹{df['Salary'].mean():,.0f}")
print(f"Median salary: ₹{df['Salary'].median():,.0f}")
print(f"\nUnique Job Titles: {df['Job Title'].nunique()}")
print(f"\nSample data:")
print(df.head(10))
print(f"\nEducation distribution:")
print(df["Education Level"].value_counts())
print(f"\nGender distribution:")
print(df["Gender"].value_counts())
