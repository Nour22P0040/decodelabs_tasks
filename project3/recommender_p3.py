import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. DATA AND SETUP
# ==========================================
catalog = [
    {"role": "Data Scientist",
     "skills": "python machine_learning deep_learning sql statistics data_analysis numpy pandas tensorflow scikit_learn"},
    {"role": "Machine Learning Engineer",
     "skills": "python machine_learning deep_learning tensorflow pytorch model_deployment docker kubernetes mlops"},
    {"role": "Data Analyst",
     "skills": "python sql excel power_bi tableau statistics data_visualization data_analysis reporting"},
    {"role": "Backend Developer",
     "skills": "python java nodejs sql rest_api microservices docker databases postgresql mongodb"},
    {"role": "Frontend Developer", "skills": "javascript react html css typescript nodejs ui_design responsive_design"},
    {"role": "Full Stack Developer", "skills": "javascript python react nodejs sql html css rest_api docker databases"},
    {"role": "DevOps Engineer",
     "skills": "docker kubernetes aws linux ci_cd jenkins git automation bash cloud_computing"},
    {"role": "Cloud Architect",
     "skills": "aws azure gcp cloud_computing networking security infrastructure terraform devops"},
    {"role": "Cybersecurity Analyst",
     "skills": "networking linux security penetration_testing ethical_hacking firewalls encryption siem"},
    {"role": "NLP Engineer",
     "skills": "python nlp machine_learning transformers bert gpt text_processing deep_learning huggingface"},
    {"role": "Computer Vision Engineer",
     "skills": "python opencv deep_learning cnn image_processing tensorflow pytorch computer_vision"},
    {"role": "Database Administrator",
     "skills": "sql postgresql mysql oracle databases performance_tuning backup_recovery data_modeling"},

    # --- CUSTOM HARDWARE & BACKEND ROLES ---
    {"role": "Embedded Systems Engineer",
     "skills": "c++ python arduino pid_control digital_logic udp_tcp data_structures algorithms oop vscode git github"},
    {"role": "IoT Software Engineer",
     "skills": "c++ javascript python nodejs expressjs mongodb mongoose rest_api udp_tcp computer_networks arduino git github jira agile scrum"},
    {"role": "Robotics & Mechatronics Engineer",
     "skills": "c++ python matlab signal_processing pid_control machine_learning algorithms data_structures oop"},
    {"role": "Backend Systems Engineer",
     "skills": "java python javascript nodejs expressjs mongodb mongoose sql_server ssms rest_api computer_networks oop git github jira agile vscode"}
]

df_catalog = pd.DataFrame(catalog)

print("=== CAREER CATALOG LOADED ===")
print(df_catalog[['role']].to_string())
print(f"\nTotal career paths available: {len(df_catalog)}")

# Fit the vectorizer on the catalog
tfidf_vectorizer = TfidfVectorizer()
catalog_tfidf_matrix = tfidf_vectorizer.fit_transform(df_catalog['skills'])

print(f"\n=== VECTORIZATION COMPLETE ===")
print(f"Vocabulary size (unique skills): {len(tfidf_vectorizer.vocabulary_)}")
print(f"Catalog matrix shape: {catalog_tfidf_matrix.shape}")


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def handle_cold_start(strategy="onboarding"):
    """Handles cases where user input has 0 overlap with catalog vocabulary."""
    if strategy == "onboarding":
        print("\n=== ONBOARDING SURVEY ===")
        print("Since you're new or your skills aren't in our system yet, let's find your starting point.")
        print("Q1: What is your primary interest area?")
        print("  A) Web Development   B) Data Science   C) Cloud/DevOps")

        choice = input("Enter A, B, or C: ").strip().upper()

        starter_profiles = {
            "A": "javascript html css react nodejs frontend",
            "B": "python machine_learning sql statistics data_analysis",
            "C": "aws docker kubernetes linux cloud_computing devops"
        }
        return starter_profiles.get(choice, starter_profiles["B"])

    elif strategy == "trending":
        print("\n=== TRENDING RECOMMENDATIONS (Cold Start Fallback) ===")
        trending_roles = ["Data Scientist", "Machine Learning Engineer", "Full Stack Developer"]
        for i, role in enumerate(trending_roles, 1):
            print(f"  {i}. {role}")
        return None


def get_user_profile():
    """Captures and normalizes user input."""
    print("\n" + "=" * 55)
    print("  DecodeLabs Tech Stack Recommender")
    print("  Enter your skills (spaces will be auto-formatted)")
    print("  Example: python, machine learning, sql, aws")
    print("  Minimum 3 skills required for accurate matching.")
    print("=" * 55)

    while True:
        raw_skills = input("\nEnter your skills (comma-separated): ")

        # Auto-normalize spaces to underscores to match catalog tokens
        skills_list = [skill.lower().strip().replace(" ", "_") for skill in raw_skills.split(',')]
        skills_list = [s for s in skills_list if s]

        # Prevent sparse vector instability
        if len(skills_list) < 3:
            print(f"⚠️  Only {len(skills_list)} skill(s) detected. Please enter at least 3.")
            continue

        user_profile_string = " ".join(skills_list)

        print(f"\n✅ Profile accepted: {skills_list}")
        print(f"   Normalized Profile string: '{user_profile_string}'")
        return user_profile_string, skills_list


def recommend_tech_stack(user_profile_string, top_n=3):
    """Calculates cosine similarity and returns top matches + 1 serendipity match."""
    # Check for vocabulary overlap
    user_vocab_check = [
        skill for skill in user_profile_string.split()
        if skill in tfidf_vectorizer.vocabulary_
    ]

    # Proper Cold Start Integration
    if len(user_vocab_check) == 0:
        print("\n⚠️  COLD START DETECTED: None of your skills match our catalog vocabulary.")
        new_profile_string = handle_cold_start(strategy="onboarding")

        if new_profile_string:
            print(f"\n🔄 Rerunning recommendation with starter profile...")
            # Recursive call with the new injected profile
            return recommend_tech_stack(new_profile_string, top_n)
        else:
            return pd.DataFrame()  # Handled by trending

    # Use .transform, not .fit_transform
    user_tfidf_vector = tfidf_vectorizer.transform([user_profile_string])

    # Use Cosine Similarity, not Euclidean
    similarity_scores = cosine_similarity(
        user_tfidf_vector,
        catalog_tfidf_matrix
    )

    scores_flat = similarity_scores.flatten()
    sorted_indices = np.argsort(scores_flat)[::-1]

    top_indices = sorted_indices[:top_n]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            'Rank': str(rank),
            'Career Path': df_catalog.iloc[idx]['role'],
            'Match Score': f"{scores_flat[idx] * 100:.1f}%",
        })

    # The Serendipity Problem (Filter Bubble)
    # Grab the least similar career path to offer an "outside the box" option
    least_similar_idx = sorted_indices[-1]
    if scores_flat[least_similar_idx] < 0.15:  # Only suggest if it's genuinely different
        results.append({
            'Rank': '💡',
            'Career Path': df_catalog.iloc[least_similar_idx]['role'] + " (Explore Something New!)",
            'Match Score': f"{scores_flat[least_similar_idx] * 100:.1f}%",
        })

    return pd.DataFrame(results)


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    user_profile_string, user_skills = get_user_profile()

    print("\n🔍 Analyzing your profile against the catalog...")
    top_recommendations = recommend_tech_stack(user_profile_string, top_n=3)

    if not top_recommendations.empty:
        print("\n" + "=" * 65)
        print("  🏆 YOUR CAREER PATH RECOMMENDATIONS")
        print("=" * 65)
        print(top_recommendations.to_string(index=False))
        print("\n  (Match score = cosine similarity × 100)")
        print("  A score of 100% = perfect skill alignment.")
        print("=" * 65)