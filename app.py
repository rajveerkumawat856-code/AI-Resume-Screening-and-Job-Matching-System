
import os
import joblib
import pymupdf
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Screening System")

@st.cache_resource
def load_classifier():
    return joblib.load("resume_screening_model.pkl")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path):
    text = ""
    with pymupdf.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text() + "\n"
    return text.strip()

folder_path = st.text_input(
    "Enter the folder path containing PDF resumes",
    placeholder=r"C:\Users\YourName\Downloads\resumes"
)

job_category = st.selectbox(
    "Select the job category",
    [
        "ACCOUNTANT",
        "BANKING",
        "BUSINESS-DEVELOPMENT",
        "DESIGNER",
        "ENGINEERING",
        "FINANCE",
        "HEALTHCARE",
        "HR",
        "INFORMATION-TECHNOLOGY",
        "TEACHER"
    ]
)

job_description = st.text_area(
    "Enter the job description",
    height=220,
    placeholder="Paste the complete job description here..."
)

top_n = st.number_input(
    "Number of top candidates to display",
    min_value=1,
    max_value=100,
    value=10,
    step=1
)

if st.button("Screen Resumes", type="primary"):
    if not folder_path.strip():
        st.error("Please enter a resume folder path.")
    elif not os.path.isdir(folder_path.strip()):
        st.error("Folder not found. Please check the path.")
    elif not job_description.strip():
        st.error("Please enter a job description.")
    else:
        folder_path = folder_path.strip()
        resume_names = []
        resume_texts = []
        errors = []

        pdf_files = sorted(
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(folder_path, filename))
        )

        if not pdf_files:
            st.warning("No PDF files found in the selected folder.")
        else:
            for filename in pdf_files:
                file_path = os.path.join(folder_path, filename)

                try:
                    text = extract_text(file_path)
                    if text:
                        resume_names.append(filename)
                        resume_texts.append(text)
                    else:
                        errors.append(f"{filename}: No extractable text")
                except Exception as e:
                    errors.append(f"{filename}: {e}")

            if not resume_texts:
                st.error("Could not extract text from any PDF resumes.")
            else:
                try:
                    classifier = load_classifier()
                    embedding_model = load_embedding_model()

                    predictions = classifier.predict(resume_texts)

                    filtered_names = []
                    filtered_texts = []
                    filtered_predictions = []

                    for name, text, prediction in zip(
                        resume_names, resume_texts, predictions
                    ):
                        if prediction == job_category:
                            filtered_names.append(name)
                            filtered_texts.append(text)
                            filtered_predictions.append(prediction)

                    if not filtered_texts:
                        st.warning(
                            f"No resumes were predicted as {job_category}. "
                            "Try another category or check the resumes."
                        )
                    else:
                        documents = [job_description] + filtered_texts

                        embeddings = embedding_model.encode(
                            documents,
                            batch_size=32,
                            show_progress_bar=False,
                            normalize_embeddings=True
                        )

                        job_embedding = embeddings[0:1]
                        resume_embeddings = embeddings[1:]

                        similarity_scores = cosine_similarity(
                            resume_embeddings,
                            job_embedding
                        ).flatten()

                        results = pd.DataFrame({
                            "Resume Name": filtered_names,
                            "Predicted Category": filtered_predictions,
                            "Cosine Similarity (%)": similarity_scores * 100
                        })

                        results = results.sort_values(
                            by="Cosine Similarity (%)",
                            ascending=False
                        ).reset_index(drop=True)

                        results.insert(
                            0, "Rank", range(1, len(results) + 1)
                        )

                        st.success(
                            f"Found {len(results)} resumes predicted as "
                            f"{job_category}."
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric("Matching resumes", len(results))
                        col2.metric(
                            "Top candidate",
                            results.iloc[0]["Resume Name"]
                        )
                        col3.metric(
                            "Top similarity",
                            f'{results.iloc[0]["Cosine Similarity (%)"]:.2f}%'
                        )

                        st.subheader(f"Ranked {job_category} Candidates")

                        display_results = results.head(int(top_n)).copy()
                        display_results["Cosine Similarity (%)"] = (
                            display_results["Cosine Similarity (%)"].round(2)
                        )

                        st.dataframe(
                            display_results,
                            use_container_width=True,
                            hide_index=True
                        )

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            label="Download ranked candidates CSV",
                            data=csv_data,
                            file_name=f"ranked_{job_category.lower()}_candidates.csv",
                            mime="text/csv"
                        )

                    if errors:
                        with st.expander(
                            f"Skipped or unreadable files ({len(errors)})"
                        ):
                            for error in errors:
                                st.write(error)

                    st.caption(
                        "Similarity scores are ranking signals, not "
                        "probabilities of qualification. Review candidates "
                        "against the actual job requirements."
                    )

                except Exception as e:
                    st.error(f"An error occurred during screening: {e}")
