# @Author: Dhananjay Kumar
# @Date: 13-12-2024
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 13-12-2024
# @Title: Python program to perform Gen AI tasks to store and retrieve embeded documents from MongoDB and generate summaries of asked query using google gemini API Key return the summary the query and PyPDF2 library and perform task using streamlit app.

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from pymongo import MongoClient
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import os


# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase1']
collection = db['mycollection']
print("Connected to MongoDB!")


# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to process query and generate summary
def summarization(content,chat_session):
    try:
        # Summarize the email
        summary_prompt = f"Summarize the information in ten lines: {content}"
        summary_response = chat_session.send_message(summary_prompt)
        summary = summary_response.text
        return summary

    except Exception as e:
        print(f"Error processing email: {e}")
        

# Streamlit App
def main():
    # App Mode Selection
    app_mode = st.sidebar.selectbox("Select Mode", ["Upload PDF", "Ask Query"])

    if app_mode == "Upload PDF":
        # PDF File Upload
        st.title("PDF File Uploader and Text Processing with LangChain")
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp_uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the PDF using PyPDFLoader
            loader = PyPDFLoader("temp_uploaded_file.pdf")
            docs = loader.load()

            # Use RecursiveCharacterTextSplitter to split the document
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            final_docs = splitter.split_documents(docs)
            
            # Insert the documents into MongoDB
            embeddings = OllamaEmbeddings(model="gemma:2b")
            data_to_insert = [
                {
                    "key_original": f"original_chunk_{i}",
                    "value_original": final_docs[i].page_content,
                    "key_vector": f"vector_chunk_{i}",
                    "value_vector": embeddings.embed_documents([final_docs[i].page_content])
                }
                for i in range(len(final_docs))
            ]
            result = collection.insert_many(data_to_insert)
            st.success(f"Inserted {len(result.inserted_ids)} documents into MongoDB.")
            
    elif app_mode == "Ask Query":
        # Query Tab
        st.title("Ask a Query Based on Uploaded PDF")
        query = st.text_input("Enter your query:")

        if query:
            # Generate embedding for the query
            st.info("Generating query embedding...")
            embeddings = OllamaEmbeddings(model="gemma:2b")
            query_embedding = embeddings.embed_documents([query])[0]

            # Fetch all documents from the collection
            st.info("Fetching data from the database...")
            documents = list(collection.find())

            # Progress bar initialization
            st.info("Calculating similarity scores...")
            progress_bar = st.progress(0)  # Progress bar
            total_documents = len(documents)
            
            # Compute similarity scores and store results
            results = []
            for idx, doc in enumerate(documents):
                for vector in doc["value_vector"]:
                    similarity = cosine_similarity(query_embedding, vector)
                    results.append({
                        "key_original": doc["key_original"],
                        "value_original": doc["value_original"],
                        "similarity_score": similarity
                    })
                # Update the progress bar
                progress_bar.progress((idx + 1) / total_documents)

            # Sort results by similarity score (descending)
            results.sort(key=lambda r: r["similarity_score"], reverse=True)
            top_results = results[:3]  # Top 3 results

            # Display Top Results
            st.subheader("Top 3 Similar Results:")
            z = []
            for i, res in enumerate(top_results, 1):
                st.write(f"**Rank {i}:**")
                st.write(f"**Chunk:** {res['value_original']}")
                st.write(f"**Similarity Score:** {res['similarity_score']:.4f}")
                z.append(res["value_original"])
    # Load API key from environment variable
    try:
        # Load API key from environment variable
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Ensure it's set in the .env file.")
        genai.configure(api_key=api_key)

        # Create a generative model
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        chat_session = model.start_chat()

        summary = summarization(z, chat_session)
        st.subheader("Summary of Results:")
        st.write(summary)

    except Exception as e:
        st.write(f"Error: {e}")

           
if __name__ == "__main__":
    main()
