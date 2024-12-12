# @Author: Dhananjay Kumar
# @Date: 10-12-2024
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 10-12-2024
# @Title: Python program to perform Gen AI tasks transforming emails using Gemini API

import google.generativeai as genai
import os
import csv
from dotenv import load_dotenv

# Function to summarize and process an email
def email_summarization(content, chat_session):
    try:
        # Summarize the email
        summary_prompt = f"Summarize the email below in two lines: {content}"
        summary_response = chat_session.send_message(summary_prompt)
        summary = summary_response.text

        # Translate the summary to Hindi
        translation_prompt = f"Translate the following text to Hindi: {summary}"
        translation_response = chat_session.send_message(translation_prompt)
        translation = translation_response.text

        # Extract sender, receiver, and email body information
        sender, receiver, body = extract_email_info(content)

        # Save results to a CSV file
        save_to_csv(sender, receiver, body, summary, translation)

    except Exception as e:
        print(f"Error processing email: {e}")

# Function to extract sender, receiver, and email body information
def extract_email_info(email_content):
    lines = email_content.split("\n")
    sender = next((line.split(":")[1].strip() for line in lines if line.lower().startswith("from:")), "Unknown")
    receiver = next((line.split(":")[1].strip() for line in lines if line.lower().startswith("to:")), "Unknown")
    
    # Extract the body of the email (assuming it's everything after a blank line)
    try:
        body_index = lines.index("") + 1
        body = "\n".join(lines[body_index:]).strip()
    except ValueError:
        body = "No Body Found"
    
    return sender, receiver, body

# Function to save email data to a CSV file
def save_to_csv(sender, receiver, body, summary, translation):
    row = (sender, receiver, body, summary, translation)
    file_name = "email_summary.csv"

    try:
        with open(file_name, 'a', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Check if file is empty to write header
                writer.writerow(['Sender', 'Receiver', 'Body', 'Summary', 'Translation'])
            writer.writerow(row)
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Main function to configure the Generative AI API and process emails
def main():
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

        # Read and process emails from email.txt
        with open('email.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Split emails by a delimiter (e.g., "---" or any suitable separator)
            emails = content.split('---')  # Adjust the delimiter as needed
            
            for email_content in emails:
                if email_content.strip():  # Ignore empty blocks
                    email_summarization(email_content.strip(), chat_session)

    except Exception as e:
        print(f"Error: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()
