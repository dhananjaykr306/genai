# @Author: Dhananjay Kumar
# @Date: 11-12-2024
# @Last Modified by: Dhananjay Kumar
# @Last Modified time: 11-12-2024
# @Title: Python program to perform Gen AI tasks infering sentiment analysis and replying to reviews using Gemini API

import google.generativeai as genai
import os
import csv
from dotenv import load_dotenv
import time

# Function to send messages to the generative AI model with retries
def send_with_retry(session, prompt, retries=3, delay=2):
    """Send a message to the AI model with retry mechanism."""
    for attempt in range(retries):
        try:
            # Try sending the prompt and handle different response types
            response = session.send_message(prompt)
            if isinstance(response, str):
                return response
            elif hasattr(response, 'text'):
                return response.text
            else:
                return "Invalid response"
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)  # Wait before retrying
    raise Exception("All attempts to send message failed.")

# Function to save the processed data to a new CSV file
def save_to_csv(product, review, sentiment, model_sentiment, product_name, reply):
    """Save the review data to a CSV file."""
    try:
        row = (product, review, sentiment, model_sentiment, product_name, reply)
        with open("process_review.csv", 'a', encoding='utf-8') as file:
            writer = csv.writer(file)
            if file.tell() == 0:  # Check if the file is empty to write the header row
                writer.writerow(['Product', 'Review', 'Sentiment', 'Model Sentiment', 'Product Name', 'Reply'])
            writer.writerow(row)  # Write the review data
    except Exception as e:
        print(f"Error saving to CSV: {e}")

# Function to perform sentiment analysis and generate replies
def sentiment_analysis(chat_session):
    """Process reviews in CSV and analyze sentiment using generative AI."""
    try:
        with open("sample_reviews.csv", mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            
            for row in csv_reader:
                try:
                    # Extract review text and prepare prompts
                    review_text = row[1]
                    sentiment_prompt = f"Categorize the sentiment of this review as Positive, Negative, or Neutral: {review_text}"
                    product_prompt = f"What is the name of the product according to the review: {review_text} in one word"
                    reply_prompt = f"Add a 20-word reply to this review: {review_text} based on the sentiment"

                    # Get responses from the AI model
                    model_sentiment = send_with_retry(chat_session, sentiment_prompt)
                    product_name = send_with_retry(chat_session, product_prompt)
                    reply = send_with_retry(chat_session, reply_prompt)

                    # Save the results to the CSV file
                    save_to_csv(row[0], row[1], row[2], model_sentiment, product_name, reply)
                except Exception as inner_e:
                    print(f"Error processing row {row}: {inner_e}")
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")

# Main function to configure the model and start the sentiment analysis process
def main():
    """Main function to load environment variables and start the sentiment analysis."""
    try:
        # Load environment variables from the .env file
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Ensure it's set in the .env file.")

        # Configure the Google generative AI model
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 512,
        }
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        
        # Start a new chat session
        chat_session = model.start_chat(history=[])
        
        # Perform sentiment analysis
        sentiment_analysis(chat_session)
    except Exception as e:
        print(f"Error in main function: {e}")

# Entry point of the script
if __name__ == "__main__":
    main()