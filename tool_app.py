import requests
import os
import imaplib
import email
import pdfplumber
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import asyncio
import concurrent.futures
from mysql.connector import Error
import mysql.connector
from queue import Queue
import json
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Global queue to store batch data
batch_data_queue = Queue()

def main(uploaded_file_path=None):

    # Load environment variables from .env file
    load_dotenv()

    # Configure the Google Generative AI API with the API key from the environment variable
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def get_db_connection():
        return mysql.connector.connect(
            host='cheetahdb.cx8yyoqogq59.us-east-1.rds.amazonaws.com',  # e.g., 'RDS Endpoint'
            database='cheetah',  # e.g., 'user_db'
            user='CheetahAI_DB',  # e.g., 'root'
            password=os.getenv("PASSWORD"),  # your MySQL password
            autocommit=True
        )

    model = genai.GenerativeModel('gemini-1.5-flash')
    def get_gemini_response_for_currency_name(cn_val):
        input_text = f"Give me the currency name for the currency code {cn_val}. Just give the answer and do not give any other text"
        response = model.generate_content(input_text)
        return response.text


    def get_exchange_rate_to_inr(api_key, invoice_currency):
        # API endpoint (this one is for ExchangeRate-API)
        url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{invoice_currency}"
        
        # Send the request
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check if the response contains INR conversion rate
            if 'conversion_rates' in data and 'INR' in data['conversion_rates']:
                return data['conversion_rates']['INR']
            else:
                return f"INR conversion rate not available for {invoice_currency}."
        else:
            return f"Error fetching data: {response.status_code}"


    def get_pdf_text(pdf_paths):
        def extract_text(pdf):
            text = ""
            doc = PdfReader(pdf)
            for page in doc.pages:
                text += page.extract_text()
            return text
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            texts = executor.map(extract_text, pdf_paths)
        
        return "".join(texts)

    # Dividing the texts into smaller chunks...
    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    # Convert these chunks into vectors...
    def get_vector_stores(text_chunks):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_stores = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_stores.save_local("faiss-index")

    def get_conversational_chain():
        prompt_template = """
        You are an expert form-filling assistant. Your task is to answer questions based on the provided context, which contains information extracted from relevant documents such as invoices, packing lists, and other shipping-related papers.

        Instructions:
        1. Carefully analyze the entire context before answering each question.
        2. Provide detailed and accurate answers whenever possible.
        3. If the exact information is not present, use your knowledge to infer a reasonable answer based on related information in the context.
        4. Only respond with "n/a" if you are absolutely certain that no relevant information exists in the context and no reasonable inference can be made.
        Remember: Your goal is to fill as many form fields as possible with accurate information. Thoroughly examine the context for both direct and indirect answers to minimize "n/a" responses.

        Context:\n {context}?\n
        Questions: \n{questions}\n
        Answers:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "questions"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    async def get_batch_response(questions, embeddings, vector_stores, semaphore, retries=3):
        async with semaphore:
            for attempt in range(retries):
                try:
                    loop = asyncio.get_event_loop()
                    docs = await loop.run_in_executor(None, vector_stores.similarity_search, questions)
                    chain = get_conversational_chain()
                    response = await loop.run_in_executor(None, chain, {"input_documents": docs, "questions": questions}, True)
                    return response["output_text"]
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for questions '{questions}' with error: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            raise Exception(f"Failed to get response for questions '{questions}' after {retries} attempts")

    def generate_question(field_id):
        return f"{field_id}?"

    # Add this function to check if IEC exists in the database
    def check_iec_in_database(iec):
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            
            query = "SELECT * FROM cha WHERE IEC = %s"
            cursor.execute(query, (iec,))
            result = cursor.fetchone()
            
            return result is not None
        except Error as e:
            print(f"Database error: {e}")
            return False
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()

    async def collect_batch_data(batch, semaphore):
        batch_data = {}
        questions = []
        field_ids = []
        excluded_ids = []  # Initialize excluded_ids list
        
        def clean_string(s, is_problematic_id=False):
            # Remove any leading/trailing whitespace, quotes, and backslashes
            s = s.strip().strip('"\'\\').strip()
            # Remove any JSON formatting artifacts
            s = s.replace('```json', '').replace('```', '')
            s = s.replace('{', '').replace('}', '')
            s = s.replace('\n', '')
            if is_problematic_id:
                # Additional cleaning for problematic IDs
                s = s.lstrip('"').rstrip('"')
            return s
        
        for element in batch:
            field_id = element.get_attribute("id")
            if field_id:
                if field_id == "user_job_date":
                    current_date = datetime.now().strftime("%d-%m-%Y")
                    batch_data[field_id] = current_date
                elif field_id == "Supporting_Documents_supporting_documents_upload_four":
                    file_path = os.path.abspath('./Documents/invoice1r.pdf')
                    batch_data[field_id] = file_path
                elif field_id == "effective_date":
                    curr_date = datetime.now().strftime("%d-%m-%Y")
                    batch_data[field_id] = curr_date
                else:
                    question = generate_question(field_id)
                    questions.append(question)
                    field_ids.append(field_id)
        
        # Check if there are any questions to ask
        if questions:
            combined_questions = "\n".join(questions)
            response = await get_batch_response(combined_questions, embeddings, vector_stores, semaphore)
            
            # Process the response
            lines = response.split(',')
            current_field_id = None
            icc_value = None
            
            for line in lines:
                line = clean_string(line)
                if line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        current_field_id = parts[0].strip()
                        answer = parts[1].strip()
                        
                        # Special handling for problematic IDs
                        if current_field_id in ["jec_avallatiin_yes", "equipment_serial_number"]:
                            current_field_id = clean_string(current_field_id, is_problematic_id=True)
                            answer = clean_string(answer, is_problematic_id=True)
                        else:
                            current_field_id = clean_string(current_field_id)
                            answer = clean_string(answer)
                        
                        # Store the answer without extra quotes
                        batch_data[current_field_id] = answer

                        # For Exchange Rate Details...
                        if current_field_id == "invoice_currency":
                            icc_value = answer

                        elif current_field_id == "currency_name":
                            cn_value = get_gemini_response_for_currency_name(icc_value)
                            batch_data[current_field_id] = clean_string(cn_value)

                        elif current_field_id == "rate":
                            api_key = os.getenv('CURRENCY_API')
                            rate_value = get_exchange_rate_to_inr(api_key, icc_value)
                            batch_data[current_field_id] = clean_string(str(rate_value))

                        # Check if the current field is IEC and update excluded_ids
                        elif current_field_id == "IEC" and check_iec_in_database(answer):
                            excluded_ids.extend(["Exporter_Name", "GSTN_ID"])

        # Put the data in the queue
        batch_data_queue.put({
            "Batch data to be filled": batch_data,
            "Excluded IDs": excluded_ids
        })
        

    # Function to extract the first thirty characters from PDF using pdfplumber
    def extract_first_twenty_chars_from_pdf(filename):
        text = ""
        with pdfplumber.open(filename) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
                if len(text) >= 20:
                    break
        return text[:20].lower()  # Return only the first 20 characters
        
    login_username = os.getenv('LOGIN_USERNAME')
    login_password = os.getenv('LOGIN_PASSWORD')

    # Process the uploaded file
    pdf_docs = []
    if uploaded_file_path and os.path.exists(uploaded_file_path):
        pdf_docs.append(uploaded_file_path)
    else:
        print("No PDF document uploaded.")
        return {"error": "No PDF document uploaded"}
        
    # Measure time for making FAISS index
    start_faiss_time = time.time()
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_stores(text_chunks)
    end_faiss_time = time.time()
    faiss_duration = end_faiss_time - start_faiss_time

    # Load the vector stores and embeddings once
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_stores = FAISS.load_local("faiss-index", embeddings, allow_dangerous_deserialization="True")

    # Set up the webdriver without headless mode
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.binary_location = "/opt/render/project/.render/chrome/opt/google/chrome/chrome"

    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        # Open the webpage
        driver.get("https://cheetah-ai.netlify.app/form")
        driver.maximize_window()

        # Fill in login details
        username_field = driver.find_element(By.ID, "username")  # Replace with actual username field ID
        password_field = driver.find_element(By.ID, "password")  # Replace with actual password field ID
        login_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Login')]")  # Adjust XPATH as needed

        username_field.send_keys(login_username)  # Replace with actual username
        password_field.send_keys(login_password)  # Replace with actual password
        login_button.click()

        # Retrieve all input fields
        # all_elements = driver.find_elements(By.TAG_NAME, "input")
        all_elements = driver.find_elements(By.CSS_SELECTOR, "input:not([type='hidden'])")

        excluded_ids = []

        # Measure time for filling the form
        start_fill_time = time.time()

        # Define a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(2)  # Adjust the number of concurrent requests as needed

        # Run the async function
        asyncio.run(collect_batch_data(all_elements, semaphore=semaphore))

        # After running collect_batch_data, get the result from the queue
        result = batch_data_queue.get() if not batch_data_queue.empty() else None

        return result
        time.sleep(10)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        driver.quit()

# if _name_ == "_main_":
#     main()
