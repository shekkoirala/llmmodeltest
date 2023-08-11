from flask import Flask, request, jsonify
import os
import tiktoken
from getpass import getpass
from rich.markdown import Markdown
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import openai

app = Flask(__name__)

# Set your OpenAI API key
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

# # Other configurations
# os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
# os.environ["WANDB_PROJECT"] = "llmapps"

# Define your directories and model name
MODEL_NAME = "text-davinci-003"

def find_md_files(directory):
    "Find all markdown files in a directory and return a LangChain Document"
    dl = DirectoryLoader(directory, "**/*.md")
    return dl.load()

tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

md_text_splitter = MarkdownTextSplitter(chunk_size=1000)

embeddings = OpenAIEmbeddings()

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/retrieve_and_answer', methods=['POST'])
def retrieve_and_answer():
    data = request.get_json()
    query = data['query']
    
    # Document retrieval and question-answering logic
    documents = find_md_files('/home/shekhar/Desktop/flaskapi/data/')
    document_sections = md_text_splitter.split_documents(documents)
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
    token_counts = [len(tokenizer.encode(document.page_content)) for document in document_sections]
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(document_sections, embeddings)
    retriever = db.as_retriever(search_kwargs=dict(k=3))
    docs = retriever.get_relevant_documents(query)
    
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_template = """Use the following pieces of context to answer the question at the end.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]).format(context=context, question=query)
    
    llm = OpenAI()
    response = llm.predict(prompt)
    answer = response    
    # return jsonify({"answer": str(answer)})
    messages = [
    {"role": "system", "content": "You are a helpful assistant that is expert in mermaid script generation."},
    {"role": "user", "content": f"Generate mermaid script to show the sequence diagram for the following. {answer}"}
    ]   

    # Generate a response using the OpenAI API in answering mode
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 Turbo model for answering
        messages=messages
    )

    # Get the generated text from the response
    generated_text = response.choices[0].message['content']
    
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
