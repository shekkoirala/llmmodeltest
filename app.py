import os
from flask import Flask, request
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from langchain import PromptTemplate

app = Flask(__name__)


documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/query', methods=['GET'])
def query():
    topic = request.args.get("text", None)

    prompt = PromptTemplate(
        input_variables=["topic"],
        template="You are an expert in mermaid script generation. You need to generate the mermaid script for the following: {topic}."
    )

    responses = prompt.format(topic=topic)

    query_engine = index.as_query_engine()
    response = query_engine.query(responses)

    return str(response), 200

if __name__ == '__main__':
    app.run(debug=True)
