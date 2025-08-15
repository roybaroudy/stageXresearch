import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import weaviate
import weaviate.classes.config as wc
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from weaviate.classes.query import MetadataQuery , Filter
from langchain_core.runnables import Runnable
from pdfminer.high_level import extract_text
import json
import ollama
from flask import Flask, request, jsonify
from flask_cors import CORS
import re


dotenv.load_dotenv()
WEAVIATE_URL =  "http://localhost:8080"
app = Flask(__name__)
CORS(app)


def extract_pdf_text(pdf_path):
    full_text = extract_text(pdf_path)
    with open("C:/Users/royba/Downloads/stage/T3/text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text

def chunk_pdf_text(pdf_text, chunk_size=1000, overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(pdf_text)
    return [{"chunk_id": f"chunk_{i+1}", "content": c} for i, c in enumerate(chunks)]



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def store_chunks_in_weaviate(chunks):
    collection = client.collections.get("PDFChunk")

    for chunk in chunks:
            vector = embeddings.embed_query(chunk["content"])
            collection.data.insert(
            properties={
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"]
            },
            vector=vector
            )

def retrieve_relevant_chunks(query: str, top_k=5):
    vector = embeddings.embed_query(query)
    collection = client.collections.get("PDFChunk")
    response = collection.query.near_vector(
        near_vector=vector,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True)
    )
    return response




def generate_answer_from_retrieved_chunks(query: str, results) -> str:
    """
    Generates an answer using the Ollama model based on retrieved chunks from Weaviate.
    """
    retrieved_chunks = [obj.properties["content"] for obj in results.objects]
    context = "\n\n".join(retrieved_chunks)
    prompt_text = f"""
    You are an AI assistant. Check the context below to answer the user's question.

    Context:
    {context}

    Question:
    {query}
    """
    response = ollama.generate(model="deepseek-r1:7b", prompt=prompt_text)
    return response['response']


def save_chunks_to_file(chunks, filename="chunks.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)




@app.route('/rag', methods=['POST'])
def rag_endpoint():
    if not client.is_ready():
        return jsonify({"error": "Weaviate is not ready"}), 503
    
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    results = retrieve_relevant_chunks(query)
    answer = generate_answer_from_retrieved_chunks(query, results)
    print("\nAnswer:\n", answer)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)


    return jsonify({
        "query": query,
        "answer": answer,
    })

if __name__ == "__main__":

    client = weaviate.connect_to_local(
    host=WEAVIATE_URL.replace("http://", "").split(":")[0],
    port=int(WEAVIATE_URL.split(":")[-1]) if ":" in WEAVIATE_URL else 8080
    )
    print("✅ Weaviate ready:", client.is_ready())

    
    existing_classes = client.collections.list_all()
    if "PDFChunk" not in existing_classes:
    
        client.collections.create(
            name="PDFChunk",
            properties=[
                wc.Property(name="chunk_id", data_type=wc.DataType.TEXT),
                wc.Property(name="content", data_type=wc.DataType.TEXT),
            ],
        )
        print("✅ Created PDFChunk class")
    else:
        print("✅ PDFChunk class already exists")



    collection = client.collections.get("PDFChunk")
    aggregation = collection.aggregate.over_all(total_count=True)
    total_objects = aggregation.total_count or 0

    if total_objects == 0:
        pdf_path = "C:/Users/royba/Downloads/stage/data/ISO_IEC_42001_2023en.pdf"
        chunks = chunk_pdf_text(extract_pdf_text(pdf_path))
        store_chunks_in_weaviate(chunks)
        save_chunks_to_file(chunks,"C:/Users/royba/Downloads/stage/T3/chunks.txt")
        print("✅ Chunks stored in Weaviate!")
    else:
        print(f"✅ Found {total_objects} existing chunks in Weaviate, skipping upload.")
    try:
        app.run(debug=True, port=5000)
    finally:
        client.close()
        print("✅ Client connection closed.")


