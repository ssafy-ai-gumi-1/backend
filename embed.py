import os
import json

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로드
load_dotenv()

# 2. Upstage embedding 모델 정의
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# 3. Pinecone 설정
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "qa-data-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

print("start")

# 4. JSON 데이터 로딩 및 파일 구성
json_path = "qa_data.json"
with open(json_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

documents = []
for data in json_data:
    content = f"Q: {data['question']}\nA: {data['answer']}"
    metadata = {
        "question": data["question"],
        "answer": data["answer"],
        "category": data["category"],
        "keywords": data["keywords"],
    }
    documents.append(Document(page_content=content, metadata=metadata))

# 5. Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(documents)

# 6. 벡터 임베딩 + Pinecone 업로드
PineconeVectorStore.from_documents(
    splits, embedding_upstage, index_name=index_name
)

print("JSON 파싱 및 업로드 완료")
