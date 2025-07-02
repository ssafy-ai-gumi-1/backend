import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain_upstage import UpstageEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 1. 환경 변수 로드
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

# 2. 임베딩 모델
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# 3. 텍스트 분할기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# 4. 메타데이터 파싱 함수
def extract_metadata_from_md(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    category = None
    keywords = []

    for line in lines:
        line = line.strip()
        if line.startswith("카테고리:"):
            category = line.replace("카테고리:", "").strip()
        elif line.startswith("키워드:"):
            keyword_raw = line.replace("키워드:", "").strip()
            keyword_raw = keyword_raw.strip("[]")
            keywords = [kw.strip() for kw in keyword_raw.split(",") if kw.strip()]
        if category and keywords:
            break

    return category, keywords

# 5. 전체 md 파일을 하나의 인덱스로 업로드
def upload_all_to_single_index(base_dir: str, index_name: str):
    all_docs = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)
                print(f"📄 처리 중: {path}")

                category, keywords = extract_metadata_from_md(path)
                loader = UnstructuredMarkdownLoader(path)
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)

                for doc in split_docs:
                    doc.metadata["category"] = category
                    doc.metadata["keyword"] = keywords

                all_docs.extend(split_docs)

    # 인덱스 생성
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=4096,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    print(f"📦 총 {len(all_docs)} chunks 업로드 중...")

    # 업로드 (batch로 나눔)
    BATCH_SIZE = 100
    for i in range(0, len(all_docs), BATCH_SIZE):
        batch = all_docs[i:i + BATCH_SIZE]
        PineconeVectorStore.from_documents(batch, embedding_upstage, index_name=index_name)
        print(f"✅ 업로드: {i} ~ {i + len(batch) - 1}")

# 6. 실행
print("🚀 모든 Markdown 문서를 단일 인덱스에 업로드 시작...")
upload_all_to_single_index("DB/", "cs-interview-index")
print("✅ 전체 완료")
