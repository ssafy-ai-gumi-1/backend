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
    chunk_size=600,
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

# 5. 폴더별 문서 파싱 및 업로드
def process_and_upload_by_folder(base_dir: str):
    category_docs_map = {}

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                mid_folder = os.path.basename(os.path.dirname(full_path))
                print(f"📄 처리 중: {full_path}")

                category, keywords = extract_metadata_from_md(full_path)
                loader = UnstructuredMarkdownLoader(full_path)
                docs = loader.load()
                split_docs = text_splitter.split_documents(docs)

                for doc in split_docs:
                    doc.metadata["category"] = category
                    doc.metadata["keyword"] = keywords

                if mid_folder not in category_docs_map:
                    category_docs_map[mid_folder] = []
                category_docs_map[mid_folder].extend(split_docs)

    # 폴더별로 Pinecone 인덱스 생성 및 업로드
    for folder_name, docs in category_docs_map.items():
        index_name = f"cs-{folder_name.lower()}-index"

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=4096,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        print(f"📤 업로드 중: {folder_name} → {index_name} ({len(docs)} chunks)")
        PineconeVectorStore.from_documents(docs, embedding_upstage, index_name=index_name)
        print(f"✅ 완료: {folder_name} 업로드")

# 6. 실행
print("🚀 Markdown 파싱 및 카테고리별 업로드 시작...")
process_and_upload_by_folder("DB/")
print("✅ 전체 완료")
