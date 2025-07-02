import os
import json
import random

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain.chains import RetrievalQA
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# 1. 환경 변수 로드
load_dotenv()

# 2. Upstage embedding 모델 정의
chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

# 3. Pinecone 설정
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "cs-interview-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=4096,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_vectorstore = PineconeVectorStore(index=pc.Index(index_name), embedding=embedding_upstage)

pinecone_retriever = pinecone_vectorstore.as_retriever(
    search_type='mmr',  # default : similarity(유사도) / mmr 알고리즘
    search_kwargs={"k": 3}  # 쿼리와 관련된 chunk를 3개 검색하기 (default : 4)
)

app = FastAPI()

with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

@app.get("/random-question")
async def random_question(category: str = Query(default=None)):
    # 1. 카테고리 필터링 (없으면 그냥 랜덤하게)
    filtered = [q for q in qa_data if category is None or q["category"] == category]

    if not filtered:
        return {"error": "해당 카테고리에 질문이 없습니다."}

    # 2. 무작위 질문 선택
    sample = random.choice(filtered)

    # 3. 질문과 해당 질문이 어떤 카테고리, 키워드에 있는지 반환함.
    return {
        "question": sample["question"],
        "category": sample["category"],
        "keywords": sample["keywords"]
    }

# @app.get("/generate-question")
# async def generate_question():
#     system_prompt = """
#     너는 신입 개발자를 위한 CS 면접관이다.
#     아래 조건에 맞는 면접 질문을 한 개 생성해줘:

#     - 주제는 컴퓨터공학 전반 (자료구조, 운영체제, 네트워크, DB, 알고리즘 등)
#     - 난이도는 신입 개발자 수준
#     - 질문은 명확하고 간결하게 1문장으로 생성할 것
#     - 출력은 질문만, 설명은 하지 마
#     """

#     question = chat_upstage.invoke(system_prompt.strip())
#     return {"question": question}

# @app.post("/chat")
# async def chat_endpoint(req: MessageRequest):
#     qa = RetrievalQA.from_chain_type(llm=chat_upstage,
#                                      chain_type="stuff",
#                                      retriever=pinecone_retriever,
#                                      return_source_documents=True)
#     result = qa(req.message)
#     return {"reply": result['result']}

class EvaluateRequest(BaseModel):
    question: str
    answer: str

@app.post("/evaluate-ragas")
async def evaluate_ragas(req: EvaluateRequest):
    question = req.question.strip()
    user_answer = req.answer.strip()

    # 1. ground_truth 찾기
    matched = next((q for q in qa_data if q["question"].strip() == question), None)
    if not matched:
        raise HTTPException(status_code=404, detail="질문에 대한 정답 기준을 찾을 수 없습니다.")

    ground_truth = matched["answer"]

    # 2. RAG: context 검색
    retrieved_docs = pinecone_retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    # 3. ragas 평가용 Dataset 구성
    dataset = Dataset.from_list([{
        "question": question,
        "answer": user_answer,
        "contexts": contexts,
        "ground_truth": ground_truth
    }])

    # 4. 평가 수행
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    # 5. 결과 반환
    scores = results.to_pandas().iloc[0].to_dict()
    return {
        "question": question,
        "user_answer": user_answer,
        "ground_truth": ground_truth,
        "ragas_scores": scores
    }


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
