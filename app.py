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

load_dotenv()

chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

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
    search_type='mmr',
    search_kwargs={"k": 3}
)

app = FastAPI()

with open("qa_data.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

@app.get("/random-question")
async def random_question(category: str = Query(default=None)):
    filtered = [q for q in qa_data if category is None or q["category"] == category]

    if not filtered:
        return {"error": "해당 카테고리에 질문이 없습니다."}

    sample = random.choice(filtered)

    return {
        "question": sample["question"],
        "category": sample["category"],
        "keywords": sample["keywords"]
    }


class EvaluateRequest(BaseModel):
    question: str
    answer: str

@app.post("/evaluate-ragas")
async def evaluate_ragas(req: EvaluateRequest):
    question = req.question.strip()
    user_answer = req.answer.strip()

    matched = next((q for q in qa_data if q["question"].strip() == question), None)
    if not matched:
        raise HTTPException(status_code=404, detail="질문에 대한 정답 기준을 찾을 수 없습니다.")

    ground_truth = matched["answer"]

    retrieved_docs = pinecone_retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in retrieved_docs]

    dataset = Dataset.from_list([{
        "question": question,
        "answer": user_answer,
        "contexts": contexts,
        "ground_truth": ground_truth
    }])

    # results = evaluate(
    #     dataset=dataset,
    #     metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    # )
    
    # answer_relevancy만 평가
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy]
    )

    # scores = results.to_pandas().iloc[0].to_dict()
    # return {
    #     "question": question,
    #     "user_answer": user_answer,
    #     "ground_truth": ground_truth,
    #     "ragas_scores": scores
    # }
    # answer_relevancy 점수만 추출
    score = results.to_pandas().iloc[0]["answer_relevancy"]

    return {
        "question": question,
        "user_answer": user_answer,
        "ground_truth": ground_truth,
        "answer_relevancy": score
    }


@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
