import os
import json
import random

from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from langchain.chains import RetrievalQA
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel

load_dotenv()

chat_upstage = ChatUpstage()
embedding_upstage = UpstageEmbeddings(model="embedding-query")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    question = req.question.strip()
    user_answer = req.answer.strip()

    matched = next((q for q in qa_data if q["question"].strip() == question), None)
    if not matched:
        raise HTTPException(status_code=404, detail="질문에 대한 정답 기준을 찾을 수 없습니다.")

    ground_truth = matched["answer"]

    retrieved_docs = pinecone_retriever.get_relevant_documents(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    
    if ground_truth not in contexts:
        contexts.append(ground_truth) 
  
    prompt = build_feedback_prompt(
        question=question,
        user_answer=user_answer,
        context_list=contexts,
        ground_truth=ground_truth
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("LLM 응답이 비어 있습니다.")
    feedback = content.strip()
    
    return {"reply": feedback}

def build_system_prompt():
    return (
        "당신은 시니어 개발자이며, 후배 개발자의 기술 면접을 도와주는 면접 코치입니다. "
        "친절하고 논리적으로 피드백을 주고, 너무 공격적이지 않게 개선점을 알려주세요."
    )

def build_feedback_prompt(question, user_answer, context_list, ground_truth):
    context_text = "\n".join(context_list)

    prompt = f"""
당신은 개발자 면접 코치입니다. 아래의 질문과 사용자의 답변을 평가하여, 실전 면접처럼 자연스럽고 논리적인 피드백을 한국어로 작성해주세요.  
다음 3가지 항목을 중심으로 구성해 주세요:

1. 사용자가 어떤 핵심 키워드를 잘 언급했는지  
2. 어떤 핵심 개념이 누락되었는지 (문맥 또는 모범답안 기준)  
3. 면접 스타일(설명 순서, 표현 방식 등)에서 어떻게 보완하면 좋을지  

---

  **중요 지침** (절대 위반하지 마세요):

- 사용자의 답변이 너무 짧거나 단어만 나열되어 있으면, **이해도를 판단하기 어렵다고 명확하게 지적**해야 합니다.  
  → "핵심 용어는 언급되었지만 설명이 없어 면접에서 낮은 평가를 받을 수 있습니다." 라고 평가해 주세요.

- 사용자가 말하지 않은 내용을 **추론하거나 과하게 긍정적으로 말하지 마세요.**  
  → 반드시 실제 답변 내용에 **한정해서** 평가해 주세요.

- 피드백은 지적과 격려의 균형을 맞춰서, 성장할 수 있도록 **구체적이고 따뜻하게** 작성해주세요.

---

📌 질문:  
{question}

✍️ 사용자 답변:  
{user_answer}

반드시 사용자 답변으로만 판단해서 답변하세요. 반드시!!!!!

📚 검색된 문맥 정보:  
{context_text}

✅ 모범답변:  
{ground_truth}
"""
    return prompt



@app.get("/health")
@app.get("/")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
