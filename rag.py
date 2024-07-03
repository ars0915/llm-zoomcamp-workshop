from openai import OpenAI
from elasticsearch import Elasticsearch
from transformers import T5ForConditionalGeneration, T5Tokenizer


es = Elasticsearch("http://localhost:9200")
index_name = "course-questions"


context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()


prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def retrieve_documents(
        query,
        index_name="course-questions",
        max_results=5,
        course="data-engineering-zoomcamp"
    ):
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt



model_name = "google/flan-t5-xl"

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
tokenizer.model_max_length = 2048

def llm(prompt, generate_params=None):
    if generate_params is None:
        generate_params = {}

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=generate_params.get("max_length", 100),
        num_beams=generate_params.get("num_beams", 5),
        do_sample=generate_params.get("do_sample", False),
        temperature=generate_params.get("temperature", 1.0),
        top_k=generate_params.get("top_k", 50),
        top_p=generate_params.get("top_p", 0.95),
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def qa_bot(user_question, course):
    context_docs = retrieve_documents(user_question, course=course)
    prompt = build_prompt(user_question, context_docs)
    answer = llm(prompt)
    return answer