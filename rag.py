from langchain import PromptTemplate
from langchain.llms import CTransformers, Ollama
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
import os
import json

# In summary, with these lines of code, you've set up a FastAPI application, configured it to use Jinja2 for rendering HTML templates, and mounted a directory to serve static files like CSS, JavaScript, and images.
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# llama 2 based model fine tuned on medical terminologies on PubMED
local_llm = "meditron-7b.Q4_K_M.gguf"
# local_llm = "BioMistral-7B.Q4_K_M"

config = {
'max_new_tokens': 1024,
'context_length': 2048,
'repetition_penalty': 1.1,
'temperature': 0.1,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'threads': int(os.cpu_count() / 2)
}

# The **config syntax is used to unpack the config dictionary and pass its contents as keyword arguments to the CTransformers constructor. This allows for a flexible way to configure the language model without explicitly specifying each parameter as a separate argument.

llm = CTransformers(
    model=local_llm,
    model_type="llama",
    lib="avx2",
    **config
)


# llm = Ollama(model="llama3", **config)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333/dashboard"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_database")

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = db.as_retriever(search_kwargs={"k":1})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_response")
async def get_response(query: str = Form(...)):
    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    
    response = qa(query)
    print(response)
    answer = response['result']
    source_document = response['source_documents'][0].page_content
    doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
    res = Response(response_data)
    return res