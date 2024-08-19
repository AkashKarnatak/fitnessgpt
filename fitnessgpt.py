from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import glob, os
from functools import reduce
import os
import openai
import streamlit as st

openai_api_key = os.getenv('OPENAI_KEY')

if not openai_api_key:
    raise Exception('Provide a valid OpenAI key')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def extract_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = reduce(
        lambda text, page: text + page.extract_text(),
        pdf_reader.pages,
        ''
    )
    return text
    
def exists(file_loc, collection):
    n_items = len(collection.get(where={"file": file_loc}, include=['metadatas'])['ids'])
    return n_items > 0

def store_chunks(chunks, file_loc, collection):
    start_idx = collection.count()
    collection.upsert(
        documents=chunks,
        metadatas=[{"file": file_loc}] * len(chunks),
        ids=[f'id{i}' for i in range(start_idx, start_idx + len(chunks))]
    )
    
class BertE5SmallEmbeddingFunction(EmbeddingFunction):
    @torch.no_grad()
    def __call__(self, texts: Documents) -> Embeddings:
        # Tokenize the input texts
        batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        # normalize embeddings to unit vector
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.tolist()

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len
)

chroma_client = chromadb.PersistentClient(os.path.join(os.getcwd(), './protein_db/'))
e5_ef = BertE5SmallEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="protein_kb",
    embedding_function=e5_ef,
    metadata={"hnsw:space": "l2"} # or cosine
)

openai.api_key = openai_api_key

def answer(question, model='gpt-3.5-turbo-16k'):
    n_results = 9
    if model == 'gpt-4':
        n_results = 20
    contextText = reduce(lambda t, d: t + '\n\n' + d, collection.query(query_texts=question, n_results=n_results)['documents'][0], '')
    messages = [
      {"role": m["role"], "content": m["content"]}
      for m in st.session_state.messages
    ]
    messages = messages + [
      {"role": "user", "content": f"Here is the research study:\n{contextText}"},
      {"role": "user", "content": f'''
Answer my next question using only the above research study. You must also follow the below rules when answering:

- Do not make up answers that are not provided in the documentation.
- If you are unsure and the answer is not explicitly written in the research study, say "Sorry, I don't know how to help with that."
- Prefer splitting your response into multiple paragraphs.
    '''},
      {"role": "user", "content": f"Here is my question:\n{question}"}
    ]
    streaming_response = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      stream=True,
    )
    return streaming_response
    
## StreamLIT
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "messages" not in st.session_state:
    systemMsg = '''You are a very enthusiastic fitness coach who loves to help people achieve their fitness goals! Given the following information from scientific research articles under "research study", answer the question using only that information. If you are unsure and the answer is not explicitly written in the research study, say "Sorry, I don't know how to help with that."'''
    st.session_state.messages = [{"role": "system", "content": systemMsg}]
            
st.title("FitnessGPT")

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        streaming_response = answer(prompt)
        for response in streaming_response:
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
