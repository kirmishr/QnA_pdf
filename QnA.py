import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time
import pickle

def create_embedding(query):
    """
    Creates embeddings for the query.

    Parameters:
    query (str): The question to be asked.

    Returns:
    numpy.ndarray: The embedding of the query.
    """
    model_name = 'paraphrase-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query)
    return query_embedding

def retrieve_documents(query, embedded_docs, splits, top_n=3):
    """
    Embeds the query and, based on cosine similarity, selects top n documents related to it.

    Parameters:
    query (str): The query to embed.
    embedded_docs (numpy.ndarray): Precomputed document embeddings.
    splits (list): List of text splits from the documents.
    top_n (int): Number of top documents to retrieve.

    Returns:
    list: The top n most similar documents.
    """
    start_time = time.time()
    query_embedding = create_embedding(query)

    # Calculate cosine similarity between query and all document embeddings
    similarities = cosine_similarity([query_embedding], embedded_docs)[0]

    # Extract indices of n most similar documents
    similarity_with_index = list(enumerate(similarities))
    similarity_with_index.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarity_with_index[:top_n]]
    top_documents = [splits[i] for i in top_indices]

    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write("Time for retrieving data:", time_elapsed)
    return top_documents

def get_context_and_answer(splits, query, embedded_docs):
    """
    Takes query, gets the context from retrieval, and then gets the answer from that context.

    Parameters:
    splits (list): List of text splits from the documents.
    query (str): The user's question.
    embedded_docs (numpy.ndarray): Precomputed document embeddings.

    Returns:
    tuple: The context and the generated answer.
    """
    start_time = time.time()
    # Retrieve documents
    relevant_docs = retrieve_documents(query, embedded_docs, splits)
    context = ' '.join(relevant_docs)

    model_name = 'deepset/roberta-base-squad2'
    # Model for context-based answering
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(query, context, return_tensors="pt")
    output = model(**inputs)

    # Selecting best answer
    ans_start = torch.argmax(output.start_logits)
    ans_end = torch.argmax(output.end_logits)
    ans_token = inputs.input_ids[0, ans_start: ans_end + 1]

    # Decoding the answer
    answer = tokenizer.decode(ans_token)

    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write("Time for answering question based on retrieved data:", time_elapsed)
    return context, answer

def get_context_and_answer_long(splits, query, embedded_docs):
    """
    Takes query, gets the context from retrieval, and then gets the answer from that context.

    Parameters:
    splits (list): List of text splits from the documents.
    query (str): The user's question.
    embedded_docs (numpy.ndarray): Precomputed document embeddings.

    Returns:
    tuple: The context and the generated answer.
    """
    start_time = time.time()
    # Retrieve documents
    relevant_docs = retrieve_documents(query, embedded_docs, splits)
    context = ' '.join(relevant_docs)
    
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    result = qa_pipeline(question=query, context=context)
    answer = result['answer']
    
    end_time = time.time()
    time_elapsed = end_time - start_time
    st.write("Time for answering question based on retrieved data:", time_elapsed)
    return context, answer

# Streamlit UI
st.title("PDF Question Answering System")

# Load embeddings from file
with open('embeddings.pkl', 'rb') as f:
    embedded_data = pickle.load(f)

# Load splits from file
with open('splits.pkl', 'rb') as f:
    splits = pickle.load(f)

st.write("Embeddings loaded from the backend.")

question = st.text_input("Ask a question about the PDF (few words answer type)")

if question:
    context, answer = get_context_and_answer(splits, question, embedded_data)
    st.write("Context:", context)
    st.write("Answer:", answer)

long_question = st.text_input("Ask a question about the PDF (long answer type)")

if long_question:
    context, answer = get_context_and_answer_long(splits, long_question, embedded_data)
    st.write("Context:", context)
    st.write("Answer:", answer)
    
# slow_question = st.text_input("Ask a question about the PDF(Slow but Accurate)")

# if slow_question:
#     context, answer = get_context_and_answer_slow(splits, slow_question, embedded_data)
#     st.write("Context:", context)
#     st.write("Answer:", answer)




#tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")           # openai-community/gpt2-xl  openai-community/gpt2
#model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")


## SLOWER MODEL (Can use gpt2 instead of gpt2-xl)

# def get_context_and_answer_slow(splits, query, embedded_docs):
#     """ Takes query gets the context from retreival and then get answer from that context to the provided question"""
#     start_time  = time.time()
#     # retreiving docs
#     relevant_docs = retrieve_documents(query, embedded_docs, splits)
#     context = ' '.join(relevant_docs)
    
#     tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")           # openai-community/gpt2-xl  openai-community/gpt2
#     model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-xl")

#     if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
#     prompt = f"Answer the following question based on the context provided. Context: {context}. Question: {query}"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
#     gen_tokens = model.generate(
#         input_ids,
#         do_sample= False,
#         no_repeat_ngram_size=2,
#         max_length=250,
#         pad_token_id=tokenizer.pad_token_id
#     )
#     # gen_tokens = model.generate(**input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200)
#     gen_text = tokenizer.batch_decode(gen_tokens)[0]
#     answer = gen_text[len(prompt):]
    
    
#     end_time  = time.time()
#     time_elapsed = end_time - start_time
#     print("Time for answering question based on retrieved data:",time_elapsed)
#     print()
#     print("Context:",context)
#     print()
#     print("Answer:", answer)