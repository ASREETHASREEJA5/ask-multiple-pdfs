import streamlit as st
st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from langchain.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download("punkt")

# ================= NLI LOADING ================= #

@st.cache_resource
def load_nli_model():
    model_name = "roberta-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

nli_tokenizer, nli_model = load_nli_model()


def nli_score(premise, hypothesis):
    inputs = nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][2].item()  # entailment score


# ================= PDF PROCESSING ================= #

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# ================= CONVERSATION CHAIN ================= #

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key="answer" 
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        output_key="answer"   # IMPORTANT
    )


# ================= USER INPUT HANDLING ================= #

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    answer_text = response['answer']
    retrieved_docs = response['source_documents']

    # Combine retrieved text as evidence
    evidence_text = " ".join([doc.page_content for doc in retrieved_docs])

    # Split answer into claims
    claims = nltk.sent_tokenize(answer_text)

    verified_claims = []
    total_claims = len(claims)

    for claim in claims:
        max_score = 0

        for doc in retrieved_docs:
            score = nli_score(doc.page_content, claim)
            max_score = max(max_score, score)

        print("CLAIM:", claim)
        print("MAX SCORE:", max_score)
        score = nli_score(evidence_text, claim)
        if score >= 0.4:
            verified_claims.append(claim)

    # Final verified answer
    verified_answer = " ".join(verified_claims)

    if not verified_answer.strip():
        verified_answer = answer_text

    # Confidence score
    confidence = (len(verified_claims) / total_claims) * 100 if total_claims > 0 else 0

    st.session_state.chat_history = response['chat_history']

    # Display chat
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", verified_answer), unsafe_allow_html=True)

    st.markdown(f"**Verification Confidence:** {confidence:.2f}%")



# ================= MAIN APP ================= #

def main():
    load_dotenv()

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed successfully!")


if __name__ == '__main__':
    main()
