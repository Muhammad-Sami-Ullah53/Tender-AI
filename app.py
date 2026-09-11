import streamlit as st
import fitz
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_pages(pdf_file):
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text")
        text = clean_text(text)

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()
    return pages


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    return text.strip()


def create_chunks(pages, chunk_size=1200, overlap=200):
    chunks = []

    for page_data in pages:
        text = page_data["text"]
        page_number = page_data["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


def retrieve_evidence(query, chunks, top_k=5):
    if not chunks:
        return []

    documents = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(documents + [query])

    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    ranked_indexes = scores.argsort()[::-1][:top_k]

    evidence = []

    for index in ranked_indexes:
        if scores[index] <= 0:
            continue

        evidence.append({
            "text": chunks[index]["text"],
            "page": chunks[index]["page"],
            "document": chunks[index].get("document", ""),
            "score": round(float(scores[index]), 4)
        })

    return evidence


def add_document_name(chunks, document_name):
    for chunk in chunks:
        chunk["document"] = document_name

    return chunks


def process_document(pdf_file, document_name):
    pages = extract_pages(pdf_file)
    chunks = create_chunks(pages)
    chunks = add_document_name(chunks, document_name)

    return {
        "document_name": document_name,
        "pages": pages,
        "chunks": chunks
    }


def build_output(tender_data, company_data):
    return {
        "tender_document": tender_data,
        "company_document": company_data
    }


st.set_page_config(
    page_title="TenderWise AI - Sami PDF Processor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 TenderWise AI")
st.subheader("PDF Processing + RAG Evidence")

st.write(
    "Upload a Tender PDF and Company Profile PDF. "
    "The system extracts page-wise text, creates chunks, "
    "and retrieves evidence with document and page references."
)

col1, col2 = st.columns(2)

with col1:
    tender_file = st.file_uploader(
        "Upload Tender PDF",
        type=["pdf"],
        key="tender"
    )

with col2:
    company_file = st.file_uploader(
        "Upload Company Profile PDF",
        type=["pdf"],
        key="company"
    )


if tender_file and company_file:

    if st.button("🚀 Process Documents", use_container_width=True):

        with st.spinner("Processing PDFs..."):

            tender_data = process_document(
                tender_file,
                tender_file.name
            )

            company_data = process_document(
                company_file,
                company_file.name
            )

            output = build_output(
                tender_data,
                company_data
            )

            st.session_state["output"] = output

        st.success("Documents processed successfully!")


if "output" in st.session_state:

    output = st.session_state["output"]

    tender_data = output["tender_document"]
    company_data = output["company_document"]

    st.divider()

    st.header("📊 Processing Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Tender Pages",
        len(tender_data["pages"])
    )

    col2.metric(
        "Tender Chunks",
        len(tender_data["chunks"])
    )

    col3.metric(
        "Company Pages",
        len(company_data["pages"])
    )

    col4.metric(
        "Company Chunks",
        len(company_data["chunks"])
    )

    st.divider()

    st.header("🔍 Evidence Retrieval")

    query = st.text_input(
        "Enter a requirement or question",
        placeholder="Example: minimum 5 years experience"
    )

    document_choice = st.selectbox(
        "Search document",
        [
            "Tender",
            "Company Profile",
            "Both Documents"
        ]
    )

    top_k = st.slider(
        "Number of evidence results",
        min_value=1,
        max_value=10,
        value=5
    )

    if st.button("🔎 Retrieve Evidence"):

        if document_choice == "Tender":
            chunks = tender_data["chunks"]

        elif document_choice == "Company Profile":
            chunks = company_data["chunks"]

        else:
            chunks = (
                tender_data["chunks"] +
                company_data["chunks"]
            )

        evidence = retrieve_evidence(
            query,
            chunks,
            top_k
        )

        if evidence:
            st.subheader("Retrieved Evidence")

            for i, item in enumerate(evidence, start=1):

                st.markdown(
                    f"### Evidence {i}"
                )

                st.write(item["text"])

                st.caption(
                    f"📄 Document: {item['document']} | "
                    f"Page: {item['page']} | "
                    f"Relevance: {item['score']}"
                )

                st.divider()

            st.session_state["evidence"] = evidence

        else:
            st.warning(
                "No relevant evidence found."
            )

    st.header("📥 Sami Module Output")

    json_data = json.dumps(
        output,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="Download Processed JSON",
        data=json_data,
        file_name="tenderwise_processed_documents.json",
        mime="application/json"
    )