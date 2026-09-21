import streamlit as st
import PyPDF2
import requests
import os
from dotenv import load_dotenv

# ============================================================
# ENVIRONMENT / CONFIG
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF Document Assistant",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    .stButton > button,
    .stDownloadButton > button {
        background-color: #2a6cd6;
        color: white;
        border-radius: 4px;
        border: none;
        padding: 0.4em 1em;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #1f5bb5;
        color: white;
    }

    .stRadio > div {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 0.5em;
    }

    .stTextInput > div > input {
        border-radius: 4px;
        border: 1px solid #ccc;
        padding: 0.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GROQ API
# ============================================================

def call_groq(messages):
    """
    Send a request to Groq and safely process the response.
    """

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY to your Render environment variables."
        )
        return ""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "reasoning_effort": "low"
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if not response.ok:
            st.error(
                f"Groq API Error: HTTP {response.status_code}"
            )

            try:
                error_data = response.json()
                st.json(error_data)
            except Exception:
                st.code(response.text)

            return ""

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        try:
            result = response.json()
        except ValueError:
            st.error("Groq returned an invalid JSON response.")
            st.code(response.text)
            return ""

        # ----------------------------------------------------
        # CHECK RESPONSE STRUCTURE
        # ----------------------------------------------------

        if "choices" not in result:
            st.error("Groq returned an unexpected response.")
            st.json(result)
            return ""

        if not result["choices"]:
            st.error("Groq returned an empty response.")
            st.json(result)
            return ""

        message = result["choices"][0].get("message", {})

        content = message.get("content")

        if not content:
            st.error(
                "Groq returned a response without any content."
            )
            st.json(result)
            return ""

        return content

    except requests.exceptions.Timeout:
        st.error(
            "The request to Groq timed out. Please try again."
        )
        return ""

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to Groq. Please try again."
        )
        return ""

    except requests.exceptions.RequestException as e:
        st.error(
            f"Groq request failed: {e}"
        )
        return ""

    except Exception as e:
        st.error(
            f"Unexpected error while communicating with Groq: {e}"
        )
        return ""


# ============================================================
# SUMMARIZER
# ============================================================

def summarize_text(document_text):

    # Keep the current behavior of your application.
    # This can be upgraded later to intelligent chunking.
    document_content = document_text[:6000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional document summarizer. "
                "Analyze the provided document and produce a concise, "
                "clear and business-friendly summary.\n\n"

                "Return exactly 5 bullet points.\n\n"

                "Focus on the most important:\n"
                "- Facts\n"
                "- Decisions\n"
                "- Numbers\n"
                "- Dates\n"
                "- Requirements\n"
                "- Business implications\n\n"

                "Do not invent information. "
                "Only use information contained in the document."
            )
        },
        {
            "role": "user",
            "content": (
                "Summarize this document:\n\n"
                + document_content
            )
        }
    ]

    return call_groq(messages)


# ============================================================
# Q&A
# ============================================================

def ask_about_document(document_text, question):

    document_content = document_text[:6000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document assistant. "

                "Answer the user's question using ONLY "
                "information contained in the provided document.\n\n"

                "Rules:\n"
                "1. Do not invent information.\n"
                "2. Do not use outside knowledge.\n"
                "3. If the answer cannot be found in the document, "
                "reply: I don't know.\n"
                "4. Keep the answer clear and concise."
            )
        },
        {
            "role": "user",
            "content": (
                "DOCUMENT:\n"
                "-------------------------\n"
                f"{document_content}\n"
                "-------------------------\n\n"

                f"QUESTION:\n{question}"
            )
        }
    ]

    return call_groq(messages)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file):

    try:
        reader = PyPDF2.PdfReader(uploaded_file)

        text = ""

        for page_number, page in enumerate(reader.pages):

            try:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            except Exception as e:
                st.warning(
                    f"Could not extract page {page_number + 1}: {e}"
                )

        return text

    except Exception as e:
        st.error(
            f"Error reading the PDF: {e}"
        )
        return ""


# ============================================================
# APPLICATION
# ============================================================

st.title("📄 PDF Document Assistant")

st.write(
    """
    Upload a PDF document, then choose to either get a
    concise summary or ask questions about its contents.
    """
)


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)


if uploaded_file:

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    with st.spinner("Reading PDF..."):
        document_text = extract_text(uploaded_file)

    # --------------------------------------------------------
    # EMPTY PDF
    # --------------------------------------------------------

    if not document_text.strip():

        st.warning(
            "This PDF seems empty or could not be parsed."
        )

    else:

        # Store document
        st.session_state["document_text"] = document_text

        # Basic document information
        word_count = len(document_text.split())

        st.info(
            f"📄 PDF loaded successfully — "
            f"{word_count:,} words extracted."
        )

        # ----------------------------------------------------
        # MODE
        # ----------------------------------------------------

        mode = st.radio(
            "Choose your mode:",
            [
                "Summarize",
                "Ask Questions"
            ]
        )

        # ====================================================
        # SUMMARIZE MODE
        # ====================================================

        if mode == "Summarize":

            st.subheader("Summary")

            if st.button(
                "Generate Summary",
                key="summarize_button"
            ):

                with st.spinner(
                    "Creating summary, please wait..."
                ):

                    summary = summarize_text(
                        document_text
                    )

                # Only show success if we actually received
                # a response from Groq.

                if summary:

                    st.success(
                        "Summary completed."
                    )

                    st.markdown(summary)

                    st.download_button(
                        "Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )


        # ====================================================
        # Q&A MODE
        # ====================================================

        elif mode == "Ask Questions":

            st.subheader(
                "Ask about this Document"
            )

            question = st.text_input(
                "Type your question here:",
                placeholder="Example: What are the main conclusions?"
            )

            if st.button(
                "Get Answer",
                key="question_button"
            ):

                if not question.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Getting answer..."
                    ):

                        answer = ask_about_document(
                            document_text,
                            question
                        )

                    if answer:

                        st.success(
                            "Answer"
                        )

                        st.markdown(answer)
