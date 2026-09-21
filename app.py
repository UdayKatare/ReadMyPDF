import streamlit as st
import PyPDF2
import requests
import os
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
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

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. "
            "Please add it under Render → Environment Variables."
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
        "include_reasoning": False
    }

    try:

        # ----------------------------------------------------
        # CALL GROQ
        # ----------------------------------------------------

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        st.caption(
            f"Groq model: `{MODEL}` | "
            f"HTTP status: `{response.status_code}`"
        )

        # ----------------------------------------------------
        # HANDLE HTTP ERROR
        # ----------------------------------------------------

        if not response.ok:

            st.error(
                f"Groq API returned HTTP {response.status_code}"
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

            st.error(
                "Groq returned a response that is not valid JSON."
            )

            st.code(response.text)

            return ""

        # ----------------------------------------------------
        # DEBUG RESPONSE STRUCTURE
        # ----------------------------------------------------

        if "choices" not in result:

            st.error(
                "Groq response does not contain a 'choices' field."
            )

            st.write("Full Groq response:")

            st.json(result)

            return ""

        # ----------------------------------------------------
        # CHECK CHOICES
        # ----------------------------------------------------

        if not result["choices"]:

            st.error(
                "Groq returned an empty choices array."
            )

            st.json(result)

            return ""

        # ----------------------------------------------------
        # EXTRACT MESSAGE
        # ----------------------------------------------------

        message = result["choices"][0].get(
            "message",
            {}
        )

        content = message.get(
            "content"
        )

        # ----------------------------------------------------
        # CHECK CONTENT
        # ----------------------------------------------------

        if not content:

            st.error(
                "Groq returned a message without content."
            )

            st.json(result)

            return ""

        return content

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        st.error(
            "The request to Groq timed out after 60 seconds."
        )

        return ""

    # --------------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError as e:

        st.error(
            f"Could not connect to Groq: {e}"
        )

        return ""

    # --------------------------------------------------------
    # REQUEST ERROR
    # --------------------------------------------------------

    except requests.exceptions.RequestException as e:

        st.error(
            f"Groq request failed: {e}"
        )

        return ""

    # --------------------------------------------------------
    # UNKNOWN ERROR
    # --------------------------------------------------------

    except Exception as e:

        st.error(
            f"Unexpected error: {repr(e)}"
        )

        return ""


# ============================================================
# SUMMARIZER
# ============================================================

def summarize_text(document_text):

    # Current application behavior:
    # Send first 6000 characters.
    #
    # We can implement intelligent chunking later
    # for large documents.

    document_content = document_text[:6000]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional document summarizer. "
                "Analyze the provided document and create a "
                "clear, concise business-friendly summary.\n\n"

                "Return exactly 5 bullet points.\n\n"

                "Focus on:\n"
                "- Important facts\n"
                "- Key decisions\n"
                "- Important numbers\n"
                "- Dates\n"
                "- Requirements\n"
                "- Business implications\n\n"

                "Do not invent information. "
                "Use only information contained in the document."
            )
        },
        {
            "role": "user",
            "content": (
                "Summarize the following document:\n\n"
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
                "You are a precise document assistant.\n\n"

                "Answer the user's question using ONLY "
                "information contained in the provided document.\n\n"

                "Rules:\n"
                "1. Do not invent information.\n"
                "2. Do not use outside knowledge.\n"
                "3. If the answer cannot be found in the document, "
                "reply exactly: I don't know.\n"
                "4. Keep the answer clear and concise."
            )
        },
        {
            "role": "user",
            "content": (
                "DOCUMENT:\n"
                "--------------------------------\n"
                f"{document_content}\n"
                "--------------------------------\n\n"

                "QUESTION:\n"
                f"{question}"
            )
        }
    ]

    return call_groq(messages)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_text(uploaded_file):

    try:

        reader = PyPDF2.PdfReader(
            uploaded_file
        )

        text = ""

        for page_number, page in enumerate(
            reader.pages
        ):

            try:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            except Exception as e:

                st.warning(
                    f"Could not extract page "
                    f"{page_number + 1}: {e}"
                )

        return text

    except Exception as e:

        st.error(
            f"Error reading the PDF: {e}"
        )

        return ""


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📄 PDF Document Assistant")

st.write(
    """
    Upload a PDF document, then choose whether you want
    a concise summary or ask questions about the document.
    """
)


# ============================================================
# API STATUS
# ============================================================

if not api_key:

    st.warning(
        "⚠️ GROQ_API_KEY is not configured. "
        "Add it under Render → Environment Variables."
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file:

    # --------------------------------------------------------
    # EXTRACT PDF
    # --------------------------------------------------------

    with st.spinner(
        "Reading PDF..."
    ):

        document_text = extract_text(
            uploaded_file
        )

    # --------------------------------------------------------
    # CHECK PDF
    # --------------------------------------------------------

    if not document_text.strip():

        st.warning(
            "This PDF seems empty or could not be parsed."
        )

    else:

        # ----------------------------------------------------
        # STORE DOCUMENT
        # ----------------------------------------------------

        st.session_state[
            "document_text"
        ] = document_text

        # ----------------------------------------------------
        # DOCUMENT INFO
        # ----------------------------------------------------

        word_count = len(
            document_text.split()
        )

        character_count = len(
            document_text
        )

        st.info(
            f"📄 PDF loaded successfully — "
            f"{word_count:,} words | "
            f"{character_count:,} characters"
        )

        # ----------------------------------------------------
        # MODE SELECTION
        # ----------------------------------------------------

        mode = st.radio(
            "Choose your mode:",
            [
                "Summarize",
                "Ask Questions"
            ]
        )

        # ====================================================
        # SUMMARIZE
        # ====================================================

        if mode == "Summarize":

            st.subheader(
                "Summary"
            )

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

                if summary:

                    st.success(
                        "Summary completed."
                    )

                    st.markdown(
                        summary
                    )

                    st.download_button(
                        "Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain"
                    )

        # ====================================================
        # ASK QUESTIONS
        # ====================================================

        elif mode == "Ask Questions":

            st.subheader(
                "Ask about this Document"
            )

            question = st.text_input(
                "Type your question here:",
                placeholder=(
                    "Example: What are the main conclusions?"
                )
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

                        st.markdown(
                            answer
                        )
