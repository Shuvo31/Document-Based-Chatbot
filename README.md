# Document-Based Chatbot 🤖

This is a **Document-Based Chatbot** built using **Streamlit**, **LangChain**, and **OpenAI**. It allows users to upload documents (PDF, DOCX, CSV, XLSX), process them, and ask questions about the content. The chatbot uses **FAISS** for document retrieval and **GPT-4** (via OpenAI) for generating responses.

---

## Features

- **Document Upload**: Upload up to 5 documents in PDF, DOCX, CSV, or XLSX format.
- **Document Processing**: Automatically processes uploaded documents and creates embeddings using **OpenAI Embeddings**.
- **Chat Interface**: Ask questions about the uploaded documents and get responses in real-time.
- **Chat History**: View the conversation history between the user and the chatbot.
- **PDF Preview**: Preview uploaded PDF files in the sidebar.

---

## Prerequisites

Before running the application, ensure you have the following:

1. **Python 3.12 or higher**.
2. An **OpenAI API key**. You can get one from [OpenAI](https://platform.openai.com/).
3. A `.env` file in the root directory with your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KavinRajagopal/GMI-POC.git
   cd GMI-POC
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Upload Documents**:
   - Use the sidebar to upload up to 5 documents in PDF, DOCX, CSV, or XLSX format.
   - Once uploaded, the documents will be processed automatically.

2. **Ask Questions**:
   - Type your question in the input box and click "Send".
   - The chatbot will generate a response based on the content of the uploaded documents.

3. **View Chat History**:
   - The conversation history will be displayed in the main window.
   - You can clear the chat history using the "Clear Chat History" button in the sidebar.

4. **Preview PDFs**:
   - Uploaded PDF files can be previewed in the sidebar.

---

## Code Overview

### Key Components

1. **Document Loaders**:
   - Supported file types: PDF, DOCX, CSV, XLSX.
   - Uses `PyPDFLoader`, `Docx2txtLoader`, `CSVLoader`, and `UnstructuredExcelLoader` from LangChain.

2. **Text Splitting**:
   - Documents are split into smaller chunks using `RecursiveCharacterTextSplitter` for better embeddings.

3. **Embeddings**:
   - Uses **OpenAI Embeddings** to generate embeddings for the document chunks.

4. **Vector Store**:
   - **FAISS** is used to store and retrieve document embeddings efficiently.

5. **Chat Interface**:
   - Built using **Streamlit**.
   - Uses **ConversationalRetrievalChain** to handle user queries and generate responses.

6. **Memory**:
   - **ConversationBufferMemory** is used to store the chat history.

---

## File Structure

```
Document-Based-Chatbot/
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── .env                    # Environment variables (e.g., OpenAI API key)
├── README.md               # Project documentation
```

---

## Dependencies

- **Streamlit**: For building the web interface.
- **LangChain**: For document loading, text splitting, embeddings, and retrieval.
- **OpenAI**: For generating embeddings and responses using GPT-4o.
- **FAISS**: For efficient similarity search and retrieval.
- **python-dotenv**: For loading environment variables from a `.env` file.

---

## Customization

- **Temperature**: Adjust the `temperature` parameter in the `ChatOpenAI` initialization to control the creativity of the responses.
  ```python
  llm = ChatOpenAI(temperature=0.1, model="gpt-4o")
  ```

- **Chunk Size and Overlap**: Modify the `chunk_size` and `chunk_overlap` parameters in the `RecursiveCharacterTextSplitter` to control how documents are split.
  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  ```

---

## Troubleshooting

1. **No Output**:
   - Ensure that the documents are uploaded and processed successfully.
   - Check the OpenAI API key in the `.env` file.

2. **Error Loading Documents**:
   - Ensure that the uploaded files are in the correct format (PDF, DOCX, CSV, XLSX).
   - Check for any errors in the console.

3. **FAISS Index Not Found**:
   - Ensure that the `faiss_index` directory is created after processing the documents.

---


## Acknowledgments

- **Streamlit**: For the easy-to-use web framework.
- **LangChain**: For the powerful document processing and retrieval tools.
- **OpenAI**: For the GPT-4 model and embeddings.
