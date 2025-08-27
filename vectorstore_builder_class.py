# vectorstore_loader.py

import os
import json
import pandas as pd  # Add pandas for handling Excel files
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import Document  # Import Document class


class VectorstoreBuilder:
    def __init__(self, pdf_directory="./docs", persist_directory="./chroma_db"):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.processed_files_record = os.path.join(persist_directory, "processed_files.json")

        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)

    def robust_load_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif ext == ".xlsx":
                # Load Excel file and convert it to a list of Document objects
                docs = []
                df = pd.read_excel(file_path)
                for _, row in df.iterrows():
                    content = " ".join(map(str, row.values))  # Combine all cell values in a row
                    docs.append(Document(page_content=content, metadata={"source": os.path.basename(file_path), "file_name": os.path.basename(file_path)}))
                return docs
            else:
                print(f"Unsupported file type: {file_path}")
                return []

            docs = loader.load()
        except Exception as error:
            print(f"Failed to load {file_path}: {error}")
            return []

        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_name"] = filename  # Add file_name metadata for all docs
        return docs

    def build_or_update_vectorstore(self):
        if os.path.exists(self.processed_files_record):
            with open(self.processed_files_record, "r") as f:
                processed_files = set(json.load(f))
        else:
            processed_files = set()

        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.persist_directory
        )

        new_documents = []

        for filename in os.listdir(self.pdf_directory):
            if filename.lower().endswith((".pdf", ".txt", ".xlsx")) and filename not in processed_files:
                file_path = os.path.join(self.pdf_directory, filename)
                print(f"Processing new file: {file_path}")
                docs = self.robust_load_file(file_path)
                if docs:
                    new_documents.extend(docs)
                    processed_files.add(filename)

        if new_documents:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000, chunk_overlap=300
            )
            new_doc_splits = splitter.split_documents(new_documents)
            vectorstore = Chroma.from_documents(
                documents=new_doc_splits,
                collection_name="rag-chroma",
                embedding=OpenAIEmbeddings(),
                persist_directory=self.persist_directory
            )

            with open(self.processed_files_record, "w") as f:
                json.dump(list(processed_files), f)

        return vectorstore

    def get_retriever_tool(self):
        vectorstore = self.build_or_update_vectorstore()
        retriever = vectorstore.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            "technical_docs_retriever",
            "Search and return information of how to configure or set up Multitech Gateway and or Dot/xDot. " \
            "It also shows how to configure the Multitech gateway to connect to other LNS server/Basic station like AWS." \
            "If there is no information, please use the internet search tool. Please, include the source of the information in the response."           
        )
        return retriever_tool

    def delete_file_from_vectorstore(self, file_name):
        """
        Remove all documents from the vectorstore whose 'source' metadata matches file_name.
        """
        # Load the vectorstore
        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.persist_directory
        )
        # Chroma supports deletion by filter
        filter_dict = {"source": file_name}
        try:
            vectorstore._collection.delete(where=filter_dict)
            print(f"Deleted all vectors with source={file_name} from vectorstore.")
        except Exception as e:
            print(f"Error deleting vectors for {file_name}: {e}")
        return vectorstore
