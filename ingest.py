import os
from typing import List, Tuple, Union
import logging

from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

from consts import pinecone_db, DOCUMENT_MAP, SOURCE_DIRECTORY
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default=SOURCE_DIRECTORY)
parser.add_argument(
    "--device_type",
    type=str,
    default="cpu",
    choices=["cuda", "cpu", "hip"],
    help="The compute power that you have",
)


def load_single_document(file_path: str) -> Document:
    """
    Load one document from the source documents directory
    """

    file_extension = os.path.splitext(file_path)[1]
    # ingestor = Ingestor(file_extension)
    try:
        loader_class = DOCUMENT_MAP[file_extension]
    except KeyError:
        raise KeyError(f"File extension {file_extension} is not supported")
    finally:
        pass

    loader = loader_class(file_path)

    return loader.load()[0]


def load_document(source_dir: str) -> List[Document]:
    """
    Load all documents from the source documents directory
    """

    all_files = os.listdir(source_dir)
    documents = []

    for file in all_files:
        source_file_path = os.path.join(source_dir, file)
        documents.append(load_single_document(source_file_path))

    return documents


def main():
    args = parser.parse_args()

    logging.info(f"Loading documents from {args.source_dir}")
    documents = load_document(args.source_dir)

    logging.info("Splitting documents into chunks and processing text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)

    logging.info("Creating embedding for the documents")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": args.device_type}
    )

    index_name = "flowise"

    # Create a vector store index
    Pinecone.from_documents(
        texts,
        embeddings,
        index_name=index_name,
    )

    logging.info("Finished create vectorDB index")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    main()
