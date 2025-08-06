import os
import logging
from typing import List, Iterator, Any
import torch
from chardet.universaldetector import UniversalDetector
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TextSplitter


class RetrievalDatabaseBuilder:
    """
    Builds and manages Chroma vector databases for document retrieval.
    
    Handles file encoding detection, document splitting, embedding generation,
    and database persistence/loading.
    """
    
    def __init__(self, persist_root: str = "./RetrievalBase", device: str = "auto"):
        """
        Initialize the builder.
        
        Args:
            persist_root: Root directory for persisting databases
            device: Device for embedding generation ('auto', 'cuda', or 'cpu')
        """
        self.persist_root = persist_root
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logging.info(f"Using device: {self.device}")

    def find_all_files(self, path: str) -> Iterator[str]:
        """
        Recursively find all files in the given directory.
        
        Args:
            path: Directory path to search
            
        Yields:
            Full path to each file found
        """
        for root, _, files in os.walk(path):
            for f in files:
                yield os.path.join(root, f)

    def get_file_encoding(self, path: str) -> str:
        """
        Detect the encoding of a text file.
        
        Args:
            path: Path to the file
            
        Returns:
            Detected encoding (e.g., 'utf-8', 'iso-8859-1')
        """
        detector = UniversalDetector()
        with open(path, "rb") as file:
            for line in file:
                detector.feed(line)
                if detector.done:
                    break
        detector.close()
        return detector.result["encoding"]

    def get_embedding_model(self, name: str, batch_size: int = 256) -> HuggingFaceEmbeddings:
        """
        Create a HuggingFace embeddings model.
        
        Args:
            name: Name of the HuggingFace model
            batch_size: Batch size for embedding generation
            
        Returns:
            Configured embedding model
        """
        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": self.device},
            encode_kwargs={"device": self.device, "batch_size": batch_size},
        )

    def construct(
        self,
        dataset_paths: List[str],
        splitters: List[TextSplitter],
        encoder_model_name: str = "bge-large-en-v1.5",
        db_batch_size: int = 256,
    ) -> Chroma:
        """
        Construct a Chroma database from files in dataset paths.
        
        Loads existing database if found, otherwise builds new one from files.
        
        Args:
            dataset_paths: Directory paths containing documents
            splitters: Text splitters for each dataset
            encoder_model_name: HuggingFace model for embeddings
            db_batch_size: Batch size for embedding generation
            
        Returns:
            The constructed or loaded Chroma database
            
        Raises:
            AssertionError: If dataset paths and splitters don't match
        """
        assert len(dataset_paths) == len(
            splitters
        ), "Each dataset must have a corresponding splitter."

        db_name = "_".join([os.path.basename(p.rstrip("/")) for p in dataset_paths])
        persist_path = os.path.join(self.persist_root, db_name, encoder_model_name)
        
        # Check if DB already exists (directory exists and is non-empty)
        if os.path.exists(persist_path) and os.listdir(persist_path):
            logging.info(f"Chroma DB already exists at {persist_path}. Loading...")
            return self.load(persist_path, encoder_model_name, db_batch_size)

        embed_model = self.get_embedding_model(encoder_model_name, db_batch_size)
        all_docs = []
        for path, splitter in zip(dataset_paths, splitters):
            for file in self.find_all_files(path):
                enc = self.get_file_encoding(file)
                loader = TextLoader(file, encoding=enc)
                docs = loader.load()
                all_docs.extend(splitter.split_documents(docs))

        logging.info(f"Building Chroma DB at {persist_path}...")
        db = Chroma.from_documents(
            documents=all_docs, embedding=embed_model, persist_directory=persist_path
        )
        db.persist()
        logging.info(f"Persisted Chroma DB at {persist_path}.")
        return db

    def construct_from_documents(
        self,
        documents: List[Any],
        encoder_model_name: str = "bge-large-en-v1.5",
        db_name: str = "default",
        db_batch_size: int = 256,
    ) -> Chroma:
        """
        Construct a Chroma database from pre-loaded documents.
        
        Args:
            documents: Pre-loaded documents to add to database
            encoder_model_name: HuggingFace model for embeddings
            db_name: Name for the database
            db_batch_size: Batch size for embedding generation
            
        Returns:
            The constructed or loaded Chroma database
        """
        persist_path = os.path.join(self.persist_root, db_name, encoder_model_name)
        # Check if DB already exists (directory exists and is non-empty)
        if os.path.exists(persist_path) and os.listdir(persist_path):
            logging.info(f"Chroma DB already exists at {persist_path}. Loading...")
            return self.load(persist_path, encoder_model_name, db_batch_size)

        embed_model = self.get_embedding_model(encoder_model_name, db_batch_size)
        logging.info(f"Building Chroma DB at {persist_path}...")
        db = Chroma.from_documents(
            documents=documents, embedding=embed_model, persist_directory=persist_path
        )
        db.persist()
        logging.info(f"Persisted Chroma DB at {persist_path}.")
        return db

    def load(
        self,
        persist_path: str,
        encoder_model_name: str = "bge-large-en-v1.5",
        db_batch_size: int = 256,
    ) -> Chroma:
        """
        Load an existing Chroma database from disk.
        
        Args:
            persist_path: Path to the persisted database
            encoder_model_name: HuggingFace model for embeddings
            db_batch_size: Batch size for embedding generation
            
        Returns:
            The loaded Chroma database
        """
        embed_model = self.get_embedding_model(encoder_model_name, db_batch_size)
        logging.info(f"Loading Chroma DB from {persist_path}...")
        return Chroma(embedding_function=embed_model, persist_directory=persist_path)