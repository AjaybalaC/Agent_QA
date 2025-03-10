import pandas as pd
import PyPDF2
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class DocumentProcessor:
    def __init__(self):
        """Initialize embedding model"""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model

    @staticmethod
    def read_pdf(file) -> str:
        """Read and extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                text += extracted if extracted else ""
            return text.strip()
        except Exception as e:
            print(f"PDF Processing Error: {e}")
            return None

    @staticmethod
    def read_csv(file) -> Dict:
        """Read CSV file and return raw DataFrame, text, and JSON"""
        try:
            df = pd.read_csv(file)
            pd.set_option('display.max_rows', None)
            return {
                "df": df,
                "text": df.to_string(index=False).strip(),
                "json": df.to_json(orient="records")
            }
        except Exception as e:
            print(f"CSV Processing Error: {e}")
            return None

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word) + 1
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def chunk_csv(self, df: pd.DataFrame, chunk_size: int = 10) -> List[str]:
        """Chunk CSV rows into groups"""
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i + chunk_size]
            chunks.append(chunk_df.to_string(index=False))
        return chunks

    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        return self.embedder.encode(chunks, convert_to_tensor=False).tolist()