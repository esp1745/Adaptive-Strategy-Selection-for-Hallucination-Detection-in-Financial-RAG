"""
Financial RAG System
Loads SEC filings and creates searchable index for financial Q&A
"""

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict

class FinancialRAG:
    """RAG system for financial documents from SEC filings"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize RAG system with embedding model"""
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.embeddings = None
        self.index = None
        print(f"Model loaded: {model_name}")
    
    def chunk_document(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split document into overlapping chunks"""
        words = text.split()
        
        if len(words) < chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = ' '.join(words[start:end])
            
            if len(chunk.strip()) > 100:
                chunks.append(chunk)
            
            start = end - overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def load_sec_filings(self, filings_dir: str = 'data/raw/sec_filings'):
        """Load and index SEC filings from JSON files"""
        filings_path = Path(filings_dir)
        
        if not filings_path.exists():
            raise FileNotFoundError(
                f"Directory not found: {filings_dir}\n"
                "Please run: python3 sec_downloader.py first"
            )
        
        print(f"\nLoading SEC filings from: {filings_path}")
        
        json_files = list(filings_path.glob('*.json'))
        
        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {filings_dir}\n"
                "Please run: python3 sec_downloader.py first"
            )
        
        print(f"Found {len(json_files)} company files")
        
        all_chunks = []
        all_metadata = []
        
        for json_file in json_files:
            print(f"\n  Loading: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    company_data = json.load(f)
                
                company_name = company_data['company_name']
                
                for filing in company_data['filings']:
                    content = filing['content']
                    filing_type = filing['type']
                    filing_date = filing['date']
                    
                    print(f"    Processing {filing_type} from {filing_date}...")
                    print(f"    Content length: {len(content):,} characters")
                    
                    chunks = self.chunk_document(content, chunk_size=300, overlap=50)
                    print(f"    Created {len(chunks)} chunks")
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadata.append({
                            'company': company_name,
                            'filing_type': filing_type,
                            'filing_date': filing_date,
                            'chunk_id': i,
                            'total_chunks': len(chunks),
                            'source_file': json_file.name
                        })
            
            except Exception as e:
                print(f"    ERROR loading {json_file.name}: {e}")
                continue
        
        if not all_chunks:
            raise ValueError("No chunks created! Check your SEC filing files.")
        
        print(f"\nTotal chunks loaded: {len(all_chunks)}")
        print(f"Companies: {len(set(m['company'] for m in all_metadata))}")
        
        self.documents = all_chunks
        self.metadata = all_metadata
        
        print("\nCreating embeddings...")
        print("(This may take a few minutes for large documents)")
        
        self.embeddings = self.encoder.encode(
            all_chunks,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        print(f"Created embeddings: {self.embeddings.shape}")
        
        print("\nBuilding FAISS search index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"FAISS index built with {self.index.ntotal} vectors")
        print("\n" + "="*60)
        print("RAG SYSTEM READY")
        print("="*60)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            raise Exception("No documents loaded! Call load_sec_filings() first")
        
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'text': self.documents[idx],
                'score': float(dist),
                'company': self.metadata[idx]['company'],
                'filing_type': self.metadata[idx]['filing_type'],
                'filing_date': self.metadata[idx]['filing_date'],
                'chunk_id': self.metadata[idx]['chunk_id']
            })
        
        return results
    
    def save(self, output_dir: str = 'models/financial_rag'):
        """Save RAG system to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving RAG system to: {output_path}")
        
        faiss.write_index(self.index, str(output_path / 'faiss_index.bin'))
        print("  Saved FAISS index")
        
        with open(output_path / 'documents.json', 'w', encoding='utf-8') as f:
            json.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f, indent=2, ensure_ascii=False)
        print("  Saved documents and metadata")
        
        np.save(output_path / 'embeddings.npy', self.embeddings)
        print("  Saved embeddings")
        
        print("\nRAG system saved successfully")
    
    def load(self, input_dir: str = 'models/financial_rag'):
        """Load a previously saved RAG system"""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"RAG system not found: {input_dir}")
        
        print(f"\nLoading RAG system from: {input_path}")
        
        self.index = faiss.read_index(str(input_path / 'faiss_index.bin'))
        print("  Loaded FAISS index")
        
        with open(input_path / 'documents.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
        print("  Loaded documents and metadata")
        
        self.embeddings = np.load(input_path / 'embeddings.npy')
        print("  Loaded embeddings")
        
        print(f"\nRAG system loaded: {len(self.documents)} documents")


def test_rag_system(rag: FinancialRAG):
    """Test the RAG system with sample financial queries"""
    
    test_queries = [
        "What was Apple's total revenue in fiscal year 2023?",
        "What are Tesla's main risk factors?",
        "What business segments does Microsoft operate in?",
        "What was Amazon's operating income?",
        "What is Google's advertising business strategy?",
    ]
    
    print("\n" + "="*80)
    print("TESTING RAG SYSTEM WITH FINANCIAL QUERIES")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print("="*80)
        
        results = rag.retrieve(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['company']}] {result['filing_type']} ({result['filing_date']})")
            print(f"   Relevance Score: {result['score']:.3f} (lower is better)")
            print(f"   Chunk {result['chunk_id']+1}")
            print(f"   Text: {result['text'][:300]}...")
            if len(result['text']) > 300:
                print(f"         [...{len(result['text'])-300} more characters]")


def main():
    """Main execution: Load filings, build RAG, test, and save"""
    
    print("\n" + "="*80)
    print("FINANCIAL RAG SYSTEM - BUILD & TEST")
    print("="*80)
    
    rag = FinancialRAG()
    
    try:
        rag.load_sec_filings('data/raw/sec_filings')
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run: python3 sec_downloader.py first")
        return
    
    test_rag_system(rag)
    
    rag.save('models/financial_rag')
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total documents: {len(rag.documents)}")
    print(f"Total companies: {len(set(m['company'] for m in rag.metadata))}")
    print(f"Companies: {', '.join(sorted(set(m['company'] for m in rag.metadata)))}")
    print(f"Embedding dimension: {rag.embeddings.shape[1]}")
    print(f"Model: all-MiniLM-L6-v2")
    print(f"\nRAG system saved to: models/financial_rag")
    
    print("\n" + "="*80)
    print("SUCCESS - Your RAG system is ready for Week 1")
    print("="*80)
    
    print("\nNext Steps:")
    print("  1. Create 20 test questions about these companies")
    print("  2. Label ground truth for each question")
    print("  3. Measure retrieval accuracy (Precision@3, Recall@3)")
    print("  4. Document baseline performance")
    print("\nWeek 1 Goal: RAG baseline complete by Feb 16")


if __name__ == "__main__":
    main()