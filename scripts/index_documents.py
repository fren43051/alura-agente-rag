"""Script de indexación: carga todos los docs de data/ hacia Pinecone.

Uso:
    python scripts/index_documents.py
    python scripts/index_documents.py --dir docs/contratos
    python scripts/index_documents.py --stats
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorstore.pinecone_store import PineconeStore
from src.config import DOCS_DIR


def main():
    parser = argparse.ArgumentParser(description='Indexar documentos en Pinecone')
    parser.add_argument('--dir',   default=str(DOCS_DIR), help='Directorio de documentos')
    parser.add_argument('--stats', action='store_true',   help='Ver estadísticas del índice')
    args = parser.parse_args()

    store = PineconeStore()

    if args.stats:
        stats = store.stats()
        print('\n=== Estadísticas del índice Pinecone ===')
        print(f'  Total vectores : {stats["total_vector_count"]}')
        print(f'  Dimensión      : {stats["dimension"]}')
        return

    total = store.upsert_from_directory(args.dir)
    print(f'\n✅ Indexación completada: {total} vectores en Pinecone')


if __name__ == '__main__':
    main()
