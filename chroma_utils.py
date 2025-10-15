#!/usr/bin/env python3
"""
ChromaDB Utility Script - Manage your vector database
Usage: python chroma_utils.py [command]
"""

import argparse
import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
import chromadb 
from chromadb.config import Settings

CHROMA_PATH = "chroma"


def stats():
    """Show database statistics"""
    if not os.path.exists(CHROMA_PATH):
        print("❌ No database found. Run 'python populate_database.py' first.")
        return
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        
        results = db.get()
        
        print("=" * 60)
        print("📊 ChromaDB Statistics")
        print("=" * 60)
        
        if not results["ids"]:
            print("Database is empty.")
            return
        
        # Count documents
        sources = set()
        for metadata in results["metadatas"]:
            source = metadata.get("source", "Unknown")
            sources.add(source)
        
        print(f"\n📁 Total Documents: {len(sources)}")
        print(f"📄 Total Chunks: {len(results['ids'])}")
        
        print("\n📚 Documents:")
        for i, source in enumerate(sorted(sources), 1):
            # Count chunks per document
            chunk_count = sum(1 for m in results["metadatas"] 
                            if m.get("source") == source)
            print(f"  {i}. {os.path.basename(source)} ({chunk_count} chunks)")
        
        # Database size
        size = get_folder_size(CHROMA_PATH)
        print(f"\n💾 Database Size: {size:.2f} MB")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def get_folder_size(path):
    """Calculate folder size in MB"""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)


def search_test(query: str, k: int = 3):
    """Test search functionality"""
    if not os.path.exists(CHROMA_PATH):
        print("❌ No database found.")
        return
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        
        print(f"\n🔍 Searching for: '{query}'")
        print("=" * 60)
        
        results = db.similarity_search_with_score(query, k=k)
        
        if not results:
            print("No results found.")
            return
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n{i}. Relevance Score: {score:.4f}")
            print(f"   Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
            print(f"   Page: {doc.metadata.get('page', 'N/A')}")
            print(f"   Preview: {doc.page_content[:150]}...")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def backup(backup_path: str = "chroma_backup"):
    """Backup database"""
    if not os.path.exists(CHROMA_PATH):
        print("❌ No database to backup.")
        return
    
    try:
        if os.path.exists(backup_path):
            response = input(f"⚠️  Backup exists at {backup_path}. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Backup cancelled.")
                return
            shutil.rmtree(backup_path)
        
        shutil.copytree(CHROMA_PATH, backup_path)
        size = get_folder_size(backup_path)
        print(f"✅ Database backed up to {backup_path} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def restore(backup_path: str = "chroma_backup"):
    """Restore database from backup"""
    if not os.path.exists(backup_path):
        print(f"❌ Backup not found at {backup_path}")
        return
    
    try:
        print("⚠️  This will replace your current database.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Restore cancelled.")
            return
        
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        
        shutil.copytree(backup_path, CHROMA_PATH)
        size = get_folder_size(CHROMA_PATH)
        print(f"✅ Database restored from {backup_path} ({size:.2f} MB)")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def clear():
    """Clear database"""
    if not os.path.exists(CHROMA_PATH):
        print("❌ No database to clear.")
        return
    
    try:
        print("⚠️  WARNING: This will delete all data in the database.")
        response = input("Are you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Clear cancelled.")
            return
        
        # Initialize the client with settings that allow reset
        client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(allow_reset=True) # <--- ADD THIS LINE
        )
        client.reset()
        
        # The reset() function now handles directory removal
        print("✅ Database cleared.")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def list_docs():
    """List all documents"""
    if not os.path.exists(CHROMA_PATH):
        print("❌ No database found.")
        return
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )
        
        results = db.get()
        
        if not results["metadatas"]:
            print("No documents in database.")
            return
        
        sources = set()
        for metadata in results["metadatas"]:
            sources.add(metadata.get("source", "Unknown"))
        
        print("\n📚 Documents in Database:")
        print("=" * 60)
        for i, source in enumerate(sorted(sources), 1):
            chunk_count = sum(1 for m in results["metadatas"] 
                            if m.get("source") == source)
            print(f"{i}. {os.path.basename(source)}")
            print(f"   Path: {source}")
            print(f"   Chunks: {chunk_count}\n")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ChromaDB Utility - Manage your vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chroma_utils.py stats                    # Show database statistics
  python chroma_utils.py list                     # List all documents
  python chroma_utils.py search "your query"      # Test search
  python chroma_utils.py backup                   # Backup database
  python chroma_utils.py restore                  # Restore from backup
  python chroma_utils.py clear                    # Clear database (careful!)
        """
    )
    
    parser.add_argument(
        "command",
        choices=["stats", "list", "search", "backup", "restore", "clear"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query (for 'search' command)"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of results (for 'search' command)"
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        default="chroma_backup",
        help="Backup path (for 'backup'/'restore' commands)"
    )
    
    args = parser.parse_args()
    
    if args.command == "stats":
        stats()
    elif args.command == "list":
        list_docs()
    elif args.command == "search":
        if not args.query:
            print("❌ Please provide a query: python chroma_utils.py search \"your query\"")
            return
        search_test(args.query, args.top_k)
    elif args.command == "backup":
        backup(args.path)
    elif args.command == "restore":
        restore(args.path)
    elif args.command == "clear":
        clear()


if __name__ == "__main__":
    main()
