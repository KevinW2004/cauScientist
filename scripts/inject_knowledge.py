import sys
import os
from pathlib import Path
import fitz  # PyMuPDF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from memory.memory_service import MemoryService

memory = MemoryService()

def inject_knowledge():
    """将 data/documents 目录下的文档注入到向量数据库"""
    # 获取 documents 目录路径
    project_root = Path(__file__).parent.parent
    documents_dir = project_root / "data" / "documents"
    
    if not documents_dir.exists():
        print(f"❌ Documents directory not found: {documents_dir}")
        return
    
    # 支持的文件类型
    supported_extensions = [".txt", ".pdf"]
    
    # 遍历目录中的所有文件
    files = [f for f in documents_dir.iterdir() if f.is_file() and f.suffix in supported_extensions]
    
    if not files:
        print(f"⚠️ No supported files found in {documents_dir}")
        return
    
    print(f"📂 Found {len(files)} document(s) to inject:")
    for file in files:
        print(f"  - {file.name}")
    
    # 逐个处理文件
    for file_path in files:
        try:
            print(f"\n📄 Processing: {file_path.name}...")
            
            # 根据文件类型读取内容
            if file_path.suffix == ".txt":
                content = read_txt_file(file_path)
            elif file_path.suffix == ".pdf":
                content = read_pdf_file(file_path)
            else:
                continue
            
            if not content or not content.strip():
                print(f"⚠️ Empty content in {file_path.name}, skipping...")
                continue
            
            # 准备元数据
            metadata = {
                "source": "inject_knowledge",
                "filename": file_path.name,
                "file_type": file_path.suffix[1:],  # 去掉点号
                "file_path": str(file_path.relative_to(project_root)),
            }
            
            # 注入到向量数据库
            memory.save_memory(text=content, metadata=metadata)
            print(f"✅ Successfully injected: {file_path.name}")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {str(e)}")
            continue
    
    print(f"\n🎉 Knowledge injection completed!")


def read_txt_file(file_path: Path) -> str:
    """读取 TXT 文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf_file(file_path: Path) -> str:
    """读取 PDF 文件内容（使用 pymupdf）"""
    doc = fitz.open(file_path)
    content = ""
    for page in doc:
        blocks = page.get_text("blocks", sort=True) #type: ignore
        for block in blocks:
            content += block[4]
    return content
    

if __name__ == "__main__":
    inject_knowledge()

