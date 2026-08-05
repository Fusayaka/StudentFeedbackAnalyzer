import os
import sys
import urllib.request
import zipfile
import shutil
from abc import ABC, abstractmethod

from config import Config

# ==================================================
#                 ABSTRACT INTERFACE                
# ==================================================
class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> str:
        pass

# ==================================================
#                   IMPLEMENTATION                  
# ==================================================
class VnCoreNLPTokenizer(Tokenizer):
    def __init__(self, save_dir: str = os.path.join(Config.ROOT_DIR, "VnCoreNLP")):
        if os.name == 'nt':
            java_home = os.environ.get('JAVA_HOME') or Config.JAVA_HOME
            
            if not java_home:
                raise EnvironmentError(
                    "\n[Error] Không tìm thấy cấu hình JAVA_HOME trên Windows."
                    "\n👉 Cách 1: Thiết lập biến môi trường JAVA_HOME trong hệ thống."
                    "\n👉 Cách 2: Thêm biến JAVA_HOME=\"đường_dẫn_tới_jdk\" vào file .env"
                )
            else:
                os.environ['JAVA_HOME'] = java_home
                jvm_path = os.path.join(java_home, 'bin', 'server')
                os.environ['PATH'] = jvm_path + os.pathsep + os.environ.get('PATH', '')

        import py_vncorenlp
        
        self.save_dir = os.path.abspath(save_dir)
        
        if not os.path.exists(self.save_dir):
            print(f"Đang tải và cài đặt VnCoreNLP tại: {self.save_dir}...")
            self._download_and_extract()
        else:
            print(f"Đã nạp VnCoreNLP từ: {self.save_dir}")

        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir=self.save_dir, 
            max_heap_size='-Xmx500m'
        )

    def _download_and_extract(self):
        import py_vncorenlp
        
        # Windows
        if os.name == 'nt':
            url = "https://github.com/vncorenlp/VnCoreNLP/archive/master.zip"
            zip_path = "VnCoreNLP_master.zip"
            
            try:
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(".")
                if os.path.exists("VnCoreNLP-master"):
                    os.rename("VnCoreNLP-master", self.save_dir)
            except Exception as e:
                raise RuntimeError(f"[Error] Không thể tải file ZIP: {e}")
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        # Linux / MacOS   
        else:
            py_vncorenlp.download_model(save_dir=self.save_dir)

        self._cleanup_unnecessary_files()
        print("Tải và thiết lập VnCoreNLP hoàn tất.")

    def _cleanup_unnecessary_files(self):
        # Chỉ giữ lại wordsegmenter để tối ưu không gian lưu trữ
        models_dir = os.path.join(self.save_dir, "models")
        for folder in ["dep", "ner", "postagger"]:
            p = os.path.join(models_dir, folder)
            if os.path.exists(p): 
                shutil.rmtree(p)
                
        trash_items = ["src", "pom.xml", "Readme.md", "LICENSE.md", "TagsetDescription.md", "VnCoreNLP-1.1.1.jar"]
        for item in trash_items:
            p = os.path.join(self.save_dir, item)
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)

    def tokenize(self, text: str) -> str:
        segmented = self.rdrsegmenter.word_segment(text)
        if isinstance(segmented, list) and len(segmented) > 0:
            return " ".join(segmented)
        return str(segmented)
    
class UndertheseaTokenizer(Tokenizer):
    def __init__(self):
        from underthesea import word_tokenize
        self.word_tokenize = word_tokenize
        print("Đã nạp Underthesea Tokenizer.")

    def tokenize(self, text: str) -> str:
        tokens = self.word_tokenize(text, format="text")
        if isinstance(tokens, list):
            return " ".join(tokens)
        return tokens