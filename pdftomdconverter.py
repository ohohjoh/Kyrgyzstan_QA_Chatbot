import os
import pymupdf4llm
from tqdm import tqdm # 진행률 표시를 위한 라이브러리 (선택 사항: pip install tqdm)

def convert_pdfs_to_md(input_folder="data", output_folder="data/processed_md"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"'{input_folder}' 폴더에 PDF 파일이 없습니다.")
        return

    print(f"'{input_folder}'의 PDF 파일을 '{output_folder}'로 Markdown 변환 중...")
    for pdf_file in tqdm(pdf_files, desc="Converting PDFs"):
        pdf_path = os.path.join(input_folder, pdf_file)
        md_filename = os.path.splitext(pdf_file)[0] + ".md"
        output_md_path = os.path.join(output_folder, md_filename)

        try:
            # PDF를 Markdown 텍스트로 변환
            md_text = pymupdf4llm.to_markdown(pdf_path)
            
            # 변환된 Markdown을 파일로 저장
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            # print(f" - '{pdf_file}' -> '{md_filename}' 변환 완료.")
        except Exception as e:
            print(f" - 오류: '{pdf_file}' 변환 중 오류 발생: {e}")

if __name__ == "__main__":
    convert_pdfs_to_md()
    print("PDF를 Markdown으로 변환하는 전처리 과정이 완료되었습니다.")
    print("이제 app.py에서 'data/processed_md' 폴더의 마크다운 파일을 로드하도록 설정해야 합니다.")