from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import tempfile
import os
from main import PaperReaderProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.processor = PaperReaderProcessor('../config/config.json', '../config/api_keys.json')
    yield
    # 可选：在这里清理资源
    # del app.state.processor 或关闭文件/释放GPU等

app = FastAPI(lifespan=lifespan)

@app.post("/process")
async def process_input(
    request: Request,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    if file and text:
        raise HTTPException(status_code=400, detail="只能上传 PDF 或输入文本中的一个，不可同时上传。")
    
    if not file and not text:
        raise HTTPException(status_code=400, detail="请输入文本或上传 PDF 文件。")
    
    processor = request.app.state.processor

    if file:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="仅支持 PDF 文件。")
        
        # 将 PDF 保存到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            result = processor.process_pdf(tmp_path, f'../outputs')
            # =====================================
        finally:
            os.remove(tmp_path)

        return JSONResponse(content=result)

    if text:
        result = processor.process_query(text)
        return JSONResponse(content=result)

