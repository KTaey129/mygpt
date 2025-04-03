from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ./static 폴더를 '/static' 경로로 매핑
# map ./static folder to '/static'
app.mount("/static", StaticFiles(directory="static"), name="static")
