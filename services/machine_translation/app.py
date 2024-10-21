from fastapi import FastAPI
from langserve import add_routes

from src.translator import ToEnglishTranslator, ToRussianTranslator
from src.langchain_api import TranslatorAPI

app = FastAPI()

ru2en_model = ToEnglishTranslator()
en2ru_model = ToRussianTranslator()

add_routes(app, TranslatorAPI(model=ru2en_model), path="/ru2en")
add_routes(app, TranslatorAPI(model=en2ru_model), path="/en2ru")
