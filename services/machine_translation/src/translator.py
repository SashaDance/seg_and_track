from abc import ABC, abstractmethod
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.runnables import RunnablePassthrough

from services_api.machine_translation import TranslationResponse

device = "cuda"


class BaseTranslator(ABC):
    @abstractmethod
    def translator(self, text: str) -> TranslationResponse:
        pass


class LangChainTranslator(BaseTranslator):
    def __init__(self, model_name: str, prefix: str = ""):
        self.model_name = model_name
        translator = pipeline("translation", model=model_name, device=device)
        self.llm = HuggingFacePipeline(pipeline=translator)
        self.prompt = PromptTemplate(template=f"{prefix}{{text}}", input_variables=["text"])
        self.translation_chain = {"text": RunnablePassthrough()} | self.prompt | self.llm

    def translator(self, text: str) -> TranslationResponse:
        translation = self.translation_chain.invoke(text)
        ret = TranslationResponse(translation_response=translation)
        return ret


class ToEnglishTranslator(LangChainTranslator):
    def __init__(self):
        super().__init__("Helsinki-NLP/opus-mt-tc-big-zle-en")


class ToRussianTranslator(LangChainTranslator):
    def __init__(self):
        super().__init__("Helsinki-NLP/opus-mt-tc-big-en-zle", prefix=">>rus<< ")
