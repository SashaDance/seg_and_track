from typing import Any, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from .translator import BaseTranslator

import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)


class TranslatorAPI(LLM):
    model: BaseTranslator

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.model.translator(prompt).json()

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model.model_name}

    @property
    def _llm_type(self) -> str:
        return "machine_translation"
