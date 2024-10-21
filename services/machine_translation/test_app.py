import pytest

# from src.models import TranslationResponse
from services_api import machine_translation
from tests.data import test_cases

# f = open("test_app.log", "wt")


@pytest.mark.parametrize("test_case", test_cases.values(), ids=test_cases.keys())
def test(test_case):
    if test_case.ru2en:
        response = machine_translation.ru2en_chain.invoke(test_case.request.translation_request)
    else:
        response = machine_translation.en2ru_chain.invoke(test_case.request.translation_request)
    print("response: ", response)
    response.translation_response in test_case.responses
    # f.write("--- " + str(response.translation_response)+"\n")
    if response.translation_response in test_case.responses:
        print("response: ", response)
    assert response.translation_response in test_case.responses
