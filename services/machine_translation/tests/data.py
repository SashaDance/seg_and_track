from pydantic.v1 import BaseModel
# from . import response_comparators

from services_api.machine_translation import TranslationRequest


class TestCase(BaseModel):
    name: str
    ru2en: bool
    request: TranslationRequest
    responses: list[str]
    # response_comparator: object = response_comparators.is_translation_equal


test_cases = [
    TestCase(
        name="translate_to_english_start",
        ru2en=True,
        request=TranslationRequest(translation_request="Приступаю к выполнению задания"),
        responses=["Getting Started on a Task", "I'm going to start my assignment"],
    ),
    TestCase(
        name="translate_to_english_started",
        ru2en=True,
        request=TranslationRequest(translation_request="Начал выполнение задания"),
        responses=["Started the task", "I started the assignment"],
    ),
    TestCase(
        name="translate_to_english_action",
        ru2en=True,
        request=TranslationRequest(translation_request="Начал действие"),
        responses=["Started Action", "I started the action"],
    ),
    TestCase(
        name="translate_to_english_shelf",
        ru2en=True,
        request=TranslationRequest(translation_request="Подошел к нужной полке"),
        responses=["I went to the right shelf.", "I approached the right shelf"],
    ),
    TestCase(
        name="translate_to_english_destination",
        ru2en=True,
        request=TranslationRequest(translation_request="Подошел к месту назначения"),
        responses=["Approached the destination", "I approached my destination"],
    ),
    TestCase(
        name="translate_to_english_shelf_2",
        ru2en=True,
        request=TranslationRequest(translation_request="Направляюсь к полке 2"),
        responses=["Heading for shelf 2.", "I am heading for shelf 2"],
    ),
    TestCase(
        name="translate_to_english_shelf_approached_2",
        ru2en=True,
        request=TranslationRequest(translation_request="подошел к полке 2"),
        responses=["I went to the shelf 2.", "I approached shelf 2"],
    ),
    TestCase(
        name="translate_to_english_shelf_approached_5",
        ru2en=True,
        request=TranslationRequest(translation_request="подошел к полке 5"),
        responses=["I approached shelf 5"],
    ),
    TestCase(
        name="translate_to_english_box_5",
        ru2en=True,
        request=TranslationRequest(translation_request="поставил коробку 5 на полку 6"),
        responses=["put box 5 on shelf 6", "I have put box 5 on shelf 6"],
    ),
    TestCase(
        name="translate_to_english_box",
        ru2en=True,
        request=TranslationRequest(translation_request="Взял коробку с товаром"),
        responses=["I took a box of goods.", "I took a box of merchandise"],
    ),
    TestCase(
        name="translate_to_english_collection",
        ru2en=True,
        request=TranslationRequest(translation_request="Выполняю забор товара"),
        responses=["I carry out the collection of goods", "I'm picking up the merchandise"],
    ),
    TestCase(
        name="translate_to_english_off_shelf",
        ru2en=True,
        request=TranslationRequest(translation_request="Взял товар с полки"),
        responses=["Took the goods off the shelf", "I took the merchandise off the shelf"],
    ),
    TestCase(
        name="translate_to_english_collection_5",
        ru2en=True,
        request=TranslationRequest(translation_request="взял коробку 5"),
        responses=["took a box of 5", "I took box 5"],
    ),
    TestCase(
        name="translate_to_english_bring_5",
        ru2en=True,
        request=TranslationRequest(translation_request="Переношу коробку 5"),
        responses=["Carrying box 5", "I am moving box 5"],
    ),
    TestCase(
        name="translate_to_english_moving",
        ru2en=True,
        request=TranslationRequest(translation_request="Двигаюсь к следующей полке"),
        responses=["Moving on to the next shelf.", "I am moving on to the next shelf"],
    ),
    TestCase(
        name="translate_to_english_next",
        ru2en=True,
        request=TranslationRequest(translation_request="Товар на месте, приступаю к следующему шагу"),
        responses=[
            "The goods are in place, proceed to the next step",
            "The goods are in place, I proceed to the next step",
        ],
    ),
    TestCase(
        name="translate_to_english_put",
        ru2en=True,
        request=TranslationRequest(translation_request="Успешно положил товар на полку"),
        responses=["Successfully put the goods on the shelf", "I have successfully put the product on the shelf"],
    ),
    TestCase(
        name="translate_to_english_another",
        ru2en=True,
        request=TranslationRequest(translation_request="Переношу коробку на другую полку"),
        responses=["I'm moving the box to the other shelf.", "I am moving the box to another shelf"],
    ),
    TestCase(
        name="translate_to_english_new",
        ru2en=True,
        request=TranslationRequest(translation_request="Перемещаю товар на новую позицию"),
        responses=["I move the goods to a new position", "I am moving the product to a new position"],
    ),
    TestCase(
        name="translate_to_english_arrived",
        ru2en=True,
        request=TranslationRequest(translation_request="Доставил товар в нужное место"),
        responses=["Delivered goods to the right place", "I delivered the goods to the right place"],
    ),
    TestCase(
        name="translate_to_english_put_5",
        ru2en=True,
        request=TranslationRequest(translation_request="Положил коробку 5 в нужное место"),
        responses=["Put box 5 in the right place", "I have put box 5 in the right place"],
    ),
    TestCase(
        name="translate_to_english_finished",
        ru2en=True,
        request=TranslationRequest(translation_request="Закончил перенос коробки"),
        responses=["Finished the transfer of the box", "I finished moving the box"],
    ),
    TestCase(
        name="translate_to_english_moved_6",
        ru2en=True,
        request=TranslationRequest(translation_request="Товар перемещен на полку 6"),
        responses=["Goods moved on the shelf 6", "The product has been moved to shelf 6"],
    ),
    TestCase(
        name="translate_to_english_is_moved_6",
        ru2en=True,
        request=TranslationRequest(translation_request="Переместил товар на полку 6"),
        responses=["Moved the goods on the shelf 6", "I moved the product to shelf 6"],
    ),
    TestCase(
        name="translate_to_english_place",
        ru2en=True,
        request=TranslationRequest(translation_request="Товар успешно размещен"),
        responses=["Product successfully placed", "The product has been successfully placed"],
    ),
    TestCase(
        name="translate_to_english_complete",
        ru2en=True,
        request=TranslationRequest(translation_request="Перенос завершен"),
        responses=["Transfer completed"],
    ),
    TestCase(
        name="translate_to_english_ready",
        ru2en=True,
        request=TranslationRequest(translation_request="Выполнил задачу, готов к следующей"),
        responses=["Completed the task, ready for the next", "Task accomplished, ready for the next one"],
    ),
    TestCase(
        name="translate_to_english_completion",
        ru2en=True,
        request=TranslationRequest(translation_request="Завершаю задачу"),
        responses=["Finishing the task", "I am completing the task"],
    ),
    TestCase(
        name="translate_to_english_begin",
        ru2en=True,
        request=TranslationRequest(translation_request="Начинаю следующую задачу"),
        responses=["I'm starting the next task.", "I am starting the next task"],
    ),
    TestCase(
        name="translate_to_english_box_complete",
        ru2en=True,
        request=TranslationRequest(translation_request="Завершил задачу с коробкой"),
        responses=["Finished the task with the box", "I completed the task with the box"],
    ),
    TestCase(
        name="translate_to_english_ready_next",
        ru2en=True,
        request=TranslationRequest(translation_request="Готов к следующей задаче"),
        responses=["Ready for the next task"],
    ),
    TestCase(
        name="translate_to_russian_start",
        ru2en=False,
        request=TranslationRequest(translation_request="I'm going to start my assignment"),
        responses=["Я собираюсь начать свое задание", "Приступаю к выполнению задания"],
    ),
    TestCase(
        name="translate_to_russian_started",
        ru2en=False,
        request=TranslationRequest(translation_request="I started the assignment"),
        responses=["Я начал задание", "Начал выполнение задания"],
    ),
    TestCase(
        name="translate_to_russian_action",
        ru2en=False,
        request=TranslationRequest(translation_request="I started the action"),
        responses=["Начал действие"],
    ),
    TestCase(
        name="translate_to_russian_shelf",
        ru2en=False,
        request=TranslationRequest(translation_request="I approached the right shelf"),
        responses=["Подошел к нужной полке"],
    ),
    TestCase(
        name="translate_to_russian_destination",
        ru2en=False,
        request=TranslationRequest(translation_request="I approached my destination"),
        responses=["Я приблизился к своей цели", "Подошел к месту назначения"],
    ),
    TestCase(
        name="translate_to_russian_shelf_2",
        ru2en=False,
        request=TranslationRequest(translation_request="I am heading for shelf 2"),
        responses=["Я направляюсь к полке 2", "Направляюсь к полке 2"],
    ),
    TestCase(
        name="translate_to_russian_shelf_approached_2",
        ru2en=False,
        request=TranslationRequest(translation_request="I approached shelf 2"),
        responses=["Я подошел к полке 2", "подошел к полке 2"],
    ),
    TestCase(
        name="translate_to_russian_shelf_approached_5",
        ru2en=False,
        request=TranslationRequest(translation_request="I approached shelf 5"),
        responses=["Я подошел к полке 5", "подошел к полке 5"],
    ),
    TestCase(
        name="translate_to_russian_box_5",
        ru2en=False,
        request=TranslationRequest(translation_request="I have put box 5 on shelf 6"),
        responses=["Я поставил коробку 5 на полку 6", "поставил коробку 5 на полку 6"],
    ),
    TestCase(
        name="translate_to_russian_box",
        ru2en=False,
        request=TranslationRequest(translation_request="I took a box of merchandise"),
        responses=["Я взял коробку с товарами.", "Взял коробку с товаром"],
    ),
    TestCase(
        name="translate_to_russian_collection",
        ru2en=False,
        request=TranslationRequest(translation_request="I'm picking up the merchandise"),
        responses=["Я забираю товар.", "Выполняю забор товара"],
    ),
    TestCase(
        name="translate_to_russian_off_shelf",
        ru2en=False,
        request=TranslationRequest(translation_request="I took the merchandise off the shelf"),
        responses=["Я снял товар с полки.", "Взял товар с полки"],
    ),
    TestCase(
        name="translate_to_russian_collection_5",
        ru2en=False,
        request=TranslationRequest(translation_request="I took box 5"),
        responses=["Я взял коробку 5", "взял коробку 5"],
    ),
    TestCase(
        name="translate_to_russian_bring_5",
        ru2en=False,
        request=TranslationRequest(translation_request="I am moving box 5"),
        responses=["Я двигаю коробку 5", "Переношу коробку 5"],
    ),
    TestCase(
        name="translate_to_russian_moving",
        ru2en=False,
        request=TranslationRequest(translation_request="I am moving on to the next shelf"),
        responses=["Я перехожу к следующей полке", "Двигаюсь к следующей полке"],
    ),
    TestCase(
        name="translate_to_russian_next",
        ru2en=False,
        request=TranslationRequest(translation_request="The goods are in place, I proceed to the next step"),
        responses=["Товар на месте, я приступаю к следующему шагу", "Товар на месте, приступаю к следующему шагу"],
    ),
    TestCase(
        name="translate_to_russian_put",
        ru2en=False,
        request=TranslationRequest(translation_request="I have successfully put the product on the shelf"),
        responses=["Я успешно поставил продукт на полку", "Успешно положил товар на полку"],
    ),
    TestCase(
        name="translate_to_russian_another",
        ru2en=False,
        request=TranslationRequest(translation_request="I am moving the box to another shelf"),
        responses=["Я перевожу коробку на другую полку.", "Переношу коробку на другую полку"],
    ),
    TestCase(
        name="translate_to_russian_new",
        ru2en=False,
        request=TranslationRequest(translation_request="I am moving the product to a new position"),
        responses=["Перемещаю товар на новую позицию"],
    ),
    TestCase(
        name="translate_to_russian_arrived",
        ru2en=False,
        request=TranslationRequest(translation_request="I delivered the goods to the right place"),
        responses=["Я доставил товар в нужное место", "Доставил товар в нужное место"],
    ),
    TestCase(
        name="translate_to_russian_put_5",
        ru2en=False,
        request=TranslationRequest(translation_request="I have put box 5 in the right place"),
        responses=["Я поставил коробку 5 в нужное место", "Положил коробку 5 в нужное место"],
    ),
    TestCase(
        name="translate_to_russian_finished",
        ru2en=False,
        request=TranslationRequest(translation_request="I finished moving the box"),
        responses=["Я закончил передвигать коробку.", "Закончил перенос коробки"],
    ),
    TestCase(
        name="translate_to_russian_moved_6",
        ru2en=False,
        request=TranslationRequest(translation_request="The product has been moved to shelf 6"),
        responses=["Товар перемещен на полку 6"],
    ),
    TestCase(
        name="translate_to_russian_is_moved_6",
        ru2en=False,
        request=TranslationRequest(translation_request="I moved the product to shelf 6"),
        responses=["Я переместил продукт на полку 6", "Переместил товар на полку 6"],
    ),
    TestCase(
        name="translate_to_russian_place",
        ru2en=False,
        request=TranslationRequest(translation_request="The product has been successfully placed"),
        responses=["Товар успешно размещен"],
    ),
    TestCase(
        name="translate_to_russian_complete",
        ru2en=False,
        request=TranslationRequest(translation_request="Transfer completed"),
        responses=["Передача завершена", "Перенос завершен"],
    ),
    TestCase(
        name="translate_to_russian_ready",
        ru2en=False,
        request=TranslationRequest(translation_request="Task accomplished, ready for the next one"),
        responses=["Задание выполнено, готово к следующему", "Выполнил задачу, готов к следующей"],
    ),
    TestCase(
        name="translate_to_russian_completion",
        ru2en=False,
        request=TranslationRequest(translation_request="I am completing the task"),
        responses=["Я выполняю задание", "Завершаю задачу"],
    ),
    TestCase(
        name="translate_to_russian_begin",
        ru2en=False,
        request=TranslationRequest(translation_request="I am starting the next task"),
        responses=["Я приступаю к следующему заданию", "Начинаю следующую задачу"],
    ),
    TestCase(
        name="translate_to_russian_box_complete",
        ru2en=False,
        request=TranslationRequest(translation_request="I completed the task with the box"),
        responses=["Я выполнил задание с коробкой", "Завершил задачу с коробкой"],
    ),
    TestCase(
        name="translate_to_russian_ready_next",
        ru2en=False,
        request=TranslationRequest(translation_request="Ready for the next task"),
        responses=["Готовы к следующему заданию", "Готов к следующей задаче"],
    ),
]

test_cases = {test_case.name: test_case for test_case in test_cases}
