Домашнє завдання №1

ReAct агент у LangGraph



Вступ



⚠️Максимальна кількість балів за домашнє завдання — 100 балів. Вага ДЗ1 у підсумковій оцінці — 20%. Орієнтовна трудомісткість: 7 годин.


У цьому домашньому завданні ви побудуєте повноцінного ReAct-агента у LangGraph — від проєктування інструментів з Pydantic-валідацією до логування траєкторії у JSON. Робота базується на матеріалах тем 1–2 (автономні агенти, tool/function calling, ReAct-патерн) та теми 4 (LangGraph). Ви реалізуєте цикл LLM–tools–LLM, додасте захисні механізми (max_steps, timeout) і протестуєте агента на 3–5 тест-кейсах.



Агент повинен працювати з обраним вами LLM-провайдером: OpenAI API (GPT-4.1), Google Gemini API (gemini-2.5-flash), Mistral API або локальна модель через Ollama.



ReAct (Reasoning + Acting) — це патерн, де LLM чергує кроки міркування (Thought: що мені потрібно зробити?) та дії (Action: виклик інструменту). Після отримання результату (Observation) LLM міркує знову і вирішує: потрібна ще одна дія чи можна сформувати фінальну відповідь. У LangGraph цей цикл реалізується як граф: вузол agent (LLM) → вузол tools (виконання) → знову agent, з conditional edge, що визначає, чи є ще tool calls.



🎯Навчальні результати

Після виконання цього домашнього завдання ви зможете:
- проєктувати інструменти (tools) з валідацією Pydantic v2 та документацією, придатною для tool selection LLM;
- будувати ReAct-агента у LangGraph через StateGraph з ToolNode та conditional edges;
- реалізовувати захисні механізми агента: max_steps, timeout, детекцію зациклення;
- логувати траєкторію виконання агента у структурованому JSON-форматі для post-mortem аналізу;
- оцінювати поведінку агента на тест-кейсах різної складності та обґрунтовувати інженерні рішення.


Що саме потрібно зробити:



Реалізувати 3–4 інструменти (tools) з Pydantic v2 схемами та валідацією. Кожен tool має: BaseModel для параметрів, Field з description, field_validator для перевірки вхідних даних, docstring для LLM.
Побудувати LangGraph з циклом LLM–tools–LLM та structured outputs. StateGraph з вузлами: agent (LLM), tools (виконання), conditional edge (router). Structured outputs через with_structured_output або Pydantic response models.
Використати на вибір один LLM-провайдер: OpenAI API (GPT-5.2), Google Gemini API (gemini-3.8-flash), Mistral API або Ollama (Llama 3.1, Qwen 3, GPT-OSS — моделі з підтримкою tool calling). Підключення через LangChain: ChatOpenAI, ChatGoogleGenerativeAI, ChatMistralAI, ChatOllama.
Додати захисні механізми: max_steps (обмеження кількості ітерацій циклу), timeout (загальний тайм-аут виконання), детекція повторюваних tool calls (зациклення). Логування повної траєкторії виконання у JSON-файл.
Прогнати 3–5 тест-кейсів і зібрати результати. Кожен тест-кейс: input query, очікуваний результат, фактичний результат, кількість кроків, tool calls, час виконання. Результати зберегти у JSON-файл.


Вимоги до роботи



Рішення оформлюється як Python-репозиторій (GitHub) або Jupyter Notebook/Google Colab.
Код структурований, з коментарями українською мовою.
Кожен tool має Pydantic v2 схему з Field та field_validator.
LangGraph граф з циклом LLM–tools–LLM працює коректно.
Захисні механізми (max_steps, timeout) реалізовані та продемонстровані.
Траєкторія логується у JSON (trajectory.json).
Результати 3–5 тест-кейсів зібрані у test_results.json.
README.md з описом проєкту, інструкцією запуску, аналізом результатів.




Як буде оцінюватися робота (100 балів)



Завдання 1. Інструменти з Pydantic-схемами (20 балів):

10 балів — 3–4 tools з BaseModel, Field(description=...), field_validator. Кожен tool має коректний docstring.
5 балів — валідація працює: некоректні входи відхиляються з зрозумілим повідомленням.
5 балів — tools протестовані окремо (unit tests або демонстрація).
Завдання 2. LangGraph з ReAct-циклом (30 балів):

10 балів — StateGraph з вузлами agent + tools + conditional edge працює.
10 балів — LLM правильно вибирає tools та формує відповідь на основі результатів.
5 балів — structured outputs: відповідь у структурованому форматі.
5 балів — працює з обраним LLM-провайдером (OpenAI / Gemini / Mistral / Ollama).
Завдання 3. Захисні механізми та логування (25 балів):

8 балів — max_steps реалізовано та продемонстровано (що відбувається при досягненні ліміту).
7 балів — timeout реалізовано та продемонстровано.
5 балів — детекція зациклення (повторюваних tool calls).
5 балів — JSON-логування траєкторії (trajectory.json з усіма кроками).
Завдання 4. Тестування та результати (25 балів):

10 балів — 3–5 тест-кейсів різної складності прогнані та задокументовані.
5 балів — test_results.json з повними результатами (query, actual, steps, tools, time).
5 балів — README з аналізом: де агент працює добре, де помиляється, чому.
5 балів — код структурований, коментарі, зрозуміла архітектура.


Формат здачі

Артефакти, які потрібно здати:

Python-репозиторій (GitHub) або Jupyter Notebook (.ipynb) / Google Colab з повним кодом.
trajectory.json — JSON-лог траєкторії виконання агента (результат Завдання 3).
test_results.json — результати тестування (результат завдання 4).
README.md з: описом проєкту, обраною доменною задачею, інструкцією запуску, аналізом результатів тестування, відповідями на питання.
Файли завантажити у систему до встановленого дедлайну. 
Назва: HW1_Прізвище.zip (або посилання на GitHub/Colab).




Підказки для легшого виконання



Перед початком роботи. Оберіть доменну задачу, яка вам цікава: фінанси, подорожі, DevOps, медицина, спорт, кулінарія — будь-що. Цікавий домен мотивує краще, ніж абстрактний приклад. Інструменти повинні бути реалістичними для обраного домену.
Робота з Pydantic v2. Використовуйте Field(description=...) для кожного параметра — LLM читає description при виборі tool. field_validator з @classmethod — для перевірки значень. model_config = ConfigDict(str_strip_whitespace=True) — автоматичне видалення пробілів.
Вибір LLM-провайдера. Gemini 3.8 Flash — безкоштовний free tier, добра якість tool calling (параметр temperature не змінюйте: для моделей Gemini 3 значення нижче 1.0 можуть спричиняти зациклення). GPT-5.2 — найкраща якість, але платний. Ollama — безкоштовний, працює локально, але якість залежить від моделі. Для ДЗ будь-який провайдер підходить.
Debugging LangGraph. Увімкніть verbose=True або використовуйте print() у кожному node для відстеження потоку виконання. Перевіряйте state["messages"] після кожного кроку. Якщо агент не викликає tool — перевірте docstring та description tools.
JSON-логування. Використовуйте ensure_ascii=False для коректного відображення кирилиці; indent=2 для читабельного форматування. Обмежуйте довжину полів ([:500]) щоб файл не став занадто великим.


Типові помилки та як їх уникнути:

«Агент не викликає tool» — перевірте docstring: LLM повинен зрозуміти, КОЛИ використовувати цей tool. Додайте приклади у docstring.
«Агент зациклюється» — перевірте, що max_steps працює; перевірте, чи tool повертає корисну інформацію (а не порожній рядок).
«Pydantic validation error» — це ДОБРЕ, це означає, що валідація працює. Просто покажіть, що агент обробляє цю помилку.
«Graph не компілюється» — перевірте, що всі ребра додані: START→agent, agent→(conditional)→tools/end, tools→agent.


Структура проєкту (рекомендована)

# Рекомендована структура файлів:
# hw1_react_agent/
# ├── tools.py              # Pydantic-схеми та tools (Завдання 1)
# ├── agent.py              # LangGraph граф (Завдання 2)
# ├── safety.py             # max_steps, timeout, LoopDetector (Завдання 3)
# ├── logger.py             # TrajectoryLogger (Завдання 3)
# ├── test_runner.py        # Тест-кейси та збір результатів (Завдання 4)
# ├── trajectory.json       # Результат логування (генерується)
# ├── test_results.json     # Результати тестів (генерується)
# ├── requirements.txt      # Залежності
# └── README.md             # Опис, інструкція, аналіз
#
# Або один Jupyter Notebook з чіткими секціями для кожного завдання.



Шаблон README.md (рекомендований)

Скопіюйте структуру нижче як основу для свого README. Замініть плейсхолдери у квадратних дужках вмістом вашого проєкту. Шаблон покриває всі вимоги до README з рубрики оцінювання (Завдання 4, разом 10 балів за README та структуру коду).



# ReAct-агент для [вашої доменної задачі]

## 1. Опис проєкту
Коротко (3-5 речень): який агент побудовано, для якої доменної задачі, на якому LLM-провайдері, які ключові можливості.

## 2. Доменна задача
- **Домен:** [фінанси / подорожі / DevOps / медицина / ...]
- **Що вирішує агент:** які запити користувача він обробляє.
- **Інструменти (tools):**
 - `tool_1` — призначення
 - `tool_2` — призначення
 - `tool_3` — призначення

## 3. Архітектура
Короткий опис: LLM-провайдер, структура графа (agent + tools + conditional edge), застосовані захисні механізми (max_steps = X, timeout = Y сек, LoopDetector з max_repeats = Z).

## 4. Інструкція запуску
1. Клонувати репозиторій: `git clone <repo-url>` (або відкрити Colab за посиланням).
2. Перейти в директорію: `cd hw1_react_agent`.
3. Встановити залежності: `pip install -r requirements.txt`.
4. Налаштувати API-ключ: `export GOOGLE_API_KEY="..."` (або інший провайдер).
5. Запустити агента на тест-кейсах: `python test_runner.py`.

## 5. Результати тестування
Підсумкова таблиця за `test_results.json`:

| Test ID | Складність | Кроків | Tool calls    | Час, мс | Статус    |
|---------|------------|--------|------------------|---------|--------------|
| TC-001 | simple   | 2   | get_weather × 1 | 1200  | ✅ success  |
| TC-002 | medium   | 4   | get_weather × 2 | 2300  | ✅ success  |
| TC-003 | complex  | 7   | get_weather × 3 | 4500  | ⚠️ partial  |

## 6. Аналіз результатів
**Чи правильно агент обирав tools?**
[1-2 абзаци з прикладами правильного та неправильного вибору]

**Де агент помилявся і чому?**
[галюцинації параметрів, неправильний вибір tool, неповна відповідь — наведіть конкретні випадки з trajectory.json]

**Кроки: simple vs complex.**
[порівняння кількості кроків між простими та складними задачами; чи є лінійна залежність від складності]

**Чи досягав агент max_steps або timeout?**
[якщо так — на якому тест-кейсі та чому; якщо ні — наскільки далеко був ліміт]

**Залежність часу виконання від складності.**
[короткий висновок: latency vs complexity]

**Випадки, коли агент НЕ використав tool, хоч мав би.**
[якщо є — приклад запиту, реакція агента та інтерпретація]

## 7. Висновки та обмеження
- Що працює добре: ...
- Що можна покращити: ...
- Відомі обмеження: ...

## 8. Структура файлів
- `tools.py` — Pydantic-схеми та tools
- `agent.py` — LangGraph граф
- `safety.py` — max_steps, timeout, LoopDetector
- `logger.py` — TrajectoryLogger
- `test_runner.py` — тест-кейси
- `trajectory.json` — лог траєкторії (генерується)
- `test_results.json` — результати тестів (генерується)







Підготовка до виконання завдання



Створити Python-проєкт (рекомендовано: окрема директорія з venv або Google Colab).


Встановити залежності:

pip install langgraph langchain langchain-core "pydantic>=2.0"
# Один з LLM-провайдерів (на вибір):
pip install langchain-openai          # для OpenAI GPT-5.2
pip install langchain-google-genai    # для Google Gemini
pip install langchain-mistralai       # для Mistral
pip install langchain-ollama          # для Ollama (локальна модель)

Налаштувати API-ключ обраного провайдера.

import os
 
# Варіант A: OpenAI
os.environ['OPENAI_API_KEY'] = 'sk-...'
 
# Варіант B: Google Gemini
os.environ['GOOGLE_API_KEY'] = 'AIza...'
 
# Варіант C: Mistral
os.environ['MISTRAL_API_KEY'] = '...'
 
# Варіант D: Ollama (не потребує ключа)
# Переконайтеся, що Ollama запущений: ollama serve

Перевірити, що LLM працює.

# Перевірка підключення (приклад для Gemini):
from langchain_google_genai import ChatGoogleGenerativeAI
 
llm = ChatGoogleGenerativeAI(model='gemini-3.8-flash') # для Gemini 3 temperature залишаємо за замовчуванням (1.0)
response = llm.invoke('Say hello in 3 words')
print(response.content)  # Очікуємо: "Hello there, friend!" або подібне





Завдання



Завдання 1. Реалізація інструментів з Pydantic-схемами (20 балів)



Мета: Створити 3–4 інструменти (tools) для доменної задачі з повною Pydantic v2 валідацією. Кожен tool — це Python-функція з декоратором @tool та Pydantic-моделлю для параметрів.



Інструкція:

Оберіть доменну задачу для агента (наприклад: фінансовий калькулятор, туристичний планувальник, аналітик даних, DevOps-асистент, або будь-яка інша).
Створіть 3-4 інструменти з BaseModel для параметрів, Field з description, та field_validator для перевірки вхідних даних.
Кожен tool повинен мати детальний docstring — LLM читає його для розуміння, коли і як використовувати інструмент.
Протестуйте кожен tool окремо (unit test) перед інтеграцією в агента.


Приклад коду (адаптуйте під свою доменну задачу):

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
 
# ── Pydantic-схема для параметрів tool ───────────────────────────
class WeatherInput(BaseModel):
    """Параметри для запиту погоди."""
    city: str = Field(description='Назва міста (наприклад, "Kyiv")')
    days: int = Field(default=1, description='Кількість днів прогнозу (1-7)')
 
    @field_validator('city')
    @classmethod
    def city_not_empty(cls, v: str) -> str:
        if not v or len(v.strip()) < 2:
            raise ValueError('Назва міста повинна містити мінімум 2 символи')
        return v.strip()
 
    @field_validator('days')
    @classmethod
    def days_in_range(cls, v: int) -> int:
        if not 1 <= v <= 7:
            raise ValueError('Кількість днів має бути від 1 до 7')
        return v
 
# ── Tool з Pydantic-схемою ───────────────────────────────────────
@tool(args_schema=WeatherInput)
def get_weather(city: str, days: int = 1) -> str:
    """Отримати прогноз погоди для міста.
 
    Використовуйте цей інструмент, коли потрібна інформація
    про погоду в конкретному місті.
 
    Args:
        city: Назва міста.
        days: Кількість днів прогнозу (1-7).
 
    Returns:
        Рядок з прогнозом погоди.
    """
    # Mock-дані (у реальному проєкті — API-виклик)
    weather_data = {
        'Kyiv': {'temp': 15, 'condition': 'Хмарно'},
        'Lviv': {'temp': 12, 'condition': 'Дощ'},
        'Odesa': {'temp': 18, 'condition': 'Сонячно'},
    }
    data = weather_data.get(city, {'temp': 20, 'condition': 'Ясно'})
    return f'{city}: {data["temp"]}°C, {data["condition"]} (прогноз на {days} дн.)'
 
# ── Unit test ────────────────────────────────────────────────────
# Валідація Pydantic працює:
try:
    WeatherInput(city='', days=1)  # Повинно впасти
except Exception as e:
    print(f'Validation error (expected): {e}')
 
# Tool працює:
result = get_weather.invoke({'city': 'Kyiv', 'days': 3})
print(f'Weather: {result}')



Створіть аналогічні tools для своєї доменної задачі. Мінімум 3 tools, кожен з Pydantic-схемою та field_validator.



Чому Pydantic-схеми важливі для tools?

По-перше, LLM читає Field(description=...) при виборі інструменту — якщо description невиразний, LLM може обрати неправильний tool.
По-друге, field_validator захищає від некоректних вхідних даних: LLM може «галюцинувати» параметри (наприклад, від'ємну ціну або порожнє місто), і валідатор поверне зрозумілу помилку замість crash.
По-третє, Pydantic автоматично генерує JSON Schema, яку LangChain використовує для bind_tools.


Очікуваний формат відповіді для Завдання 1: код створення 3–4 tools з Pydantic-схемами, демонстрація успішних і неуспішних викликів (валідація працює), unit-тести для кожного tool.

___________________________________________________





Завдання 2. LangGraph з циклом LLM–tools–LLM (30 балів)



Мета: Побудувати LangGraph StateGraph з ReAct-циклом: agent (LLM приймає рішення) → tools (виконання обраного tool) → agent (LLM аналізує результат) → ... Цикл продовжується, поки LLM не вирішить відповісти без tool call.



Інструкція:

Визначте State (TypedDict або Pydantic) з полями: messages, step_count.
Створіть вузол agent: LLM з bind_tools отримує messages та вирішує, чи потрібно викликати tool, чи відповісти напряму.
Створіть вузол tools: виконує tool call та додає результат до messages.
Додайте conditional edge (router): якщо LLM повернув tool_calls → tools, інакше → END.
Додайте structured outputs: agent повинен повертати відповідь у структурованому форматі (Pydantic-модель з полями answer, confidence, sources).


Приклад коду:

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
 
# ── State ────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
 
# ── LLM з tools ──────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0.1)
 
# Список ваших tools (з Завдання 1)
tools = [get_weather]  # Додайте всі ваші tools
llm_with_tools = llm.bind_tools(tools)
 
# ── Вузол agent ──────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """LLM приймає рішення: викликати tool чи відповісти."""
    response = llm_with_tools.invoke(state['messages'])
    return {
        'messages': [response],
        'step_count': state.get('step_count', 0) + 1,
    }
 
# ── Router ───────────────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal['tools', '__end__']:
    """Якщо LLM повернув tool_calls → tools, інакше → end."""
    last_msg = state['messages'][-1]
    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
        return 'tools'
    return '__end__'
 
# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node('agent', agent_node)
graph.add_node('tools', ToolNode(tools))
 
graph.add_edge(START, 'agent')
graph.add_conditional_edges('agent', should_continue)
graph.add_edge('tools', 'agent')  # після tools → назад до agent
 
app = graph.compile()
 
# ── Запуск ───────────────────────────────────────────────────────
result = app.invoke({
    'messages': [HumanMessage(content='Яка погода у Києві на 3 дні?')],
    'step_count': 0,
})
 
# Фінальна відповідь:
print(result['messages'][-1].content)
print(f'Steps: {result["step_count"]}')



Зверніть увагу на ключові елементи графа:

StateGraph(AgentState) — визначає тип стану, що передається між вузлами;
bind_tools(tools) — прив'язує інструменти до LLM, щоб модель знала, які tools доступні;
ToolNode(tools) — вбудований вузол LangGraph, що автоматично виконує tool calls з відповіді LLM;
conditional_edges з router-функцією should_continue — це серце ReAct-циклу: якщо LLM повернув tool_calls → йдемо до tools, інакше → завершення.


Очікуваний результат виводу: після завершення ReAct-циклу агент повинен повернути відповідь у структурованому форматі. Це можна реалізувати через додатковий вузол «formatter», який використовує with_structured_output для конвертації вільного тексту у Pydantic-модель (наприклад: answer: str, confidence: float, sources: list[str]). Або через фінальну обробку state["messages"][-1].content.



Очікуваний формат відповіді для завдання 2: працюючий LangGraph з циклом LLM–tools–LLM, демонстрація на 2–3 прикладах, де агент правильно використовує tools та формує структуровану відповідь.

______________________________________________________________________





Завдання 3. Захисні механізми та логування (25 балів)



Мета: Додати до агента захисні механізми, що запобігають зациклюванню та неконтрольованому споживанню ресурсів, а також логування повної траєкторії.



Інструкція:

max_steps: обмежити кількість ітерацій циклу LLM–tools–LLM (рекомендовано: 10–15). Якщо досягнуто ліміту, агент повертає часткову відповідь з повідомленням.
timeout: загальний тайм-аут виконання (рекомендовано: 60–120 секунд). Використовуйте time.monotonic() для відліку.
Детекція зациклення: якщо агент викликає той самий tool з тими самими аргументами 3+ разів поспіль — зупинити виконання.
JSON-логування: записувати кожен крок виконання у JSON-файл: step_number, node_name, input, output, tool_calls, timestamp, duration_ms.


Приклад коду (max_steps та timeout):

import time
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AIMessage
 
# State (з Завдання 2 — повторюємо для зручності)
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    step_count: int
 
MAX_STEPS = 10
TIMEOUT_SECONDS = 120
 
def agent_node_safe(state: AgentState) -> dict:
    """Agent з max_steps та timeout."""
    step = state.get('step_count', 0)
 
    # Max steps check
    if step >= MAX_STEPS:
        from langchain_core.messages import AIMessage
        return {
            'messages': [AIMessage(content=(
                f'Досягнуто ліміт кроків ({MAX_STEPS}). '
                f'Часткова відповідь: оброблено {step} кроків.'
            ))],
            'step_count': step,
        }
 
    response = llm_with_tools.invoke(state['messages'])
    return {
        'messages': [response],
        'step_count': step + 1,
    }



Приклад коду (JSON-логування траєкторії):

import json
import time
from datetime import datetime, timezone
 
class TrajectoryLogger:
    """Логування траєкторії виконання агента."""
 
    def __init__(self):
        self.steps = []
        self.start_time = time.monotonic()
 
    def log_step(self, step_num: int, node: str,
                 input_data: str, output_data: str,
                 tool_calls: list = None):
        self.steps.append({
            'step': step_num,
            'node': node,
            'input': input_data[:500],
            'output': output_data[:500],
            'tool_calls': tool_calls or [],
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'elapsed_ms': int((time.monotonic() - self.start_time) * 1000),
        })
 
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'total_steps': len(self.steps),
                'total_time_ms': int((time.monotonic() - self.start_time) * 1000),
                'trajectory': self.steps,
            }, f, ensure_ascii=False, indent=2)
        print(f'Trajectory saved: {filepath} ({len(self.steps)} steps)')
 
# Використання:
logger = TrajectoryLogger()
logger.log_step(1, 'agent', 'user query', 'tool call: get_weather', ['get_weather'])
logger.log_step(2, 'tools', 'get_weather(Kyiv)', 'Kyiv: 15°C')
logger.save('trajectory.json')



Пояснення: TrajectoryLogger зберігає кожен крок виконання як словник з полями: номер кроку, назва вузла (agent або tools), вхідні та вихідні дані, список tool calls, timestamp та elapsed time. Метод save() записує всю траєкторію у JSON-файл. Це критично для post-mortem аналізу: якщо агент дав неправильну відповідь, ви можете переглянути кожен крок і зрозуміти, де помилка. Як інтегрувати TrajectoryLogger у граф: створіть instance перед запуском, викликайте log_step() у кожному вузлі (agent_node та tool_node), та save() після завершення invoke(). Або використовуйте callback-підхід LangGraph для автоматичного логування. Детекція зациклення: зберігайте останні N tool calls (назва + аргументи). Якщо один і той самий виклик повторюється 3 рази — зупиніть виконання. Це захист від ситуації, коли LLM наполегливо викликає один tool з тими самими параметрами, очікуючи інший результат.





Приклад коду (детекція зациклення):

import hashlib
 
class LoopDetector:
    """Детекція зациклення agent — повторюваних tool calls."""
 
    def __init__(self, max_repeats: int = 3):
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats
 
    def check(self, tool_name: str, args: dict) -> bool:
        """Перевірити, чи виклик повторюється.
 
        Returns:
            True якщо зациклення виявлено.
        """
        call_hash = hashlib.md5(
            f'{tool_name}:{sorted(args.items())}'.encode()
        ).hexdigest()
        self.recent_calls.append(call_hash)
        # Перевірити останні N викликів
        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats:]
            if len(set(last_n)) == 1:  # всі однакові
                return True  # Зациклення!
        return False
 
# Використання у agent_node:
# detector = LoopDetector(max_repeats=3)
# if detector.check(tool_name, tool_args):
#     return {'messages': [AIMessage(content='Loop detected, stopping.')]}



Очікуваний формат відповіді для Завдання 3: agent_node з max_steps та timeout, TrajectoryLogger з збереженням у JSON, LoopDetector, демонстрація кожного механізму (що відбувається при досягненні ліміту, при timeout, при зациклюванні).

_________________________________________________________



Завдання 4. Тестування та збір результатів (25 балів)



Мета: Прогнати агента на 3–5 тест-кейсах та зібрати результати у структурованому JSON-файлі для аналізу.



Інструкція:

Визначте 3-5 тест-кейсів різної складності для вашої доменної задачі: простий (один tool call), середній (2-3 tool calls), складний (потребує планування та кількох tools).
Для кожного тест-кейсу запишіть: input query, очікуваний результат (expected), фактичний результат (actual), кількість кроків, список tool calls, час виконання, чи досягнуто max_steps/timeout.
Збережіть результати у JSON-файл test_results.json.
У README проаналізуйте: чи правильно агент використовував tools? Де він помилився? Як max_steps вплинув на результат?


Приклад коду (збір результатів):

import json
import time
 
# ── Тест-кейси ──────────────────────────────────────────────────
test_cases = [
    {
        'id': 'TC-001',
        'query': 'Яка погода у Києві?',
        'expected': 'Інформація про погоду в Києві',
        'complexity': 'simple',
    },
    {
        'id': 'TC-002',
        'query': 'Порівняй погоду у Києві та Львові',
        'expected': 'Порівняння погоди двох міст',
        'complexity': 'medium',
    },
    {
        'id': 'TC-003',
        'query': 'Сплануй одяг на 3 дні у Одесі враховуючи погоду',
        'expected': 'Рекомендації щодо одягу на основі прогнозу',
        'complexity': 'complex',
    },
]
 
# ── Запуск тестів ────────────────────────────────────────────────
results = []
for tc in test_cases:
    print(f"Running {tc['id']}: {tc['query'][:50]}...")
    start = time.monotonic()
 
    try:
        result = app.invoke({
            'messages': [HumanMessage(content=tc['query'])],
            'step_count': 0,
        })
        actual = result['messages'][-1].content
        steps = result['step_count']
        status = 'success'
        # Зібрати tool calls (всередині try — result існує тільки тут)
        tool_calls = []
        for msg in result.get('messages', []):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc_call in msg.tool_calls:
                    tool_calls.append(tc_call.get('name', 'unknown'))
    except Exception as e:
        actual = f'Error: {e}'
        steps = -1
        status = 'error'
        tool_calls = []
 
    elapsed = (time.monotonic() - start) * 1000
 
    results.append({
        'test_id': tc['id'],
        'query': tc['query'],
        'expected': tc['expected'],
        'actual': actual[:500],
        'status': status,
        'steps': steps,
        'tool_calls': tool_calls,
        'elapsed_ms': int(elapsed),
        'complexity': tc['complexity'],
    })
 
# ── Зберегти результати ─────────────────────────────────────────
with open('test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
 
print(f'Results saved: {len(results)} test cases')
for r in results:
    print(f"  {r['test_id']}: {r['status']} ({r['steps']} steps, {r['elapsed_ms']}ms)")



Пояснення тест-кейсів:

Simple — один tool call (наприклад, «Яка погода у Києві?» → get_weather). Перевіряє базову роботу ReAct-циклу.
Medium — 2–3 tool calls (наприклад, «Порівняй погоду у Києві та Львові» → get_weather двічі). Перевіряє, чи агент планує послідовність дій.
Complex — потребує аналізу та синтезу (наприклад, «Рекомендуй одяг на 3 дні» → get_weather + додаткова логіка). Перевіряє, чи агент використовує результати tools для формування комплексної відповіді.


У README обов'язково проаналізуйте результати: Чи правильно агент обирав tools? Де помилився і чому? Скільки кроків потрібно для simple vs complex задач? Чи досягнув агент max_steps на якомусь тесті — якщо так, чому? Як час виконання залежить від складності? Чи є випадки, де агент не використав tool, хоча повинен був?



Очікуваний формат відповіді для Завдання 4: test_results.json з повними результатами для 3–5 тест-кейсів, аналіз у README (що працює, де помилки, рекомендації).
