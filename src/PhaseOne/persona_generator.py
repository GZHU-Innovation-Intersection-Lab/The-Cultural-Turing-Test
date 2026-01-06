from time import sleep
from typing import List, Dict, Optional

import httpx
import pandas as pd
from openpyxl import load_workbook, Workbook
from openai import OpenAI
import os
import re


def create_client(api_key: str, base_url: str, proxy_url: Optional[str] = None) -> OpenAI:
    http_client = None
    if proxy_url:
        transport = httpx.HTTPTransport(proxy=proxy_url)
        http_client = httpx.Client(transport=transport)
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


def extract_numeric_answers(text: str, max_count: int = 24) -> List[str]:
    patterns = [
        r"Q\d.*?[:：]\s*(\d+)",
        r"A\d.*?[:：]\s*(\d+)",
        r"Answer+.*?[:：]\s*(\d+)",
        r"Q\d+[:：](\d+)",
        r"\*\*Q\d+\. .*?\n(\d+)\. ",
        r"\*\*(\d+)\.",
        r"Choice[:：]\s*(\d+)\.",
        r"\*\*Q\d+\.\s+.*?\n(\d+)",
        r"→ (\d+)\.",
        r"\*\*Q\d+[:：](\d+)",
        r"\*\*Q\d+\*\*[:：](\d+)",
    ]
    for pat in patterns:
        answers = re.findall(pat, text)
        if answers:
            return answers[:max_count]
    return []


def read_xlsx_questions(xlsx_path: str, min_row: int, max_col: int) -> List[str]:
    wb = load_workbook(filename=xlsx_path)
    ws = wb.active
    questions: List[str] = []
    for row in ws.iter_rows(min_row=min_row, max_col=max_col, max_row=ws.max_row, values_only=True):
        question = row[0] if row[0] is not None else ""
        for i in range(1, max_col):
            if row[i] is not None:
                question += str(i) + '.' + str(row[i])
        if question:
            questions.append(str(question))
    wb.close()
    return questions


def estimate_token_count(text: str) -> int:
    return len(text.split())


def trim_messages(messages: List[Dict[str, str]], max_tokens: int = 6144) -> List[Dict[str, str]]:
    total_tokens = sum(estimate_token_count(m.get('content', '')) for m in messages)
    while total_tokens > max_tokens and len(messages) > 1:
        removed = messages.pop(0)
        total_tokens -= estimate_token_count(removed.get('content', ''))
    return messages


def call_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], temperature: float = 1.3, max_tokens: int = 8192) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False
    )
    return resp.choices[0].message.content.strip()


def _load_prompt_or_role(path: str, row_number: int) -> str:
    if path.lower().endswith('.xlsx'):
        wb = load_workbook(filename=path)
        ws = wb.active
        row_number = min(max(1, row_number), ws.max_row)
        cell_value = ws.cell(row=row_number, column=1).value
        wb.close()
        return str(cell_value).strip() if cell_value is not None else ""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_persona_generation(
    prompt_txt_path: str,
    roles_xlsx_path: str,
    questions_xlsx_path: str,
    out_xlsx_path: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    proxy_url: Optional[str] = None,
    n_rows: int = 40,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    api_key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.gptsapi.net/v1").strip()
    if not api_key:
        raise ValueError("Missing API key. Set OPENAI_API_KEY or pass api_key.")

    client = create_client(api_key=api_key, base_url=base_url, proxy_url=proxy_url)
    questions = ["Please generate a persona"] + read_xlsx_questions(questions_xlsx_path, 2, 6)

    df_columns = [f"Q{i + 1}" for i in range(24)]
    df = pd.DataFrame(columns=df_columns)

    for window_id in range(1, n_rows + 1):
        prompt = _load_prompt_or_role(prompt_txt_path, window_id)
        role_to_generate = _load_prompt_or_role(roles_xlsx_path, window_id)

        messages_history = [
            {"role": "system", "content": prompt},
            {"role": "system", "content": "The role to generate is: " + role_to_generate},
        ]

        q = " ".join(questions[:1])
        messages_history.append({"role": "user", "content": q})
        trim = trim_messages(messages_history.copy())
        try:
            response_text = call_chat(client, model=model, messages=trim)
        except Exception:
            response_text = "Error"

        row = [response_text] + [""] * (len(df_columns) - 1)
        df = pd.concat([df, pd.DataFrame([row], columns=df_columns)], ignore_index=True)

        if sleep_seconds:
            sleep(sleep_seconds)

    os.makedirs(os.path.dirname(out_xlsx_path) or ".", exist_ok=True)
    if os.path.exists(out_xlsx_path):
        with pd.ExcelWriter(out_xlsx_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            sheet_name = 'Sheet1'
            startrow = writer.sheets[sheet_name].max_row if sheet_name in writer.sheets else 0
            df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow)
    else:
        df.to_excel(out_xlsx_path, index=False)

    return df

