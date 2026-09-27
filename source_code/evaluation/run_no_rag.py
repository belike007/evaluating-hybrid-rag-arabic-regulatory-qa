import csv
import json
import ollama

from src.config import LLM_MODEL, TEMPERATURE

INPUT_FILE = "data/evaluation/sama_specificity_controlled_30_questions.csv"
OUTPUT_FILE = "data/evaluation/no_rag_specificity_30_results.json"

SYSTEM_PROMPT = """
أنت مساعد متخصص في اللوائح التنظيمية للقطاع المالي.

أجب عن السؤال اعتمادًا على معرفتك فقط.
لا تستخدم أي نصوص مسترجعة أو سياق من وثائق خارجية.

كن مباشرًا ومختصرًا.
لا تسأل المستخدم سؤالًا إضافيًا.
"""


def load_questions():
    questions = []

    with open(INPUT_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            questions.append(row["Question"].strip())

    return questions


def main():
    questions = load_questions()

    print("======================================")
    print("NO-RAG EVALUATION")
    print("======================================")
    print(f"Questions: {len(questions)}")

    results = []

    for i, question in enumerate(questions, start=1):

        print(f"\n[{i}/{len(questions)}]")
        print(question)

        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            options={
                "temperature": TEMPERATURE
            },
        )

        answer = response["message"]["content"]

        results.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        print("Answer:")
        print(answer)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n======================================")
    print("NO-RAG COMPLETE")
    print("======================================")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()