import csv
import json

from src.pipeline import RAGPipeline


INPUT_FILE = "data/evaluation/sama_specificity_controlled_30_questions.csv"
OUTPUT_FILE = "data/evaluation/rag_specificity_30_results.json"

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
    print("RAG EVALUATION")
    print("======================================")
    print(f"Questions: {len(questions)}")

    pipeline = RAGPipeline()

    results = []

    for i, question in enumerate(questions, start=1):

        print(f"\n[{i}/{len(questions)}]")
        print(question)

        answer, retrieved_results = pipeline.ask(
            question,
            verbose=False
        )

        context = pipeline.build_context(retrieved_results)

        retrieved_sources = []

        for result in retrieved_results:
            retrieved_sources.append({
            
                "text": result.get("text")
            })

        results.append({
            "question": question,
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "context": context
        })

        print("\nAnswer:")
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
            indent=2
        )

    print("\n======================================")
    print("RAG EVALUATION COMPLETE")
    print("======================================")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()