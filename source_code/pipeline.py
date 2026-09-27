import ollama

from src.retriever import HybridRetriever
from src.config import (
    LLM_MODEL,
    TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
)


# =========================================================
# PROMPT
# =========================================================

SYSTEM_PROMPT = """
أنت تجيب عن الأسئلة باستخدام النص المعطى فقط.
استخرج الإجابة مباشرة من النص.
لا تستخدم معلومات خارج النص.
إذا وجدت الإجابة، أجب بها مباشرة.
"""


# =========================================================
# PIPELINE
# =========================================================

class RAGPipeline:

    def __init__(self):

        self.retriever = HybridRetriever()

    # =====================================================
    # BUILD CONTEXT
    # =====================================================
    def build_context(self, results):

        context_parts = []

        for i, result in enumerate(results, start=1):

            metadata = result["metadata"]

            context_parts.append(
                f"""
    [المصدر {i}]
    الصفحة: {metadata['page']}
    رقم المقطع: {metadata['chunk_id']}

    {result['text']}
    """
            )

        return "\n".join(context_parts)

    # =====================================================
    # ASK
    # =====================================================

   


    def ask(self, question, verbose=False):

            import time

            # ================================================
            # TOTAL TIMER
            # ================================================

            total_start = time.perf_counter()

            # ================================================
            # RETRIEVAL
            # ================================================

            retrieval_start = time.perf_counter()

            results = self.retriever.retrieve(
                question
            )

            retrieval_end = time.perf_counter()

            retrieval_time = (
                retrieval_end - retrieval_start
            )

            # ================================================
            # CONTEXT BUILDING
            # ================================================

            context_start = time.perf_counter()

            context = self.build_context(
                results
            )

            context_end = time.perf_counter()

            context_time = (
                context_end - context_start
            )

            if verbose:

                print("\n======================================")
                print("CONTEXT SENT TO AYA")
                print("======================================")

                print(context)

                print("======================================")

                print(
                    f"Retrieval time: "
                    f"{retrieval_time:.2f} seconds"
                )

                print(
                    f"Context building time: "
                    f"{context_time:.2f} seconds"
                )

            # ================================================
            # PROMPT
            # ================================================

            user_prompt = f"""
            السؤال:
            {question}

            النص:
            {context}

            استخرج الإجابة من النص وأجب عن السؤال مباشرة.
            اذكر رقم الصفحة إذا كانت موجودة.
            """

            # ================================================
            # AYA GENERATION
            # ================================================

            generation_start = time.perf_counter()

            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                options={
                    "temperature": TEMPERATURE
                },
            )

            generation_end = time.perf_counter()

            generation_time = (
                generation_end - generation_start
            )

            answer = response["message"]["content"]

            # ================================================
            # TOTAL TIME
            # ================================================

            total_end = time.perf_counter()

            total_time = (
                total_end - total_start
            )

            # ================================================
            # TIMING INFORMATION
            # ================================================

            timing = {
                "retrieval_time_seconds": round(
                    retrieval_time,
                    4
                ),
                "context_time_seconds": round(
                    context_time,
                    4
                ),
                "generation_time_seconds": round(
                    generation_time,
                    4
                ),
                "total_time_seconds": round(
                    total_time,
                    4
                ),
            }

            if verbose:

                print("\n======================================")
                print("TIMING")
                print("======================================")

                print(
                    f"Retrieval:  "
                    f"{retrieval_time:.2f} seconds"
                )

                print(
                    f"Context:    "
                    f"{context_time:.2f} seconds"
                )

                print(
                    f"Generation: "
                    f"{generation_time:.2f} seconds"
                )

                print(
                    f"Total:       "
                    f"{total_time:.2f} seconds"
                )

            return answer, results,
        


# =========================================================
# MAIN
# =========================================================

def main():

    pipeline = RAGPipeline()

    print("\n======================================")
    print("SAMA FINANCE RAG")
    print("======================================")

    print("\nType 'exit' to stop.\n")

    while True:

        question = input(
            "Question: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer, results = pipeline.ask(
            question,
            verbose=True
            
        )

        print("\n======================================")
        print("ANSWER")
        print("======================================")

        print(answer)

        print("\n======================================")
        print("RETRIEVED SOURCES")
        print("======================================")

        for i, result in enumerate(
            results,
            start=1
        ):

            metadata = result["metadata"]

            print(
                f"{i}. Page "
                f"{metadata['page']} "
                f"| Chunk "
                f"{metadata['chunk_id']}"
            )


if __name__ == "__main__":
    main()