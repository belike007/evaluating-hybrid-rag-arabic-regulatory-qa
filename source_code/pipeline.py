import ollama
import time
from src.retriever import HybridRetriever
from src.config import (
    LLM_MODEL,
    TEMPERATURE,
    MAX_CONTEXT_CHUNKS,
)
SYSTEM_PROMPT = """
أنت تجيب عن الأسئلة باستخدام النص المعطى فقط.
استخرج الإجابة مباشرة من النص.
لا تستخدم معلومات خارج النص.
إذا وجدت الإجابة، أجب بها مباشرة.
"""
class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
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

    def ask(self, question, verbose=False):
            total_start = time.perf_counter()
            retrieval_start = time.perf_counter()
            results = self.retriever.retrieve(question)
            retrieval_end = time.perf_counter()
            retrieval_time = (retrieval_end - retrieval_start)
            context_start = time.perf_counter()
            context = self.build_context(results)
            context_end = time.perf_counter()
            context_time = (context_end - context_start)
        
            if verbose:
 
                print("CONTEXT SENT TO AYA")
                print(context)
                print( f"Retrieval time:"f"{retrieval_time:.2f} seconds")
                print(f"Context building time:"f"{context_time:.2f} seconds")
            user_prompt = f"""
            السؤال:
            {question}
            النص:
            {context}
            استخرج الإجابة من النص وأجب عن السؤال مباشرة.
            اذكر رقم الصفحة إذا كانت موجودة.
            """
            generation_start = time.perf_counter()
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[ {"role": "system","content": SYSTEM_PROMPT,},
                        {"role": "user","content": user_prompt,},],
                options={"temperature": TEMPERATURE},)
            generation_end = time.perf_counter()
            generation_time = (generation_end - generation_start)
            answer = response["message"]["content"]
            total_end = time.perf_counter()
            total_time = (
                total_end - total_start
            )
            timing = {"retrieval_time_seconds": round(retrieval_time,4),
                      "context_time_seconds": round(context_time,4),
                      "generation_time_seconds": round(generation_time,4),
                      "total_time_seconds": round(total_time,4),}

            if verbose:
                print("TIMING")
                print(f"Retrieval:"f"{retrieval_time:.2f} seconds")
                print(f"Context:"f"{context_time:.2f} seconds")
                print(f"Generation:"f"{generation_time:.2f} seconds")
                print(f"Total:"f"{total_time:.2f} seconds")
            return answer, results,
        
def main():

    pipeline = RAGPipeline()
    print("SAMA FINANCE RAG")
    print("\nType 'exit' to stop.\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        answer, results = pipeline.ask(question,verbose=True) 
        print("ANSWER")
        print(answer)
        print("RETRIEVED SOURCES")
        for i, result in enumerate(results,start=1):
            metadata = result["metadata"]
            print( f"{i}.Page"f"{metadata['page']}"f"| Chunk "f"{metadata['chunk_id']}")
            
if __name__ == "__main__":
    main()
