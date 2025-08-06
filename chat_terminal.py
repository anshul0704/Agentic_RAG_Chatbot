from rag_chain import generate_answer

if __name__ == "__main__":
    print("Ask anything based on pdfs..")
    print("type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        answer=generate_answer(query)
        print("bot: ", answer)
        print()
        


