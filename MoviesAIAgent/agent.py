from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about movies

Here are some relevant movie descriptions: {descriptions}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n")
    print("-------------------------------------------------")
    question = input("Ask me something about movies (type q to quit):")
    if question == "q":
        break

    print("\n\n")
    print("Response:\n\n")
    
    descriptions = retriever.invoke(question)
    result = chain.invoke({"descriptions": descriptions, "question": question})
    print(result)