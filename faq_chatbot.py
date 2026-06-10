import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("faq.csv")

questions = data["question"]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot Started")
print("Type exit to stop")

while True:
    user_question = input("You: ")

    if user_question.lower() == "exit":
        break

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    best_match = similarity.argmax()

    print(
        "Bot:",
        data.iloc[best_match]["answer"]
    )