from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt
import instaloader
import re
my_list = []


# Initialize Instaloader
L = instaloader.Instaloader()

username=input("Enter Your Instagram User Name:- ")
password=input("Enter Your Instagram Password:- ")

try:
    L.login(username, password)
    print("\nLogin Sucessful")
except Exception as e:
    print(f"\nFailed to login: {e}")
    exit(1)

# Function to extract shortcode from URL
def extract_shortcode(url):
    match = re.search(r'instagram\.com/p/([^/]+)/', url)
    if match:
        print("\nShort code generated sucessfully")
        return match.group(1)
    else:
        raise ValueError("\nInvalid URL format")
        exit(1)

# Replace this with your actual Instagram post URL
post_url = input("Enter Post URL:-")
try:
    shortcode = extract_shortcode(post_url)
    print(f"\nExtracted shortcode: {shortcode}")

    # Load the post using the shortcode
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    
    # Print the comments
    print(f'Post: {post.shortcode}')
    comments = list(post.get_comments())
    if comments:
        for comment in comments:
           my_list.append(comment.text)
    else:
        print("No comments found on this post.")
        exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
#----------------------------------------------------------------------------------------------------
L.close()
print("Logged out successfully.")
#print(my_list)

# Load the pre-trained emotion classification model and tokenizer
model_name = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define categories (These are the emotions this model can classify)
categories = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# Function to predict emotion scores
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the probabilities for each emotion
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores[0].tolist()

# Example comments only while checking model 
'''comments = [
    "I love this product!",
    "You are a horrible person.",
    "This is the best day ever.",
    "I hate you so much.",
    "Amazing work! Keep it up.",
    "This is terrible and you should feel bad.",
    "I'm subprised!",
    "This makes me feel anxious."
]'''

# Analyze and gather scores for each comment
all_scores = {category: 0 for category in categories}

for comment in my_list:
    scores = predict_emotions(comment)
    for category, score in zip(categories, scores):
        all_scores[category] += score

# Calculate the percentage for each category
total_comments = len(my_list)
percentage_scores = {category: (score / total_comments) * 100 for category, score in all_scores.items()}

# Convert results to a DataFrame for better visualization
df = pd.DataFrame(list(percentage_scores.items()), columns=['Category', 'Percentage'])

# Print the results
print(df)

# Plot the results in a pie chart
plt.figure(figsize=(8, 8))
plt.pie(df['Percentage'], labels=df['Category'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(categories))))
plt.title('Percentage Distribution of Emotions in Comments')
plt.show()


