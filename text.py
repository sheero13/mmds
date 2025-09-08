import re
import string
import nltk

# Download NLTK resources (run this only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define stopwords and lemmatizer outside the function for efficiency
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercasing and removing numbers
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    
    # Removing punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    words = word_tokenize(text)
    
    # Removing stopwords and lemmatization
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return ' '.join(words)

data = ["Python is GREAT!!!", "NLP is used for AI & Data Science."]
cleaned_data = [preprocess(t) for t in data]
print(cleaned_data)
