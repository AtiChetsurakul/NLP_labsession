from sys import argv
import requests
from bs4 import BeautifulSoup
import pickle

assert len(argv) == 4
'''To use need 3 argument, a url, a file to store name and class of html name'''


def get_data():
    # URL = 'https://towardsdatascience.com/web-scraping-basics-82f8b5acd45c'
    URL = argv[1]
    response = requests.get(URL)
    website_html = response.text
    soup = BeautifulSoup(website_html, "html.parser")
    all_paragraph = soup.find_all(name="p", class_=argv[3])
    get_only_text = [para.getText() for para in all_paragraph]
    # print(len(get_only_text)) # 40 quite enough
    my_corpus = [text[:80] for text in get_only_text if len(text) > 40]
    with open(argv[2], 'wb') as tostore:
        pickle.dump(my_corpus, tostore)
