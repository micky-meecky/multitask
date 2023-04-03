with open('example.txt', 'r') as file:
    text = file.read()
words = text.split()
print("单词数量为：", len(words))