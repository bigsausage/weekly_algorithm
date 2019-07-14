class Node():
    def __init__(self):
        self.next = {}
        self.isWord = False
        self.fail = None

class Ahocorasick():
    def __init__(self):
        self.__root = Node()

    def addWord(self,word):
        tmpNode = self.__root
        for char in word:
            if char not in tmpNode.next:
                tmpNode.next[char] = Node()
            tmpNode = tmpNode.next[char]
        if tmpNode != self.__root:
            tmpNode.isWord = True

    def make(self):
        tmpQueue = []
        tmpQueue.append(self.__root)
        while(len(tmpQueue)>0):
            tmpNode = tmpQueue.pop()
            for key,value in tmpNode.next.items():
                if tmpNode == self.__root:
                    tmpNode.next[key].fail = self.__root
                else:
                    p = tmpNode.fail
                    while p is not None:
                        if key in p.next:
                            tmpNode.next[key].fail = p.next[key]
                            break
                        p = p.fail
                    if p is None:
                        tmpNode.next[key].fail = self.__root

                tmpQueue.append(tmpNode.next[key])

    def search(self, text):
        tmpNode = self.__root
        currentPosition = 0
        matchStart = 0
        res = []

        while (currentPosition < len(text)):
            word = text[currentPosition]

            while(word not in tmpNode.next) and tmpNode != self.__root:
                tmpNode = tmpNode.fail

            if word in tmpNode.next:
                if tmpNode == self.__root:
                    matchStart = currentPosition
                tmpNode = tmpNode.next[word]

            else:
                tmpNode = self.__root

            if tmpNode.isWord:
                res.append((matchStart,currentPosition))

            currentPosition+=1

        return res


ahocorasick = Ahocorasick()
ahocorasick.addWord("今天")
ahocorasick.addWord("过年")
ahocorasick.addWord("过年假")
ahocorasick.make()
text = "为什么今天天气真好，大家过来过年假"
results = ahocorasick.search(text)

print(results)
for result in results:
    print(text[result[0]:result[1]+1])

