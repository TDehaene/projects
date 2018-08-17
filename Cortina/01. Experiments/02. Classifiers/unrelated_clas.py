import treetaggerwrapper
from treetaggerwrapper import Tag as Tag


class TreeTaggerParser:
    def __init__(self):
        self.tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")
        treetaggerwrapper.make_tags(self.tagger.tag_text("test"))
        self.language = "en"

    def tag(self, line):
        '''
        Tag a sentence with the instantiated tagger method object

        Keyword arguments:
        :param line: (string) The line to parse.

        :return: (list of dict): A list of, per token: the word, the lemma, the pos ud tag and the pos penn tag
        '''

        tags = self.tagger.tag_text(line)
        line_processed = treetaggerwrapper.make_tags(tags)
        line_return = [
            {"word": tag.word, "lemma": tag.lemma if not tag.lemma == "@card@" else tag.word, "pos_penn": tag.pos}
            for tag in line_processed if (type(tag) == Tag)]

        return line_return

parser = TreeTaggerParser()

for i in range(1000000):
    parser.tag("I am a boy")