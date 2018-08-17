import treetaggerwrapper
from treetaggerwrapper import Tag as Tag

tagger = treetaggerwrapper.TreeTagger(TAGLANG="en")


class TestParser:
    def test_ner_tagging_ingr_ex1(self):
        for i in range(10000000):
            tags = tagger.tag_text("I am a boy")
            line_processed = treetaggerwrapper.make_tags(tags)
            line_return = [
                {"word": tag.word, "lemma": tag.lemma if not tag.lemma == "@card@" else tag.word, "pos_penn": tag.pos}
                for tag in line_processed if (type(tag) == Tag)]
