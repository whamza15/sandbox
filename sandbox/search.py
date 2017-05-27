
if __name__ == "__main__":
    import sys
    from optparse import OptionParser
    from collections import Counter
    import math

    parser = OptionParser()
    parser.add_option("-i", '--document-text', dest="documentsTextFileName", type="string")
    parser.add_option("-p", '--index-prefix', dest="indexPrefix", type="string")
    parser.add_option("-l", "--lower-case", dest="lc", action="store_true")
    parser.add_option("-n", "--top-n", dest="topN", type="int", default=5)
    options, args = parser.parse_args()

    if options.documentsTextFileName is None:
        print >>sys.stderr, "ERROR: documents text file not must be specified using -i"
        sys.exit(1)
    if options.indexPrefix is None:
        print >>sys.stderr, "ERROR: Index prefix must be specified using -p (--index-prefix)"
        sys.exit(1)

    # loading vocabulary
    with open("%s.vocab" % options.indexPrefix) as f:
        words = [l.strip() for l in f.readlines()]

    wordVocab = dict([(w, i) for i, w in enumerate(words)])

    # loading idf
    with open("%s.idf" % options.indexPrefix) as f:
        idf = [float(l.strip()) for l in f.readlines()]

    # loading tf/idf
    docsTfIdfs = []
    with open("%s.tfidf" % options.indexPrefix) as f:
        for l in f.readlines():
            toks = l.strip().split()
            docsTfIdfs.append(dict(zip(map(int, toks[0::2]), map(float, toks[1::2]))))

    # Loading title and body of all documents (for display)

    def doc_generator(sin):
        lines = []
        while True:
            line = sin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                yield lines
                lines = []
            else:
                lines.append(line)
    docTexts = []
    with open(options.documentsTextFileName) as f:
        for docLines in doc_generator(f):
            docTexts.append(docLines)

    def cosine(v1, v2):
        d2 = dict(v2)
        s = 0
        for i, tv1 in v1:
            try:
                tv2 = d2[i]
                s += tv1 * tv2
            except KeyError:
                pass
        return s

    def getTfIdf(voc, idfs, query_words):
        word_and_count = Counter(query_words).items()
        v = sorted([(voc[word], count * idf[voc[word]]) for word, count in word_and_count if word in voc],
                      key=lambda x: x[0])
        norm = math.sqrt(sum([t*t for _, t in v]))
        v = map(lambda x: (x[0], x[1]/norm), v)
        return v

    while True:

        q = raw_input("Enter Query")
        if options.lc:
            qwords = q.lower().strip().split()
        else:
            qwords = q.strip().split()
        qTfIdf = getTfIdf(wordVocab, idf, qwords)
        topN = sorted([(i, cosine(qTfIdf, dTfIdf)) for i, dTfIdf in enumerate(docsTfIdfs)],
                      key=lambda x: x[1],
                      reverse=True)[:options.topN]
        print "\n==\n".join(["%.3f - %s\n%s" % (sc, docTexts[d][0], docTexts[d][1]) for d, sc in topN])
