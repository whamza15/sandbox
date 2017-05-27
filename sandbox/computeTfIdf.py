
if __name__ == "__main__":
    import sys
    from optparse import OptionParser
    from collections import Counter
    import math

    parser = OptionParser()
    parser.add_option("-i", '--input', dest="inputFileName", type="string")
    parser.add_option("-o", '--output-prefix', dest="outputPrefix", type="string")
    parser.add_option("-l", "--lower-case", dest="lc", action="store_true")
    options, args = parser.parse_args()

    if options.inputFileName is None:
        print >>sys.stderr, "ERROR: -i (--input) must be specified"
        sys.exit(1)

    if options.outputPrefix is None:
        print >>sys.stderr, "ERROR: -o (--output-prefix) must be specified"
        sys.exit(1)


    def document_word_generator(ins, lc):
        doc_words = []
        while True:
            l = ins.readline()
            if not l:
                break

            l = l.strip()
            if not l:
                yield doc_words
                doc_words = []
            else:
                if lc:
                    l = l.lower()
                doc_words += l.split()

    freqCounter = Counter([])
    idfCounter = Counter([])
    documentsTf = []
    with open(options.inputFileName) as f:
        idx = 0
        for words in document_word_generator(f, options.lc):
            freqCounter.update(words)
            idfCounter.update(set(words))
            documentsTf.append(Counter(words).items())

            idx += 1
            if idx % 100 == 0:
                print "INFO: processed %d documents" % idx

        print "INFO: finished processing %d documents" % idx

    N = idx
    # building vocabulary based on frequency
    vocabWords = map(lambda y: y[0], sorted(freqCounter.items(), key=lambda x: x[1], reverse=True))
    vocabMap = dict([(w, i) for i, w in enumerate(vocabWords)])

    # building idf for each word
    wordIdf = []
    for word in vocabWords:
        df = idfCounter[word]
        wordIdf.append(math.log(float(N) / df))

    # building the tf.idf for all docs
    documentsTfIdf = []
    for docTf in documentsTf:
        v = sorted([(vocabMap[w], tf*wordIdf[vocabMap[w]]) for w, tf in docTf], key=lambda x: x[0])
        norm = math.sqrt(sum([t*t for i, t in v]))
        v = map(lambda x: (x[0], x[1]/norm), v)
        documentsTfIdf.append(v)

    # writing files
    # 1- vocab
    with open("%s.vocab" % options.outputPrefix, "w") as f:
        print >>f, "\n".join(vocabWords)

    # 2- idf
    with open("%s.idf" % options.outputPrefix, "w") as f:
        print >>f, "\n".join(["%.3f" % idf for idf in wordIdf])

    # 3- tf.idf vectors
    with open("%s.tfidf" % options.outputPrefix, "w") as f:
        print >>f, "\n".join([" ".join(["%s %f" % (wid, tfidf)
                                        for wid, tfidf in docTfIdf]) for docTfIdf in documentsTfIdf])

    # 3- tf.idf vectors (strings)
    with open("%s.tfidf.txt" % options.outputPrefix, "w") as f:
        print >>f, "\n".join([" ".join(["%s %f" % (vocabWords[wid], tfidf)
                                        for wid, tfidf in docTfIdf]) for docTfIdf in documentsTfIdf])
