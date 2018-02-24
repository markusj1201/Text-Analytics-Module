# Basic Text Processing Functions

# Configuration

import spacy, re, gensim, numpy as np, pandas as pd, json, codecs, pyLDAvis, warnings
from gensim.models import Phrases, Word2Vec
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import LineSentence
from collections import Counter
from itertools import chain
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from string import punctuation
punctuation = unicode(punctuation)+u'1234567890'
try:
    nlp = spacy.load('en')
    STOP_WORDS = spacy.en.language_data.STOP_WORDS
except:
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    STOP_WORDS = nlp.Defaults.stop_words
def _try_iter_(iterable, ind, errval=None):
    try:
        return iterable[ind]
    except:
        return errval

# default settings
batch_size, n_threads, fpathroot, fpathappend, entity_sub, numtopics= 1000,32,u'',u'', False, 10

def _remove_punc_(token, punc = punctuation):
    return u''.join([(c if c not in punc else u' ') for c in token]).strip()
def _config_text_analysis_(fpath):
    """
    read in configuration file for text analysis
    """
    with codecs.open(fpath, 'r', encoding = 'utf-8') as f:
        conf = json.loads(f.read())
        batch_size = _try_iter_(conf, 'batch_size')
        n_threads = _try_iter_(conf, 'n_threads')
        fpathroot = _try_iter_(conf, 'fpathroot')
        fpathappend = _try_iter_(conf, 'fpathappend', errval= u'')
        entity_sub = _try_iter_(conf, 'entity_sub', errval= False)
        numtopics = _try_iter_(conf, 'numtopics', errval = 10)
    return batch_size, n_threads, fpathroot, fpathappend, entity_sub, numtopics


# with codecs.open('./default.cfg', 'w', encoding = 'utf-8') as f:
#     f.write(json.dumps({'batch_size':1000, 'n_threads':8, 'fpathroot':u'~/', 'fpathappend':u''}))

##### Functions for single texts

def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space

def stopword_remove(token, sw=STOP_WORDS):
    if sw != False:
        return token.lemma_ not in sw
    else:
        True

def line_doc(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """
    with codecs.open(filename, encoding='utf_8') as f:
        for txt in f:
            yield txt.replace('\\n', '\n')

def sub_entity(token, sublist=True, lemmatize=True):
    """
    Substitutes tokens for entities at token level
    Multiword entities reduced to single entity type
    """
    if sublist == True:
        if token.ent_type_ != u'':
            if token.ent_iob != 3:
                return u''
            else:
                return token.ent_type_
        elif lemmatize==False:
            return token.lower_
        else:
            return token.lemma_
    else:
        if token.ent_type_ in sublist:
            if token.ent_iob != 3:
                return u''
            else:
                return token.ent_type_
        elif lemmatize==False:
            return token.lower_
        else:
            return token.lemma_


### Sentence processing
### File must must be in one document per line format
def lemmatized_sentence_corpus(filename, entity_sub = entity_sub, lemmatize= True, batch_size = batch_size, n_threads = n_threads, sw = STOP_WORDS):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """
    if entity_sub in {False, 'False','false','f',None,'F'}:
        entity_sub=False
    elif entity_sub in {True, 'True', 'true', 't', 'T'}:
        entity_sub = True
    elif type(entity_sub) in {list, set}:
        entity_sub=set(entity_sub)
    if entity_sub == False:
        for parsed_txt in nlp.pipe(line_doc(filename),
                                      batch_size=batch_size, n_threads=n_threads):

            for sent in parsed_txt.sents:
                yield u' '.join(list(chain(*[_remove_punc_(token.lemma_).split() for token in sent
                                 if (not punct_space(token))&(stopword_remove(token, sw = sw))])))
    else:
        for parsed_txt in nlp.pipe(line_doc(filename),
                                      batch_size=batch_size, n_threads=n_threads):

            for sent in parsed_txt.sents:
                yield u' '.join(list(chain(*[_remove_punc_(sub_entity(token, sublist=entity_sub, lemmatize=lemmatize)).split() for token in sent
                                 if (not punct_space(token))&(stopword_remove(token, sw = sw))])))

def _write_unigram_(txt_filepath, unigram_sentences_filepath=fpathroot+fpathappend+'_sent_gram_0.txt', args='default', entity_sub=entity_sub, keywordlist = False, sw = STOP_WORDS):
    """
    Creates a unigram, parsed version of texts at the sentence level.
    """
    if args == 'default':
        streamingfile = lemmatized_sentence_corpus(txt_filepath, entity_sub=entity_sub, sw=sw)
    else:
        streamingfile = lemmatized_sentence_corpus(txt_filepath, **args)

    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf-8') as f:
        for sentence in streamingfile:
            if keywordlist != False:
                for wd in keywordlist:
                    if wd in sentence:
                        f.write(sentence + '\n')
                        break
                    else:
                        pass
            else:
                f.write(sentence + '\n')
    return 'Success'

#### Phrase modeling and predicting
def _phrase_detection_(fpath=fpathroot+fpathappend, passes = 2, returnmodels = True,threshold=10.):
    """
    This function does pharse modeling. User specifies the number of passes.
    Each pass detects longer phrases. The maximum detectable phrase length for
    each pass, n, is 2*n.

    Returns the list of models by default. Also saves models and intermediary
    phrased sentences for each pass.
    """
    generpath = fpath+'_sent_gram_0.txt'
    ngram = list()
    for it in range(passes):
        gen = LineSentence(generpath)
        gram=Phrases(gen, threshold = threshold)
        ngram.append(gram)
        modelpath = fpath+'phrase_model_gram_'+str(it+1)
        generpath = fpath+'sent_gram_'+str(it+1)+'.txt'
        gram.save(modelpath)
        # Write sentence gram
        with codecs.open(generpath, 'w', encoding='utf_8') as f:
            for sent in gen:
                new_sent = u' '.join(gram[sent])
                f.write(new_sent + '\n')

    if returnmodels == True:
        return ngram

def lemmatized_review(parsed_txt, entity_sub = entity_sub, lemmatize= True, keywordlist=False):
    """
    Helper function for parsing documents
    """
    if entity_sub in {False, 'False','false','f',None,'F'}:
        entity_sub=False
    elif entity_sub in {True, 'True', 'true', 't', 'T'}:
        entity_sub = True
    elif type(entity_sub) in {list, set}:
        entity_sub=set(entity_sub)
    if entity_sub == False:
        out = list()
        for sent in parsed_txt.sents:
            if keywordlist != False:
                for wd in keywordlist:
                    if wd in unicode(sent):
                        out.append(list(chain(*[_remove_punc_(token.lemma_).split() for token in sent if not punct_space(token)])))
                        break
                    else:
                        pass
            else:
                out.append(list(chain(*[_remove_punc_(token.lemma_).split() for token in sent if not punct_space(token)])))
        return list(chain(*out))
    else:
        out = list()
        for sent in parsed_txt.sents:
            if keywordlist != False:
                for wd in keywordlist:
                    try:
                        if wd in unicode(sent):
                            out.append(list(chain(*[_remove_punc_(sub_entity(token, sublist=entity_sub, lemmatize=lemmatize)).split()
                                for token in sent if not punct_space(token)])))
                            break
                        else:
                            pass
                    except:
                        out.append(list(chain(*[_remove_punc_(sub_entity(token, sublist=entity_sub, lemmatize=lemmatize)).split()
                            for token in sent if not punct_space(token)])))
            else:
                out.append(list(chain(*[_remove_punc_(sub_entity(token, sublist=entity_sub, lemmatize=lemmatize)).split()
                    for token in sent if not punct_space(token)])))
        return list(chain(*out))


def _phrase_prediction_(fpath, grams, outfpath = None, entity_sub = entity_sub, lemmatize=True, stopwords=STOP_WORDS, keywordlist=False):
    """
    This function takes an input fpath and phrase model list (either model or path)
    for documents and outputs a lemmatized, phrased, and stopword removed version
    of the original documents.

    Returns output file path.
    """
    if type(grams) == list:
        if type(grams[0]) == str:
            load = True
        else:
            load = False
    else:
        if type(grams) == str:
            load = True
        else:
            load = False
        grams = [grams]
    if outfpath == None:
        outfpath = pathroot+pathappend+'_doc_ngram_'+str(len(grams))+'.txt'
    with codecs.open(outfpath, 'w', encoding='utf_8') as f:
        for parsed_txt in nlp.pipe(line_doc(fpath),
                                      batch_size=batch_size, n_threads=n_threads):
            txt_gram = lemmatized_review(parsed_txt, entity_sub=entity_sub, lemmatize=lemmatize, keywordlist=keywordlist)
            for gram in grams:
                txt_gram = gram[txt_gram]
            # remove any remaining stopwords
            txt_gram = [term for term in txt_gram
                              if term not in stopwords]
            # write the transformed review as a line in the new file
            txt_gram = u' '.join(txt_gram)
            f.write(txt_gram + u'\n')
    return outfpath
def _phrase_prediction_inmemory_(texts, grams, entity_sub = entity_sub, lemmatize=True, stopwords=STOP_WORDS, keywordlist=False):
    """
    This function takes an input list of texts and phrase model list (either model or path)
    for documents and outputs a lemmatized, phrased, and stopword removed version
    of the original documents.

    Returns output as list.
    """
    if type(grams) == list:
        if type(grams[0]) == str:
            load = True
        else:
            load = False
    else:
        if type(grams) == str:
            load = True
        else:
            load = False
        grams = [grams]

    output = list()

    for parsed_txt in nlp.pipe(texts,
                                  batch_size=batch_size, n_threads=n_threads):
        txt_gram = lemmatized_review(parsed_txt, entity_sub=entity_sub, lemmatize=lemmatize, keywordlist=keywordlist)
        for gram in grams:
            txt_gram = gram[txt_gram]
        # remove any remaining stopwords
        txt_gram = [term for term in txt_gram
                          if term not in stopwords]
        # write the transformed review as a line in the new file
        txt_gram = u' '.join(txt_gram)
        output.append(txt_gram)
#         f.write(txt_gram + u'\n')
    return output

#### Functions for LDA ####
def _make_dict_(fpath, topfilter = 95,bottomfilter =15,no_filters=True ,keep_ent=False,keep_list = {}, discard_list={},floc = fpathroot+fpathappend+'dict_gram.dict'):
    """
    This function creates the dictionary object in Gensim.
    Returns vocab set and gensim_dictionary in memory.
    """
    gram_sentences = LineSentence(fpath)
    cts = Counter()
    for s in gram_sentences:
        cts.update(s)
    occ = cts.values()
    topfilter = np.percentile(occ, topfilter)
    if no_filters == True:
        dicts = dict([it for it in cts.items() if (it[1]>bottomfilter)&(it[1]<topfilter)])
    elif keep_ent == True: # This is optional/useless if entities were not converted
        dicts = dict([it for it in cts.items() if (((it[1]>bottomfilter)&(it[1]<topfilter))|((('_' in it[0])|(len(re.findall('[A-Z]+',it[0]))>0))|(it[0] in keep_list)))&(it[0] not in discard_list)])
    else:
        dicts = dict([it for it in cts.items() if (((it[1]>bottomfilter)&(it[1]<topfilter))|((('_' in it[0])&(len(re.findall('[A-Z]+',it[0]))==0))|(it[0] in keep_list)))&(it[0] not in discard_list)])
    vocab = set(dicts.keys())
    dictionary = [[v] for v in list(vocab)]
    gensim_dictionary = Dictionary(dictionary)
    gensim_dictionary.compactify()
    gensim_dictionary.save(floc)
    print "Success"
    return vocab, gensim_dictionary, cts

def _bow_generator_(fpath, gensim_dictionary):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """
    for txt in LineSentence(fpath):
        yield gensim_dictionary.doc2bow(txt)

def _serialize_corpus_(fpath, dic, outfpath = fpathroot+fpathappend+'_serialized.mm', returncorp = True):
    """
    create serialized corpus
    """
    MmCorpus.serialize(outfpath, _bow_generator_(fpath, dic))
    if returncorp == True:
        return MmCorpus(outfpath)



def _lda_(gensim_dictionary, corpus_path=fpathroot+fpathappend+'_serialized.mm', lda_model_filepath=fpathroot+fpathappend+'_lda_'+str(numtopics),returnlda = True,numtopics=numtopics, passes = 1, iterations = 50, args = None):
    """
    Run Gensim LDA, optional return of model
    """
    if (type(corpus_path) == str) | (type(corpus_path)==unicode):
        corpus = MmCorpus(corpus_path)
    else:
        corpus = corpus_path
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        if args == None:
            lda = LdaMulticore(corpus,
                    num_topics=numtopics,
                    id2word=gensim_dictionary,
                    workers=n_threads,
                    passes = passes,
                    iterations=iterations)
        else:
            lda = LdaMulticore(corpus,
                    num_topics=numtopics,
                    id2word=gensim_dictionary,
                    workers=n_threads,
                    passes = passes,
                    iterations=iterations,
                    **args)
        lda.save(lda_model_filepath)
        if returnlda == True:
            return lda

def explore_topic(lda, topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """

    print u'{:20} {}'.format(u'term', u'frequency') + u'\n'

    for term, frequency in lda.show_topic(topic_number, topn=25):
        print u'{:20} {:.3f}'.format(term, round(frequency, 3))

def ldavis_create(lda, corpus, gensim_dict,LDAvis_data_filepath=fpathroot+fpathappend+'_lda_vis', return_ldavis = False):
    LDAvis_prepared = pyLDAvis.prepare(lda, corpus,gensim_dict)
    with open(LDAvis_data_filepath, 'w') as f:
        pickle.dump(LDAvis_prepared, f)
    if return_ldavis == True:
        return LDAvis_prepared
    else:
        pyLDAvis.display(LDAvis_prepared)

def _word2vec_train_(corpus, sent_fpath,returnw2v = True,size=100,window=5,epochs = 12, min_count=20,sg=1,workers=n_threads, args =None):
    """
    Have not finished yet
    This function trains word2vec model on sentence corpus.
    By default, the model is returned.
    """
    # Get total number of tokens
    token_count = 0
    for s in corpus:
        token_count = token_count+np.sum(dict(s).values())
    token_count = int(token_count)
    if args==None:
        # initiate the model and perform the first epoch of training
        w2v = Word2Vec(sent_fpath,size=size, window=window,
                            min_count=min_count, sg=sg, workers=n_threads)
        w2v.save(word2vec_filepath)
    else:
        w2v = Word2Vec(sent_fpath,size=size, window=window,
                            min_count=min_count, sg=sg, workers=n_threads, **args)
        w2v.save(word2vec_filepath)
    # perform another n-1 epochs of training
    for i in range(1,epochs):
        w2v.train(sent_fpath, total_examples = token_count, epochs=w2v.iter)
        w2v.save(word2vec_filepath)
    w2v.init_sims()
    if returnw2v==True:
        return w2v

def _word2vec_dataframe_(w2v):
    """
    This function creates a dataframe of the word2vec representations of words.
    Row index are words, columns are vector element values.
    """
    ordered_vocab = [(term, voc.index, voc.count)
                 for term, voc in w2v.wv.vocab.iteritems()]

    ordered_vocab = sorted(ordered_vocab, key=lambda (term, index, count): -count)

    ordered_terms, term_indices, term_counts = zip(*ordered_vocab)

    word_vectors = pd.DataFrame(w2v.wv.syn0norm[term_indices, :],
                                index=ordered_terms)
    return word_vectors
def get_related_terms(w2v,token, topn=10):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """
    for word, similarity in w2v.most_similar(positive=[token], topn=topn):
        print u'{:20} {}'.format(word, round(similarity, 3))
def word_algebra(w2v, add=[], subtract=[], topn=1):
    """
    combine the vectors associated with the words provided
    in add= and subtract=, look up the topn most similar
    terms to the combined vector, and print the result(s)
    """
    answers = w2v.most_similar(positive=add, negative=subtract, topn=topn)
    for term, similarity in answers:
        print term
def word2vec_tsne_vis(word_vectors,fpath=fpathroot+fpathappend, dims=[800,800],colors='blue',topn = 5000,stopwords=STOP_WORDS):
    """
    Have not finished yet
    Takes word_vectors dataframe (output from _word2vec_dataframe_)
    and outputs tsne representation of w2v terms in 2D.
    """
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import HoverTool, ColumnDataSource, value
    tsne_input = word_vectors.drop(stopwords, errors=u'ignore')
    tsne_input = tsne_input.head(topn)
    tsne = TSNE()
    tsne_vectors = tsne.fit_transform(tsne_input.values)
    tsne_filepath = fpathroot+fpathappend+'_tsne_model'
    tsne_vectors_filepath = fpathroot+fpathappend+'tsne_vectors.npy'
    with open(tsne_filepath, 'w') as f:
        pickle.dump(tsne, f)
    pd.np.save(tsne_vectors_filepath, tsne_vectors)
    tsne_vectors = pd.DataFrame(tsne_vectors,
                            index=pd.Index(tsne_input.index),
                            columns=[u'x_coord', u'y_coord'])
    tsne_vectors[u'word'] = tsne_vectors.index
    output_notebook()
    # add our DataFrame as a ColumnDataSource for Bokeh
    plot_data = ColumnDataSource(tsne_vectors)
    # create the plot and configure the
    # title, dimensions, and tools
    w,h = dims
    tsne_plot = figure(title=u't-SNE Word Embeddings',
                       plot_width = w,
                       plot_height = h,
                       tools= (u'pan, wheel_zoom, box_zoom,'
                               u'box_select, resize, reset'),
                       active_scroll=u'wheel_zoom')
    # add a hover tool to display words on roll-over
    tsne_plot.add_tools( HoverTool(tooltips = u'@word') )
    # draw the words as circles on the plot
    tsne_plot.circle(u'x_coord', u'y_coord', source=plot_data,
                     color=colors, line_alpha=0.2, fill_alpha=0.1,
                     size=10, hover_line_color=u'black')
    # configure visual elements of the plot
    tsne_plot.title.text_font_size = value(u'16pt')
    tsne_plot.xaxis.visible = False
    tsne_plot.yaxis.visible = False
    tsne_plot.grid.grid_line_color = None
    tsne_plot.outline_line_color = None
    # engage!
    show(tsne_plot);
