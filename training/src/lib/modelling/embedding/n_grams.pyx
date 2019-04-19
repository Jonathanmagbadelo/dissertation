from collections import defaultdict
import cython

def nested_default_dict():
    return defaultdict(lambda: defaultdict(float))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef get_co_values(list data, unsigned short window_size, bint symmetric_window):
    counts = defaultdict(nested_default_dict)
    cdef:
        int i, j
        unsigned int ind, lind, rind, doc_size
        unsigned short[:] document
    for document_co_variate, document in data:
        doc_size = document.shape[0]
        for i in range(doc_size):
            for j in range(1, window_size):
                ind = document[i]
                if ind > 0:
                    if i - j > 0:
                        lind = document[i - j]
                        if lind > 0:
                            counts[document_co_variate][ind][lind] += 1.0 / j
                    if symmetric_window and i + j < doc_size:
                        rind = document[i + j]
                        if  rind > 0:
                            counts[document_co_variate][ind][rind] += 1.0 / j
    return counts

cpdef get_co_values_d(list data, unsigned short window_size):
    counts = defaultdict(nested_default_dict)
    cdef:
        int i, j
        unsigned int doc_size
        list document
    for document_co_variate, document in data:
        doc_size = len(document)
        for i in range(doc_size):
            for j in range(1, window_size):
                ind = document[i]
                if i - j > 0:
                    lind = document[i - j]
                    counts[document_co_variate][ind][lind] += 1.0 / j
                if i + j < doc_size:
                    rind = document[i + j]
                    counts[document_co_variate][ ind][rind] += 1.0 / j
    return counts