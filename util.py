import theano, cPickle, h5py, lasagne, random, csv, gzip                                                  
import numpy as np
import theano.tensor as T         


# convert csv into format readable by rmn code
def load_data(span_path, metadata_path):
    x = csv.DictReader(gzip.open(span_path, 'rb'))
    wmap, cmap, bmap = cPickle.load(open(metadata_path, 'rb'))
    max_len = -1
    rmap = {'Agent': 0, 'Patient': 1, 'Attribute': 2}

    revwmap = dict((v,k) for (k,v) in wmap.iteritems())
    revbmap = dict((v,k) for (k,v) in enumerate(bmap))
    revcmap = dict((v,k) for (k,v) in cmap.iteritems())

    span_dict = {}
    role_dict = {}
    wexists = {}
    num_roles = len(rmap)
    for row in x:
        text = row['Words'].split()
	roles = row['Roles'].split()
        if len(text) > max_len:
            max_len = len(text)
        key = '++++$++++'.join([row['Movie'], row['Character']])
        if key not in span_dict:
            span_dict[key] = []
	    role_dict[key] = []
	span = [wmap[w] for w in text]
	role = [rmap[r] for r in roles] 
        span_dict[key].append(span)
	role_dict[key].append(role)
	ind = 0
	for w in span:
	    wexists[w * num_roles + role[ind]] = True
	    ind += 1

    span_data = []
    for key in span_dict:
        movie, character = key.split('++++$++++')
        movie = np.array([revbmap[movie]]).astype('int32')
        chars = np.array([revcmap[character]]).astype('int32')

        # convert spans to numpy matrices 
        spans = span_dict[key]
	roles = role_dict[key]
        s = np.zeros((len(spans), max_len)).astype('int32')
        m = np.zeros((len(spans), max_len)).astype('float32')
	r = np.zeros((len(spans), max_len)).astype('int32') 
        for i in range(len(spans)):
            curr_span = spans[i]
	    curr_role = roles[i]
	    s[i][:len(curr_span)] = curr_span
            m[i][:len(curr_span)] = 1.
	    r[i][:len(curr_role)] = curr_role

        span_data.append([movie, chars, s, m, r])
    return span_data, max_len, wmap, cmap, bmap, rmap, wexists


def generate_negative_samples(num_traj, span_size, negs, span_data):
    inds = np.random.randint(0, num_traj, negs)
    neg_words = np.zeros((negs, span_size)).astype('int32')
    neg_roles = np.zeros((negs, span_size)).astype('int32')
    neg_masks = np.zeros((negs, span_size)).astype('float32')
    for index, i in enumerate(inds):
        rand_ind = np.random.randint(0, len(span_data[i][2]))
        neg_words[index] = span_data[i][2][rand_ind]
        neg_masks[index] = span_data[i][3][rand_ind]
	neg_roles[index] = span_data[i][4][rand_ind]

    return neg_words, neg_masks, neg_roles

