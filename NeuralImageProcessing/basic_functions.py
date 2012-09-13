from pipeline import TimeSeries
import numpy as np
import sklearn.decomposition as sld
from scipy.spatial.distance import pdist, squareform
import scipy.ndimage as sn
import mtxfactor #import stJADE, RRI
reload(mtxfactor)

FILT = {'median': sn.filters.median_filter, 'gauss':sn.filters.gaussian_filter,
        'uniform':sn.filters.uniform_filter, 'erosion': sn.binary_erosion, 'dilation': sn.binary_dilation,
        'closing': sn.binary_closing, 'simple_erosion': sn.binary_erosion}
FILTARG = {'median': lambda value: {'size':value}, 'gauss':lambda value: {'sigma':value},
           'uniform':lambda value: {'size':value},
           'erosion':lambda value: {'structure':sn.iterate_structure(sn.generate_binary_structure(2, 1), value)},
           'dilation':lambda value: {'structure':sn.iterate_structure(sn.generate_binary_structure(2, 1), value)},
           'closing':lambda value: {'structure':sn.iterate_structure(sn.generate_binary_structure(2, 1), value)},
            'simple_erosion': lambda value: {}
           }

class Filter():
    ''' filter series with filterop in 2D'''

    def __init__(self, filterop, extend, downscale=1):
        self.downscale = downscale
        self.filterop = FILT[filterop]
        self.args = FILTARG[filterop](extend)


    def __call__(self, timeseries):
        filtered_images = []
        for image in timeseries.shaped2D():
            im = self.filterop(image, **self.args)
            if self.downscale:
                im = im[::self.downscale, ::self.downscale]
            filtered_images.append(im.flatten())
        shape = im.shape
        out = timeseries.copy()
        out.shape = shape
        out.timecourses = np.vstack(filtered_images)
        return out

class CutOut():
    ''' cuts out images of a series from range[0] to range[1]'''

    def __init__(self, cut_range):
        self.cut_range = cut_range

    def __call__(self, timeseries):
        image_cut = timeseries.copy()
        image_cut.set_timecourses(image_cut.trial_shaped()[:, self.cut_range[0]:self.cut_range[1]])
        return image_cut

class TrialMean():
    ''' splits each trial in equal parts and calculates their means'''

    def __init__(self, parts=1):
        self.parts = parts

    def __call__(self, timeseries):
        splits = np.vsplit(timeseries.timecourses, self.parts * timeseries.num_trials)
        averaged_im = [np.mean(im, 0) for im in splits]
        out = timeseries.copy()
        out.timecourses = np.vstack(averaged_im)
        return out

class RelativeChange():
    ''' gives relative change of trials to base_image '''

    def __init__(self):
        pass

    def __call__(self, timeseries, baseseries):
        relative_change = timeseries.copy()
        relative_change.set_timecourses((timeseries.trial_shaped() - baseseries.trial_shaped())
                           / baseseries.trial_shaped())
        return relative_change

class Combine():
    ''' combines two imageseries by cominefct '''

    def __init__(self, combine_fct):
        self.combine_fct = combine_fct

    def __call__(self, timeseries1, timeseries2):
        change = timeseries1.copy()
        change.timecourses = self.combine_fct(timeseries1.timecourses, timeseries2.timecourses)
        return change

class PCA():

    def __init__(self, variance=None):
        self.variance = variance

    def __call__(self, timeseries):
        self.pca = sld.PCA(n_components=self.variance)
        self.pca.fit(timeseries.timecourses)
        out = timeseries.copy()
        out.type = 'latent_series'
        out.label_objects = ['mode' + str(i) for i in range(self.pca.components_.shape[0])]
        out.shape = (len(out.label_objects),)
        out.timecourses = self.pca.transform(timeseries.timecourses)
        out.eigen = self.pca.explained_variance_ratio_
        out.base = TimeSeries(self.pca.components_, shape=timeseries.shape, name='base',
                              label_sample=out.label_objects)
        return out

class sICA():

    def __init__(self, variance=None, latent_series=False):
        self.variance = variance
        self.latent_series = latent_series

    def __call__(self, timeseries):

        self.pca = sld.PCA(n_components=self.variance)

        if self.latent_series:
            base = self.pca.fit_transform(timeseries.base.timecourses.T)
            time = np.dot(self.pca.components_, timeseries.timecourses.T)
        else:
            try:
                base = self.pca.fit_transform(timeseries.timecourses.T)
            except np.linalg.LinAlgError:
                return 'Error'
            time = self.pca.components_


        normed_base = base / np.sqrt(self.pca.explained_variance_)
        normed_time = time * np.sqrt(self.pca.explained_variance_.reshape((-1, 1)))

        print normed_base.shape, normed_time.shape

        self.ica = sld.FastICA(whiten=False)
        self.ica.fit(normed_base)


        out = timeseries.copy()
        base = self.ica.components_.T
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1))

        time = self.ica.transform(normed_time.T)
        time *= new_norm

        timesign = np.sign(np.sum(time, 0))
        time *= timesign
        base *= timesign.reshape((-1, 1))

        out.timecourses = time
        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ = 'latent_series'
        out.name += '_sica'
        out.base = TimeSeries(base, shape=timeseries.shape, name=out.name,
                              label_sample=out.label_objects)
        if self.latent_series:
            out.base.shape = timeseries.base.shape
        return out

class stICA():

    def __init__(self, variance=None, **param):
        param['latents'] = variance
        self.param = param

    def __call__(self, timeseries):

        base, time = mtxfactor.stJADE(timeseries.timecourses.T, **self.param)

        out = timeseries.copy()
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1))

        time = time.T
        time *= new_norm

        #timesign = np.sign(np.sum(time[np.abs(time) > 0.05], 0))
        print time.shape
        timesign = np.sign(np.diag(time[np.argmax(np.abs(time), 0)]))
        time *= timesign
        base *= timesign.reshape((-1, 1))

        out.timecourses = time
        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ = 'latent_series'
        out.name += '_stica'
        out.base = TimeSeries(base, shape=timeseries.shape, name=out.name,
                              label_sample=out.label_objects)

        return out

class NNMA():

    def __init__(self, variance=100, maxcount=100, **param):
        self.latents = variance
        self.maxcount = maxcount
        if not(param):
            self.param = dict(sparse_par=0, sparse_par3=0, psi=1e-12, eps=1e-7)
        else:
            self.param = param
        self.A = None
        self.X = None

    def __call__(self, timeseries):
        self.A, self.X, self.obj, count, converged = mtxfactor.RRI()(timeseries.timecourses,
                                              self.latents, A=self.A, X=self.X, verbose=1,
                                              maxcount=self.maxcount, shape=timeseries.shape, **self.param)
        out = timeseries.copy()

        base = self.X
        new_norm = np.diag(base[:, np.argmax(np.abs(base), 1)])
        base /= new_norm.reshape((-1, 1)) + 1E-15
        out.timecourses = self.A
        out.timecourses *= new_norm

        out.label_objects = ['mode' + str(i) for i in range(base.shape[0])]
        out.shape = (len(out.label_objects),)
        out.typ = 'latent_series'
        out.name += '_nnma'
        out.base = TimeSeries(base, shape=timeseries.shape, name=out.name,
                              label_sample=out.label_objects)
        return out

class SampleSimilarity():
    """calculate similarity of samples with the same Labels

    return timeseries-mask such that only samples with a self similarity below threshold are included.
    Outliers in the self similarity matrix are removed by a greedy elimination as long
    as the mean exceeds the threshold.
    """

    def __init__(self, threshold, metric='correlation'):
        self.metric = metric
        self.threshold = threshold

    def __call__(self, timeseries):
        sample_set = set(timeseries.label_sample)
        mask = np.zeros(len(timeseries.label_sample), dtype='bool')
        for label in sample_set:
            label_index = [i for i, tmp_label in enumerate(timeseries.label_sample) if tmp_label == label]
            mean_distance = np.mean(pdist(timeseries.timecourses[label_index], self.metric))
            while mean_distance > self.threshold and len(label_index) > 2:
                new_means = []
                for l in label_index:
                    tmp_label_index = [ll for ll in label_index if not ll == l]
                    distances = pdist(timeseries.timecourses[tmp_label_index], self.metric)
                    new_means.append(np.mean(distances))
                min_index = np.argmin(new_means)
                label_index.pop(min_index)
                mean_distance = new_means[min_index]
            if mean_distance < self.threshold:
                mask[label_index] = True
        out = timeseries.copy()
        out.timecourses = mask
        return out

class SampleSimilarityPure():
    """calculate similarity of samples with the same Labels

    return dictionaries with interlabel and crosslabel distances.
    The crosslabel distance shows how *standard* a response is, how similar
    to all other stimuli. We are interested in stimuli that have a high similarity
    within the same label and low similarity accross.
    """

    def __init__(self, metric='correlation'):
        self.metric = metric

    def __call__(self, timeseries):
        sample_set = set(timeseries.label_sample)
        distanceself, distancecross = {}, {}
        for label in sample_set:
            label_index = np.array([i == label for i in timeseries.label_sample])
            temp = squareform(pdist(timeseries.timecourses, self.metric))
            distancecross[label] = temp[label_index][:, np.logical_not(label_index)]
            distanceself[label] = pdist(timeseries.timecourses[label_index], self.metric)
        return distanceself, distancecross

class SelectTrials():
    ''' selects trials bases on mask

        !!! mask has to be boolean !!!
    '''
    def __init__(self):
        pass

    def __call__(self, timeseries, mask):
        mask = mask.timecourses
        selected_timecourses = timeseries.trial_shaped()[mask, :, :]
        out = timeseries.copy()
        out.set_timecourses(selected_timecourses)
        out.label_sample = [out.label_sample[i] for i in np.where(mask)[0]]
        return out

class CalcStimulusDrive():
    ''' creates pseudo trial and calculates distance metric between them (for each object)
        does not take stimuli into account with only one repetition
    '''

    def __init__(self, metric='correlation'):
        self.metric = metric

    def __call__(self, timeseries):
        labels = timeseries.label_sample

        stim_set = set(labels)
        # create dictionary with key: stimulus and value: trial where stimulus was given
        stim_pos = {}
        min_stimlen = np.nan
        for stimulus in stim_set:
            occurence = np.where([i == stimulus for i in labels])[0]
            if len(occurence) > 1:
                stim_pos[stimulus] = occurence
                min_stimlen = np.nanmin([min_stimlen, len(occurence)])



        # create list of lists, where each sublist contains for all stimuli one exclusive trial
        indices = []
        for i in range(int(min_stimlen)):
            indices.append([j[i] for j in stim_pos.values()])
        # create pseudo-trial timecourses
        trial_timecourses = np.array([timeseries.trial_shaped()[i].reshape(-1, timeseries.num_objects) for i in indices])
        # calculate correlation of pseudo-trials, aka stimulus dependency
        cor = []
        for object_num in range(timeseries.num_objects):
            dists = pdist(trial_timecourses[:, :, object_num] + 1E-12 * np.random.randn(*trial_timecourses[:, :, object_num].shape), self.metric)
            er = np.isnan(dists)
            if np.sum(er) > 0:
                print 'nan before', np.sum(np.isnan(trial_timecourses[:, :, object_num])), 'nan dist', np.sum(er)
                dists[er] = 0
            cor.append(np.mean(dists))
        out = timeseries.copy()
        out.timecourses = np.array(cor).reshape((1, -1))
        out.label_sample = [out.name]
        return out

class SelectModes():
    ''' select objects which lie under the threshold in the filtervalues '''

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, timeseries, filtervalues):
        mask = filtervalues.timecourses.squeeze() < self.threshold
        selected_timecourses = timeseries.timecourses[:, mask]
        out = timeseries.copy()
        out.timecourses = selected_timecourses
        out.label_objects = [out.label_objects[i] for i in np.where(mask)[0]]
        out.shape = (len(out.label_objects),)
        if timeseries.typ == 'latent_series':
            out.base.timecourses = out.base.timecourses[mask]
            out.base.label_sample = out.label_objects

        return out

class SingleSampleResponse():
    ''' calculates a single response for each label

    options:
    mean -> the mean of all samples
    best -> the strongest response
    attetention: reorders labels
    '''


    def __init__(self, method='mean'):
        self.method = 'mean'

    def __call__(self, timeseries):
        timecourses = timeseries.trial_shaped()
        labels = timeseries.label_sample
        label_set = list(set(labels))
        new_labels, new_timecourses = [], []
        for label in label_set:
            label_index = [i for i, tmp_label in enumerate(labels) if tmp_label == label]
            if self.method == 'mean':
                single_timecourse = np.mean(timecourses[label_index], 0)
            elif self.method == 'best':
                #select trial with most signal
                besttime = np.argmax([np.sum(np.abs(timecourses[l_ind])) for l_ind in label_index])
                single_timecourse = timecourses[label_index[besttime]]
            new_labels.append(label)
            new_timecourses.append(single_timecourse)
        new_timecourses = np.vstack(new_timecourses)

        out = timeseries.copy()
        out.timecourses = new_timecourses
        out.label_sample = new_labels
        return out

class SortBySamplename():
    ''' sort timeseries by samplename'''

    def __init__(self):
        pass

    def __call__(self, timeseries):
        timecourses = timeseries.trial_shaped()
        labels = timeseries.label_sample
        label_ind = np.argsort(labels)
        out = timeseries.copy()
        out.set_timecourses(timecourses[label_ind])
        out.label_sample = [labels[i] for i in label_ind]
        return out

class Distance():
    ''' calculate distance between objects/samples '''

    def __init__(self, metric='correlation', direction='temporal', onlymin=False):
        self.metric = metric
        self.direction = direction
        self.onlymin = onlymin

    def __call__(self, timeseries):
        if self.direction == 'temporal':
            dist = pdist(timeseries.timecourses.T, self.metric)
            labels = timeseries.label_objects
        elif self.direction == 'spatial':
            dist = pdist(timeseries.timecourses, self.metric)
            labels = timeseries.label_sample
        if self.onlymin:
            dist = np.nanmin(squareform(dist) + np.diag(np.ones(len(labels))), 0)
            new_labels = labels
        else:
            new_labels = reduce(lambda x, y:x + y, [[labels[i] + ':' + labels[j] for i in range(j + 1, len(labels))] for j in range(len(labels))])
        out = timeseries.copy()
        out.timecourses = dist.reshape((1, -1))
        out.label_objects = new_labels
        out.shape = (len(new_labels),)
        out.label_sample = [timeseries.name]
        return out

class ObjectConcat():
    ''' concat timeseries objects

    options:
    unequalsample=False, if integer n samples are reduced to common labels (each label n-times)
    unequalobj=False, if True stores the shape of each object as in out.shape as list
    '''

    def __init__(self, unequalsample=False, unequalobj=False):
        self.unequalsample = unequalsample
        self.unequalobj = unequalobj

    def __call__(self, timeserieses):
        timecourses, name, label_objects = [], [], []


        if self.unequalsample:
            common = list(set(reduce(lambda x, y: set(x).intersection(y), [ts.label_sample for ts in timeserieses])))
            common.sort()
            print 'common sample ', len(common)
        for ts in timeserieses:
            out = ts.copy()
            if self.unequalsample:
                ind = [positions(lab, ts.label_sample)[:self.unequalsample] for lab in common]
                ind = sum(ind, [])
                print ts.name, ' reduced from ', len(ts.label_sample)
                out.set_timecourses(ts.trial_shaped()[ind])
                timecourses.append(out.timecourses)
            else:
                assert ts.label_sample == out.label_sample, 'samples do not match'
                timecourses.append(ts.timecourses)
            label_objects += [ts.name + '_' + lab for lab in ts.label_objects]
            name.append(ts.name)
        out.timecourses = np.hstack(timecourses)
        out.name = common_substr(name)
        out.label_objects = label_objects
        if self.unequalsample:
            out.label_sample = sum([[tmp] * self.unequalsample for tmp in common], [])
        if self.unequalobj:
            out.shape = [ts.shape for ts in timeserieses]
        return out

class ObjectScrambledConcat():
    ''' concat timeseries objects'''

    def __init__(self, repititions, method='simple'):
        self.repititions = repititions
        method_dict = {'simple': scramble_simple, 'three':scramble_three}
        self.scrambling = method_dict[method]

    def __call__(self, timeserieses):
        timecourses, name, label_objects = [], [], []

        common = list(set(reduce(lambda x, y: set(x).intersection(y), [ts.label_sample for ts in timeserieses])))
        common.sort()
        print common
        for ts_ind, ts in enumerate(timeserieses):
            out = ts.copy()
            ind = [self.scrambling(positions(lab, ts.label_sample)[:2], self.repititions, ts_ind) for lab in common]
            ind = sum(ind, [])
            out.set_timecourses(ts.trial_shaped()[ind])
            timecourses.append(out.timecourses)
            label_objects += [ts.name + '_' + lab for lab in ts.label_objects]
            name.append(ts.name)

        out.timecourses = np.hstack(timecourses)
        out.name = common_substr(name)
        out.label_objects = label_objects
        out.label_sample = sum([[tmp] * self.repititions for tmp in common], [])
        out.shape = [ts.shape for ts in timeserieses]
        return out

class SampleConcat():
    ''' concat timeserieses timecourses'''


    def __init__(self, listout=False):
        self.listout = listout

    def __call__(self, timeserieses):
        out = timeserieses[0].copy()
        out.timecourses, out.name, out.label_sample = [], [], []
        for ts in timeserieses:
            if not(self.listout):
                assert ts.label_objects == out.label_objects, 'objects do not match'
            out.timecourses.append(ts.timecourses)
            out.label_sample += ts.label_sample
            out.name.append(ts.name)
        if not(self.listout):
            out.timecourses = np.vstack(out.timecourses)
        out.name = common_substr(out.name)
        return out

class StimulusIntegrator(object):
    """calculate mean of all values within a stimulus that are above a certain threshold"""
    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, timeseries):
        trial_shaped = timeseries.trial_shaped()
        n_trials = trial_shaped.shape[0]
        n_modes = trial_shaped.shape[2]
        integrated = np.zeros((n_trials, n_modes))
        for i_mode in range(n_modes):
            for i_trial in range(n_trials):
                trial = trial_shaped[i_trial, :, i_mode]
                integrated[i_trial, i_mode] = np.mean(trial[trial > self.threshold])
        out = timeseries.copy()
        out.set_timecourses(integrated)
        return out




def common_substr(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_substr(data[0][i:i + j], data):
                    substr = data[0][i:i + j]
        return substr
    else:
        return data[0]

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True

def positions(target, source):
    '''Produce all positions of target in source'''
    pos = -1
    out = []
    try:
        while True:
            pos = source.index(target, pos + 1)
            out.append(pos)
    except ValueError:
        return out

def scramble_simple(inds , leng, pos):
    extended_list = [inds[0]] * np.ceil(leng / 2.) + [inds[1]] * np.floor(leng / 2.)

    shifted_list = extended_list[pos:] + extended_list[:pos]
    return shifted_list

def scramble_three(inds, leng, pos):
    if pos < 3:
        return scramble_simple(inds, leng, pos)
    elif pos == 3:
        return [inds[1]] * leng
