'''
Created on 18.02.2011

@author: jan
'''

import tempfile
import os
import numpy as np
from matplotlib import collections
import pylab as plt
from scipy.stats import gaussian_kde

class VisualizeTimeseries(object):

    def __init__(self, fig=None):
        self.fig = fig
        self.axes = {'base':[], 'time':[]}


    def oneaxes(self):
        if not self.fig:
            self.fig = plt.figure(figsize=(8, 8))
        ax = self.fig.add_subplot(111)
        self.axes['base'].append(ax)
        self.axes['time'].append(ax)


    def base_and_time(self, num_objects, height=0.9, aspect=False):
        if not(self.fig):
            self.fig = plt.figure(figsize=(20, 13))

        height = height / num_objects
        if aspect:
            figheight = self.fig.get_figheight()
            figwidth = self.fig.get_figwidth()
            figaspect = figwidth / figheight
            aspect = aspect * figaspect
        else: 
            aspect = 1
            
        for i in range(num_objects):
            #create timeaxes
            ax = self.fig.add_axes([0.25, height * i + 0.05, 0.70, min(height - 0.01, 0.19 * aspect)])
            ax.set_xticklabels([])
            self.axes['time'].append(ax)
            #create baseaxes
            ax = self.fig.add_axes([0.05, height * i + 0.05, min(height, 0.2) - 0.01, min(height - 0.01, 0.19 * aspect)])
            ax.set_axis_off()
            ax.set_gid(num_objects - 1 - i)
            self.axes['base'].append(ax)
        # bring plots in order as you would expect from subplot
        self.axes['base'].reverse()
        self.axes['time'].reverse()

    def base_and_singlestimtime(self, num_objects, stimlabels, height=0.9, aspect=False):
        ''' generate figure layout with base on timeaxes for each individual stimulus
        axes['time'] is than a list of dictionarys where each stimuluslabel is the key
        to the correspoding axes '''
        
        
        totaltime_width = 0.7
        axspacing = 0.01
        
        if not(self.fig):
            self.fig = plt.figure(figsize=(20, 13))

        height = height / num_objects
        if aspect:
            figheight = self.fig.get_figheight()
            figwidth = self.fig.get_figwidth()
            figaspect = figwidth / figheight
            aspect = aspect * figaspect
        else:
            aspect = 1
        
        
        timewidth = totaltime_width / len(stimlabels)
        for i in range(num_objects):
            lab2ax = {}
            for lab_idx, lab in enumerate(stimlabels):
                #create timeaxes
                ax = self.fig.add_axes([0.25 + timewidth * lab_idx, height * i + 0.05, timewidth - axspacing,
                                         min(height - 0.01, 0.19 * aspect)])
                ax.set_xticklabels([])
                lab2ax[lab] = ax
            self.axes['time'].append(lab2ax)
            #create baseaxes
            ax = self.fig.add_axes([0.05, height * i + 0.05, min(height, 0.2) - 0.01, min(height - 0.01, 0.19 * aspect)])
            ax.set_axis_off()
            ax.set_gid(num_objects - 1 - i)
            self.axes['base'].append(ax)
        # bring plots in order as you would expect from subplot
        self.axes['base'].reverse()
        self.axes['time'].reverse()

    def subplot(self, num_objects, dim2=None):
        if not(self.fig):
            self.fig = plt.figure(figsize=(8, 8))
        if not(dim2):
            subplot_dim1 = np.ceil(np.sqrt(num_objects))
            subplot_dim2 = np.ceil(num_objects / subplot_dim1)
        else:
            subplot_dim2 = dim2
            subplot_dim1 = np.ceil(1.*num_objects / subplot_dim2)
        for axind in xrange(num_objects):
            axhandle = self.fig.add_subplot(subplot_dim1, subplot_dim2, axind + 1)
            self.axes['base'].append(axhandle)


    '''
    def alltoall(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            for obj_ind in range(num_objects):
                yield ax_ind, obj_ind


    def onetoall(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            yield ax_ind, range(num_objects)

    def onetoone(self, num_axes, num_objects):
        for ax_ind in range(num_axes):
            yield ax_ind, ax_ind

    '''

    def contourfaces(self, ax, im):
        ax.contourf(im, [0.3, 1], colors=['r'], alpha=0.2)

    def contour(self, ax, im, **cargs):
        ax.contour(im, **cargs)

    def overlay_image(self, ax, im, threshold=0.1, title=False, ylabel=False, colormap=plt.cm.jet):
        im_rgba = colormap(im / 2 + 0.5)
        #im_rgba[:, :, 3] = 0.8
        #im_rgba[np.abs(im) < threshold, 3] = 0
        
        
        alpha = np.abs(im) - threshold
        alpha /= (1 - threshold)
        alpha[alpha < 0] = 0
        alpha = alpha ** 0.7
        im_rgba[:, :, 3] = alpha
        
        ax.imshow(im_rgba, interpolation='none')
        if title:
            ax.set_title(**title)
        if ylabel:
            ax.set_ylabel(**ylabel)

    def imshow(self, ax, im, title=False, colorbar=False, ylabel=False, **imargs):
        im = ax.imshow(im, interpolation='none', aspect='equal', **imargs)
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(**title)
        if ylabel:
            ax.set_ylabel(**ylabel)
        if colorbar:
            self.fig.colorbar(im, ax=ax)
    
    def overlay_workaround(self, ax, bg, bgargs, im, imargs, endargs):
        fig = plt.figure(111)
        axtemp = fig.add_subplot(111)
        self.imshow(axtemp, bg, **bgargs)
        self.overlay_image(axtemp, im, **imargs)
        axtemp.set_axis_off()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            fig.savefig(f, transparent=True, bbox_inches='tight',
                        format='png', pad_inches=0)
            bitmap_name = f.name
        plt.close(111)
        new_im = plt.imread(bitmap_name)
        os.remove(bitmap_name)
        self.imshow(ax, new_im, **endargs)
        
        
    def plot(self, ax, time, **kwargs):
        ax.plot(time, '-', **kwargs)
        ax.xticklabels = []


    def add_labelshade(self, ax, timeseries, rotate=True):
        # create changing shade with changing label
        shade = []
        shade_color = 0
        labels = timeseries.label_sample
        reference_label = labels[0]
        self.shadelabel = []
        for label in labels:
            self.shadelabel.append(label)
            if not(label == reference_label):
                reference_label = label
                shade_color = 1 - shade_color
            elif len(self.shadelabel) > 1:
                self.shadelabel[-1] = ''
            shade.append(shade_color)
        shade = np.outer(np.array(shade), np.ones((timeseries.timepoints))).flatten()
        shade[np.hstack((np.array([0]), np.diff(shade))) == -1] = 1
        shade = np.hstack((shade, np.array([1])))



        axshade = collections.BrokenBarHCollection.span_where(
                                np.arange(len(shade) + 1) - 0.5,
                                *ax.get_ylim(), where=shade > 0,
                                facecolor='k', alpha=0.2)
        ax.add_collection(axshade)

    def add_onsets(self, ax, timeseries, stimuli_offset):
        ax.set_xticks(np.arange(timeseries.num_trials) * timeseries.timepoints + stimuli_offset)

    def add_shade(self, ax, timeseries, timepoints):
        # create changing shade according to binary timseries
        shade = np.outer(np.array(timeseries.timecourses.squeeze().astype('int')), np.ones(timepoints)).flatten()
        shade[np.hstack((np.array([0]), np.diff(shade))) == -1] = 1
        shade = np.hstack((shade, np.array([1])))
        axshade = collections.BrokenBarHCollection.span_where(
                                np.arange(len(shade) + 1) - 0.5,
                                *ax.get_ylim(), where=shade > 0,
                                facecolor='g', alpha=0.2)
        ax.add_collection(axshade)

    def add_samplelabel(self, ax, timeseries, rotation='0', toppos=False, stimuli_offset=0):
        ax.set_xticks(np.arange(timeseries.num_trials) * timeseries.timepoints + stimuli_offset)
        ax.set_xticklabels(timeseries.label_sample)

        for tick in ax.xaxis.iter_ticks():
            tick[0].label2On = toppos
            tick[0].label1On = not(toppos)
            tick[0].label2.set_rotation(rotation)
            tick[0].label2.set_ha('left')
            tick[0].label2.set_size('x-small')
            tick[0].label2.set_stretch('extra-condensed')
            tick[0].label2.set_family('sans-serif')

    def add_violine(self, ax, distributionlist, color='b', rotation='0'):
        violin_plot(ax, distributionlist, range(len(distributionlist)), color)
        ax.set_xlim((-0.5, len(distributionlist) - 0.5))


    def add_axescolor(self, where, how, timeseries, ec='g', lw=2):
        axes = self.axes[where]
        for ax_ind, obj_ind in self.mappings[how](len(axes), timeseries.num_objects):
            if timeseries[obj_ind]:
                ax = axes[ax_ind]
                for spine in ax.spines.values():
                    spine.set_edgecolor(ec)
                    spine.set_linewidth(lw)



def violin_plot(ax, data, pos, color):
    '''
    create violin plots on an axis
    '''
    dist = max(pos) - min(pos)
    w = min(0.15 * max(dist, 1.0), 0.5)
    for i, (d, p) in enumerate(zip(data, pos)):
        where_nan = np.isnan(d)
        where_inf = np.isinf(d)
        if np.sum(np.logical_or(where_nan, where_inf)) > 0:
            print 'number of nans/infs: ', np.sum(where_nan), np.sum(where_inf)
            d = d[np.logical_not(np.logical_or(where_nan, where_inf))]
            print d, type(d)
            if not(d):

                print 'skipped'
                continue
        k = gaussian_kde(d) #calculates the kernel density
        m = k.dataset.min() #lower bound of violin
        M = k.dataset.max() #upper bound of violin
        x = np.arange(m, M, (M - m) / 100.) # support for violin
        v = k.evaluate(x) #violin profile (density curve)
        v = v / v.max() * w #scaling the violin to the available space
        if len(color) > 1:
            c = color[i]
        else:
            c = color
        ax.fill_betweenx(x, p, v + p, facecolor=c, edgecolor='None', alpha=0.3)
        ax.fill_betweenx(x, p, -v + p, facecolor=c, edgecolor='None', alpha=0.3)

'''
def initmouseob(path='/media/Iomega_HDD/Experiments/Messungen/111210sph/',
           dateikenn='_nnma'):

    """ ======== load in decomposition ======== """
    measID = path.strip('/').split('/')[-1]
    db = dataimport.instantJChemInterface()

    base = np.load(path + 'base' + dateikenn + '.npy')
    norm = np.max(base, 1)
    base /= norm.reshape((-1, 1))
    timecourse = np.load(path + 'time' + dateikenn + '.npy') * norm

    shape = np.load(path + 'shape.npy')
    bg = np.asarray(Image.open(path + 'bg.png').convert(mode='L').resize(
                                                    (shape[1], shape[0])))

    namelist = pickle.load(open(path + 'ids.pik'))
    names = db.make_table_dict('cd_id', ['Name'], 'MOLECULE_PROPERTIES')
    labels = []
    for label in namelist:
        name_parts = label.split('_')
        odor_name = names[int(name_parts[0])][0][0]
        if len(name_parts) > 2:
            odor_name += name_parts[2].strip()
        if len(name_parts) > 3:
            odor_name += name_parts[3].strip()
        labels.append(odor_name)

    decomposition = ip.TimeSeries(timecourse, name=[measID], shape=shape,
                 typ='Decomposition', label_sample=labels)
    decomposition.base = base
    decomposition.bg = bg

    """ ======== load in data ======== """
    preprocessed_timecourse = np.load(path + 'data.npy')
    preprocessed = ip.TimeSeries(preprocessed_timecourse, name=[measID],
                  shape=shape, typ='Timeseries', label_sample=labels)




    """ ====== combine Timeseries ===== """
    combi = TimeSeriesDecomposition()
    combi.factorization = decomposition
    combi.data = preprocessed
    combi.roi = roidata
    return combi
'''
