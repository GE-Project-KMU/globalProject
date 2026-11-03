"""Exact original-resolution threshold sweep; no test-driven calibration."""
import math
import cv2
import numpy as np

PAINT_THRESHOLDS=(.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.925,.95,.975,.99)


def threshold_confusions(truth,delta,thresholds=PAINT_THRESHOLDS):
    if truth.ndim!=2 or delta.ndim!=2 or delta.dtype.kind!="f" or not np.isin(truth,[0,1,255]).all() or not np.isfinite(delta).all():
        raise ValueError("invalid_paint_metric_arrays")
    values=list(thresholds)
    if not values or values!=sorted(set(values)) or any(not 0<t<1 for t in values):
        raise ValueError("strictly_increasing_paint_thresholds_required")
    if delta.shape!=truth.shape:
        delta=cv2.resize(delta,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_NEAREST)
    # Nearest resize commutes with thresholding. Bins count exactly how many
    # strict delta>logit(threshold) comparisons each original pixel passes.
    cuts=np.asarray([math.log(t/(1-t)) for t in values],dtype=delta.dtype)
    bins=np.searchsorted(cuts,delta,side="left")
    positive=np.bincount(bins[truth==1],minlength=len(values)+1)
    negative=np.bincount(bins[truth==0],minlength=len(values)+1)
    ignored=int((truth==255).sum())
    result={}
    for i,t in enumerate(values):
        tp=int(positive[i+1:].sum());fp=int(negative[i+1:].sum())
        result[t]={"tp":tp,"fp":fp,"fn":int(positive.sum())-tp,"tn":int(negative.sum())-fp,"ignored":ignored}
    return result
