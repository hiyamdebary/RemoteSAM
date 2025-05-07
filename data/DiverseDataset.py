from .datasets import *

def DiverseDataset(args, image_transforms=None, max_tokens=20):

    # return image, gt, image_name, foo(optional)

    task = args.task
    dataset = args.dataset

    assert task in ['DET', 'CNT', 'MLC', 'MCC', 'CAP', 'VG'], "task must be in ['DET', 'CNT', 'MLC', 'MCC', 'CAP', 'VG'], but got {}".format(task)

    if task == 'CNT':
        assert dataset in ['DIOR', 'DOTA'], "dataset must be in ['DIOR', 'DOTA'], but got {}".format(dataset)
        if dataset == 'DIOR':
            # foo := classname
            return DIOR_CNT(args, max_tokens=max_tokens)
        elif dataset == 'DOTA':
            # foo := classname
            return DOTA_CNT(args, max_tokens=max_tokens)

    elif task == 'VG':
        assert dataset in ['RSVG'], "dataset must be in ['RSVG'], but got {}".format(dataset)
        if dataset == 'RSVG':
            return RSVG_VG(args, max_tokens=max_tokens)

    elif task == 'DET':
        assert dataset in ['DIOR', 'DOTAv2', 'DOTAv1'], "dataset must be in ['DIOR', 'DOTAv2', 'DOTAv1'], but got {}".format(dataset)
        if dataset == 'DIOR':
            # foo := (image_id, classindex)
            return DIOR_DET(args, max_tokens=max_tokens)
        elif dataset == 'DOTAv2':
            return DOTAv2_DET(args, max_tokens=max_tokens)
        elif dataset == 'DOTAv1':
            return DOTAv1_DET(args, max_tokens=max_tokens)

    elif task == 'MLC':
        assert dataset in ['DIOR', 'DOTAv2'], "dataset must be in ['DIOR', 'DOTAv2'], but got {}".format(dataset)
        if dataset == 'DIOR':
            return DIOR_MLC(args, max_tokens=max_tokens)
        elif dataset == 'DOTAv2':
            return DOTAv2_MLC(args, max_tokens=max_tokens)

    elif task == 'MCC':
        assert dataset in ['UCM', 'AID'], "dataset must be in ['UCM', 'AID'], but got {}".format(dataset)
        if dataset == 'UCM':
            return UCM_MCC(args, max_tokens=max_tokens)
        if dataset == 'AID':
            return AID_MCC(args, max_tokens=max_tokens)

    elif task == 'CAP':
        assert dataset in ['UCM'], "dataset must be in ['UCM'], but got {}".format(dataset)
        if dataset == 'UCM':
            return UCM_CAP(args, max_tokens=max_tokens)