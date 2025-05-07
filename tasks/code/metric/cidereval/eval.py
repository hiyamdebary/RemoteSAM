__author__ = 'rama'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .cider.cider import Cider

class CIDErEvalCap:
    
    def __init__(self, coco, cocoRes, df='corpus'):
        print('tokenization...')

        imgIds = {'image_id': coco.getImgIds()}['image_id']
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = coco.imgToAnns[imgId]
            res[imgId] = cocoRes.imgToAnns[imgId]

        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        self.eval = {}
        self.gts = gts
        self.res = res
        self.df = df


    def evaluate(self):
        metric_scores = {}
        scorer = Cider(df=self.df)
        score, scores = scorer.compute_score(self.gts, self.res)
        metric_scores["CIDEr"] = list(scores)
        self.setEval(score, "CIDEr")
        return metric_scores


    def setEval(self, score, method):
        self.eval[method] = score
