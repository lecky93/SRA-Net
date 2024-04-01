

class SegData():
    def __init__(self,
                 img_path=None,
                 img_name=None,
                 ori_size=None,
                 label_size=None,
                 gt_label=None,
                 cls_label=None,
                 is_splicing=False):
        self.img_path = img_path
        self.img_name = img_name
        self.ori_size = ori_size
        self.label_size = label_size
        self.gt_label = gt_label
        self.cls_label = cls_label
        self.is_splicing = is_splicing
        self.pred_logits = None
        self.pred_label = None
        self.recon = None
        self.cam = None
        self.residual = None