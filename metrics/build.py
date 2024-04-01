from metrics import MyIoUMetric

def build_metric(config):
    metric_type = config.METRIC.TYPE
    if metric_type == 'iou':
        metric = MyIoUMetric(iou_metrics=['mIoU', 'mFscore'])
    else:
        raise NotImplementedError(f"Unkown metric: {metric_type}")

    return metric