"""官方参数训练器的全量数据版本。仅关闭验证集划分。"""

from .trainer_original import OrangeClassifier as OriginalOrangeClassifier


class OrangeClassifier(OriginalOrangeClassifier):
    validation_fraction = 0.0
