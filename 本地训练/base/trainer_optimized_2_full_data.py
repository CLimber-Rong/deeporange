"""optimized_2 训练器的全量数据版本。仅关闭内部验证集划分。"""

from . import trainer_optimized_2 as TRAINER_API


class OrangeClassifier(TRAINER_API.OrangeClassifier):
    validation_fraction = 0.0
