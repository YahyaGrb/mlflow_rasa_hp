from hyperopt import hp
from hyperopt.pyll.base import scope


search_space = {
    "epochs": scope.int(hp.quniform("epochs", 0, 50,2)),
    "model": hp.choice("model", ["fr_core_news_sm", "fr_core_news_md", "fr_dep_news_trf"]),
    "threshold": hp.uniform("threshold", 0, 1)
    }