from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=2, radius=10, mean=0,
                stdev=5, seed=322, cl_size=70).gen()


cls = SGDLinearClassifier(tol_iter=20, seed=1).fit(x, y)
cls.evaluate(x, y)
cls.get_anim(save_gif=True)
