from mlaxe.classifiers import SGDLinearClassifier
from evaluation.sample import Sample2D


x, y = Sample2D(classes=2, radius=7, mean=0,
                stdev=5, seed=7, cl_size=120).gen()

cls = SGDLinearClassifier(tol_iter=20)
cls.fit(x, y)
print(cls.evaluate(x, y))
