from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=5, radius=15, mean=0,
                stdev=7, seed=3, cl_size=100).gen()

cls = SGDLinearClassifier(seed=2, max_iter=1000).fit(x, y)
print(cls.evaluate(x, y))

cls.get_anim(save_gif=False)

print(cls.iter_spent)
