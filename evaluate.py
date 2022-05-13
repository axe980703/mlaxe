from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=5, radius=18, mean=2, show=False,
                stdev=5, seed=4, cl_size=100).gen()

cls = SGDLinearClassifier(seed=5, max_iter=2e3, alt_class=True,
                          decr_lrate=False).fit(x, y)

print(cls.evaluate(x, y))
cls.get_anim(save_gif=False)
print(cls.iter_spent)
