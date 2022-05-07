from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=3, radius=12, mean=2, show=False,
                stdev=6, seed=322, cl_size=100).gen()

cls = SGDLinearClassifier(seed=2).fit(x, y)
print(cls.evaluate(x, y))

cls.get_anim(save_gif=False)

print(cls.iter_spent)
