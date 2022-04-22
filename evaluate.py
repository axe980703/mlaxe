from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=2, radius=15, mean=0,
                stdev=15, seed=3, cl_size=100, show=False).gen()

cls = SGDLinearClassifier(seed=2, lr_init=0.001, max_iter=1000).fit(x, y)
print(cls.evaluate(x, y))

cls.get_anim(save_gif=False)

print(cls.iter_spent)
