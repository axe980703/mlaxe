from mlaxe.classifiers import SGDLinearClassifier
from mlaxe.sample import Sample2D


x, y = Sample2D(classes=2, radius=10, mean=0,
                stdev=10, seed=322, cl_size=35, show=False).gen()

cls = SGDLinearClassifier(loss_func='hebb', seed=2, lr_init=0.001).fit(x, y)
print(cls.evaluate(x, y))

cls.get_anim(save_gif=False)

print(cls.iter_spent)
