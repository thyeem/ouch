<p align="center"> <img src="https://raw.githubusercontent.com/thyeem/ouch/main/ouch.png" height="250"/></p>

[![ouch](https://img.shields.io/pypi/v/ouch)](https://pypi.org/project/ouch)

# ouch

_Odd Utiltiy Collection Hub_.

[`ouch`](https://github.com/thyeem/ouch) is a collection of utilities that are based on and aligned with [`foc`](https://github.com/thyeem/foc).

```bash
$ pip install -U ouch
```

```python
from ouch import *

# soft flatten
>>> flatten([1, [2, 3, (4, 5)]])
[1, 2, 3, (4, 5)]
# hard flatten
>>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flat | collect
[1, 2, 3, 0, 1, 2]

# 'shell' command
>>> shell(f"du -hs ouch/__init__.py 2>/dev/null") | fst | ob(_.split)()
['40K', 'ouch/__init__.py']

# 'ls' command
>>> ls(".", r=True, grep="^(ouch).*py$")
['ouch/__init__.py']

# poor man's dot-accessible dict, 'dmap' and pretty-printer 'pp'
>>> d = dmap(name="yunchan lim", age=19)
>>> d.cliburn.semifinal.concerto = "Mozart Piano Concerto No.22, K.482"
>>> d.cliburn.semifinal.recital = "Liszt 12 Transcendental Etudes"
>>> d.cliburn.final = "Rachmaninov Piano Concerto No.3, Op.30"
>>> d | pp
  cliburn |     final | Rachmaninov Piano Concerto No.3, Op.30
            semifinal | concerto | Mozart Piano Concerto No.22, K.482
                         recital | Liszt 12 Transcendental Etudes
     name | yunchan lim

# poor man's 'tabulate'
>>> data = [['Name', 'Age'], ['Sofia', 9], ['Maria', 7]]
>>> print(tabulate(data, style='grid'))    # style={'org', 'markdown', ...}
+-------+-----+
| Name  | Age |
+=======+=====+
| Sofia | 9   |
+-------+-----+
| Maria | 7   |
+-------+-----+

# poor man's progress bar, 'tracker'
>>> for batch in tracker(dataloader, "training"):  # single progress bar
...     model(batch)

>>> for i in tracker(range(10), "outer"):  # nested progress bars
...     ...             
...     for j in tracker(range(20), "inner"):
...         ...              

>>> g = (x for x in range(100))
>>> for item in tracker(g, "task", total=100):  # generator with known length
...     process(item)


# and see more poor man's things in 'ouch'
```
