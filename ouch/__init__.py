import operator as op
import os
import random as rd
import re
import sys
import termios
import time
import tty
import zipfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from glob import glob
from io import BytesIO, StringIO
from multiprocessing import Pool, Process
from shutil import rmtree
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from textwrap import fill
from threading import Thread, Timer

from foc import *

__version__ = "0.0.2"

__all__ = [
    "HOME",
    "basename",
    "bin_to_bytes",
    "bytes_to_bin",
    "bytes_to_int",
    "capture",
    "captures",
    "cd",
    "choice",
    "chunks_bytes",
    "chunks_file",
    "chunks_from",
    "chunks_iter",
    "chunks_str",
    "dirname",
    "dmap",
    "exists",
    "flat",
    "flatl",
    "flatten",
    "grep",
    "int_to_bytes",
    "ls",
    "mkdir",
    "neatly",
    "normpath",
    "nprint",
    "parmap",
    "pbcopy",
    "pbpaste",
    "polling",
    "probify",
    "proc",
    "pwd",
    "rand",
    "randbytes",
    "randint",
    "randn",
    "readchar",
    "reader",
    "rmdir",
    "shell",
    "shuffle",
    "singleton",
    "taskbar",
    "thread",
    "timer",
    "timestamp",
    "writer",
]


@fx
def flatten(x, d=1):
    """Reduce the nesting depth by the given level. (swallow flatten)
    Iterables like ``str``, ``bytes`` and ``bytearray`` are not flattened.

    >>> flatten([1, [2, 3, (4, 5)]])
    [1, 2, 3, (4, 5)]
    >>> flatten([1, [(2,), [{3}, (x for x in range(3))]]], d=3)
    [1, 2, 3, 0, 1, 2]
    >>> [1, [(2,), [{3}, (x for x in range(3))]]] | flatten(d=3)
    [1, 2, 3, 0, 1, 2]
    """

    def go(x):
        return [
            a
            for o in x
            for a in (collect(o) if _lazy_iterp(o) else o if _ns_iterp(o) else [o])
        ]

    return cf_(go, rep=d)(x)


@fx
def flat(*args):
    """Flatten iterables until they can no longer be flattened. (deep flatten)
    Iterables like ``str``, ``bytes`` and ``bytearray`` are not flattened.

    >>> flat([1, [(2,), [[{3}, (x for x in range(3))]]]]) | collect
    [1, 2, 3, 0, 1, 2]
    >>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flat | collect
    [1, 2, 3, 0, 1, 2]
    """

    def go(xss):
        if _ns_iterp(xss):
            for xs in xss:
                yield from go([*xs] if _ns_iterp(xs) else xs)
        else:
            yield xss

    return go(args)


@fx
def flatl(*args):
    """The same as ``flat``, but returns in ``list``.

    >>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flatl
    [1, 2, 3, 0, 1, 2]
    >>> flatl([1, [(2,), [[{3}, (x for x in range(3))]]]])
    [1, 2, 3, 0, 1, 2]
    """
    return flat(*args) | collect


def _lazy_iterp(x):
    """Check if the given ``x`` is a lazy iterable."""
    return (
        isinstance(x, Iterator)
        or (isinstance(x, Iterable) and not hasattr(x, "__getitem__"))
        or isinstance(x, range)
    )


def _ns_iterp(x):
    """Check if the given ``x`` is a non-string-like iterable."""
    return (
        isinstance(x, Iterable)
        and not isinstance(x, dict)
        and not isinstance(x, (str, bytes, bytearray))
    )


class dmap(dict):
    """Dot-accessible dict(map).

    >>> dmap()
    {}
    >>> d = dmap(name="yunchan lim", age=19)
    >>> d.name
    'yunchan lim'
    >>> o = "Mozart Piano Concerto No.22, K.482"
    >>> d.cliburn.semifinal.concerto = o
    >>> d.cliburn.semifinal.concerto == d["cliburn"]["semifinal"]["concerto"] == o
    True
    >>> d.cliburn.semifinal.recital = "Liszt 12 Transcendental Etudes"
    >>> d.cliburn.final = "Rachmaninov Piano Concerto No.3, Op.30"
    >>> del d.age
    >>> "age" in d
    False

    When key is not found, it returns a null-like object that is
    interpreted as ``false``

    >>> bool(d.lover)
    False
    >>> repr(d.lover)
    ''

    You can use ``nprint`` together to print formatted text.
    >>> d | nprint
    cliburn  |      final  |  Rachmaninov Piano Concerto No.3, Op.30
             :  semifinal  |  concerto  |  Mozart Piano Concerto No.22, K.482
             :             :   recital  |  Liszt 12 Transcendental Etudes
       name  |  yunchan lim
    """

    __slots__ = ()
    __dwim__ = "- "

    class node:
        __slots__ = ("__parent__", "__key__")

        def __init__(self, parent, key):
            object.__setattr__(self, self.__slots__[0], parent)
            object.__setattr__(self, self.__slots__[1], key)

        def __getattr__(self, key):
            return self.__class__(self, key)

        def __setattr__(self, key, val):
            if key.startswith("__"):  # disabled for stability
                return
            path = []
            n = self
            path.insert(0, key)
            while isinstance(n, self.__class__):
                path.insert(0, n.__key__)
                n = n.__parent__
            o = n  # found root
            for k in path[:-1]:
                if k not in o or not isinstance(o[k], dmap):
                    o[k] = dmap()
                o = o[k]
            o[path[-1]] = dmap._v(val)  # put leaf

        def __bool__(self):
            return False

        def __eq__(self, o):
            return o is None

        def __repr__(self):
            return str()

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            self[key] = self._v(val)

    @classmethod
    def _v(cls, val):
        if isinstance(val, dict) and not isinstance(val, dmap):
            return dmap(val)
        elif _ns_iterp(val):
            return [cls._v(x) for x in val]
        return val

    def _k(self, key):
        if self.__class__.__dwim__:  # dmap using the DWIM key
            for s in self.__class__.__dwim__:
                k = re.sub("_", s, key)
                if k in self:
                    return k
        return key

    def __getattr__(self, key):
        if key not in self and key != "_ipython_canary_method_should_not_exist_":
            k = self._k(key)
            return self[k] if k in self else self.node(self, key)
        return self[key]

    def __setattr__(self, key, val):
        if key.startswith("__"):  # disabled for stability
            return
        self[self._k(key)] = self._v(val)

    def __delattr__(self, key):
        key = key if key in self else self._k(key)
        if key in self:
            del self[key]


def neatly(__, _cols=None, _width=10000, _repr=False, _sort=True, _root=True):
    """Create neatly formatted string for data structure of ``dict`` and ``list``."""

    def indent(x, i):
        def u(c, j=0):
            return f"{c:3}{x[j:]}"

        return (
            (u("-", 3) if i else u("+", 3))
            if x and x[0] == "|"
            else (
                f"{x}"
                if x and x[0] == ":"
                else ((u("-") if x[0] == "+" else u("")) if i else u("+"))
            )
        )

    def bullet(o, s):
        return (
            (indent(x, i) for i, x in enumerate(s))
            if _ns_iterp(o)
            else (f":  {x}" if i else f"|  {x}" for i, x in enumerate(s))
        )

    def filine(x, width, initial, subsequent):
        return fill(
            x,
            width=width,
            break_on_hyphens=False,
            drop_whitespace=False,
            initial_indent=initial,
            subsequent_indent=subsequent,
        )

    if isinstance(__, dict):
        if not __:
            return ""
        _cols = _cols or max(map(len, __.keys()))
        return unlines(
            filine(v, _width, f"{k:>{_cols}}  ", f"{' ':>{_cols}}     ")
            for a, o in (sort if _sort else id)(__.items())
            for k, v in [
                ("", b) if i else (a, b)
                for i, b in enumerate(
                    bullet(o, lines(neatly(o, _repr=_repr, _root=False)))
                )
            ]
        )
    elif _ns_iterp(__):
        if _root:
            return neatly({"'": __}, _repr=_repr, _root=False)
        return unlines(
            filine(v, _width, "", "   ")
            for o in __
            for v in bullet(o, lines(neatly(o, _repr=_repr, _root=False)))
        )
    else:
        return (repr if _repr else str)(__)


@fx
def nprint(__, *, _cols=None, _width=10000, _repr=False, _sort=True):
    """Neatly print data structures of '`dict`' and ``list`` using ``neatly``."""
    print(neatly(__, _cols=_cols, _width=_width, _repr=_repr, _sort=_sort))


def HOME():
    """Get the current user's home directory: the same as ``$HOME``."""
    return os.getenv("HOME")


def cd(path=None):
    """Change directories: similar to the shell-command ``cd``."""
    if path:
        os.chdir(normpath(path, abs=True))
    else:
        os.chdir(HOME())
    return pwd()


def pwd():
    """Get the current directory: similar to the shell-command ``pwd``."""
    return os.getcwd()


def normpath(path, abs=False):
    """Normalize the given filepath"""
    return cf_(
        os.path.abspath if abs else id,
        os.path.normpath,
        os.path.expanduser,
    )(path)


def exists(path, kind=None):
    """Check if the given filepath (file or directory) is available."""
    path = normpath(path)
    if kind == "f":
        return os.path.isfile(path)
    elif kind == "d":
        return os.path.isdir(path)
    else:
        return os.path.exists(path)


def dirname(*args, prefix=False, abs=False):
    if len(args) > 1:
        args = [normpath(a, abs=True) for a in args]
        return os.path.commonprefix(args) if prefix else os.path.commonpath(args)
    else:
        args = [normpath(a, abs=abs) for a in args]
        d = os.path.dirname(*args)
        return d if d else "."


def basename(path):
    return cf_(os.path.basename, normpath)(path)


def mkdir(path, mode=0o755):
    path = normpath(path)
    os.makedirs(path, mode=mode, exist_ok=True)
    return path


def rmdir(path, rm_rf=False):
    path = normpath(path)
    if rm_rf:
        rmtree(path)
    else:
        os.removedirs(path)


def ls(
    *paths,
    grep=None,
    a=False,
    r=False,
    i=False,
    f=False,
    d=False,
    g=False,
    _root=True,
):
    """List directory contents: just like ``ls -1``.
    Glob patterns `(*,?,[)` in `<path..>` are allowed.
    Given ``grep=<regex>``, it behaves like ``ls -1 <path..> | grep <regex>``
    +--------+----------------------------------------------+------------------+
    | Option | Description                                  | In shell        d|
    +--------+----------------------------------------------+------------------+
    | ``a``  | lists hidden files (dotfiles)                | ``ls -a``        |
    +--------+----------------------------------------------+------------------+
    | ``r``  | behaves like ``find -s <path..>``            | ``ls -R``        |
    +--------+----------------------------------------------+------------------+
    | ``i``  | makes ``lgrep`` case-insensitive             | ``ls -i``        |
    +--------+----------------------------------------------+------------------+
    | ``f``  | lists only files                             | ``find -type f`` |
    +--------+----------------------------------------------+------------------+
    | ``d``  | lists only directories                       | ``find -type d`` |
    +--------+----------------------------------------------+------------------+
    | ``g``  | returns a generator instead of a sorted list |        -         |
    +--------+----------------------------------------------+------------------+
    """
    paths = paths or ["."]
    typef = f and f ^ d
    typed = d and f ^ d

    def fd(x):
        return (typef and exists(x, "f")) or (typed and exists(x, "d"))

    def listdir(x):
        return cf_(
            id if a else filter(cf_(not_, f__(str.startswith, "."))),
            os.listdir,
        )(x)

    def root(xs):
        return flat(
            (
                glob(normpath(x))
                if re.search(r"[\*\+\?\[]", x)
                else cf_(
                    guard_(exists, f"ls, no such file or directory: {x}"),
                    normpath,
                )(x)
            )
            for x in xs
        )

    def rflag(xs):
        return flat(
            (
                (x, ls(x, grep=grep, a=a, r=r, i=i, f=f, d=d, g=g, _root=False))
                if exists(x, "d")
                else x
            )
            for x in xs
        )

    return cf_(
        id if g else sort,  # return generator or sort by filepath
        filter(fd) if typef ^ typed else id,  # filetype filter: -f or -d flag
        globals()["grep"](grep, i=i) if grep else id,  # grep -i flag
        rflag if r else id,  # recursively listing: -R flag
    )(
        flat(
            [normpath(f"{x}/{o}") for o in listdir(x)] if exists(x, "d") else x
            for x in (root(paths) if _root else paths)
        )
    )


def grep(regex, *, i=False):
    """Build a filter to select items matching ``regex`` pattern from iterables.

    >>> grep(r".json$", i=True)([".json", "Jason", ".JSON", "jsonl", "JsonL"])
    ['.json', '.JSON']
    """
    return fx(filterl(f_(re.search, regex, flags=re.IGNORECASE if i else 0)))


@fx
def shell(cmd, sync=True, o=True, *, executable="/bin/bash"):
    """Execute shell commands [sync|async]hronously and capture its outputs.
    +-----------+-----------+------------------------------------------------+
    |   o-value | Return    | Meaning                                        |
    +-----------+-----------+------------------------------------------------+
    |         1 | ``[str]`` | captures stdout/stderr (``2>&1``)              |
    +-----------+-----------+------------------------------------------------+
    |        -1 | ``None``  | discard (``&>/dev/null``)                      |
    +-----------+-----------+------------------------------------------------+
    | otherwise | ``None``  | do nothing or redirection (``2>&1 or &>FILE``) |
    +-----------+-----------+------------------------------------------------+

    >>> shell("ls -1 ~")                   # doctest: +SKIP
    >>> shell("find . | sort" o=-1)        # doctest: +SKIP
    >>> shell("cat *.md", o=writer(FILE))  # doctest: +SKIP
    """
    import shlex

    o = PIPE if o == 1 else DEVNULL if o == -1 else 0 if isinstance(o, int) else o
    sh = f_(
        Popen,
        cf_(unwords, mapl(normpath), shlex.split)(cmd),
        stdin=PIPE,
        stderr=STDOUT,
        shell=True,
        executable=executable,
    )
    if sync:
        if o == PIPE:
            proc = sh(stdout=o)
            out, _ = proc.communicate()
            return lines(out.decode())
        else:
            sh(stdout=o).communicate()
    else:
        sh(stdout=o)


@fx
def pbcopy(x):
    """Copy text to the clipboard (``macOS`` only)"""
    Popen("pbcopy", stdin=PIPE).communicate(x.encode())


def pbpaste():
    """Paste text from the clipboard (``macOS`` only)"""
    return Popen("pbpaste", stdout=PIPE).stdout.read().decode()


def reader(f=None, mode="r", zipf=False):
    """Get ready to read stream from a file or stdin, then returns the handle."""
    if f is not None:
        guard(exists(f, "f"), f"reader: not found such a file: {f}")
    return (
        sys.stdin
        if f is None
        else zipfile.ZipFile(normpath(f), mode) if zipf else open(normpath(f), mode)
    )


def writer(f=None, mode="w", zipf=False):
    """Get ready to write stream to a file or stout, then returns the handle."""
    return (
        sys.stdout
        if f is None
        else zipfile.ZipFile(normpath(f), mode) if zipf else open(normpath(f), mode)
    )


def chunks_from(kind):
    """Build a lazy-splitter that splits iterables from {iterable, file, string}
    into n-length pieces, including the last chunk if any.
    """

    @fx
    def from_iter(n, x, clip=False):
        it = iter(x)
        while True:
            chunk = []
            try:
                for _ in range(n):
                    chunk.append(next(it))
            except StopIteration:
                if not clip and chunk:
                    yield chunk
                break
            yield chunk

    @fx
    def from_file(n, x, mode="r", clip=False):
        with reader(x, mode=mode) as f:
            while True:
                o = f.read(n)
                if len(o) != n:
                    if not clip and o:
                        yield o
                    break
                yield o

    @fx
    def from_strlike(fio, n, x, clip=False):
        with fio(x) as s:
            while True:
                o = s.read(n)
                if len(o) != n:
                    if not clip and o:
                        yield o
                    break
                yield o

    return dict(
        file=from_file,  # file
        iter=from_iter,  # iterable
        bytes=from_strlike(BytesIO),  # bytes
        str=from_strlike(StringIO),  # string
    ).get(kind) or error(f"chunks_from, no such selector: {kind}")


chunks_iter = chunks_from("iter")

chunks_file = chunks_from("file")

chunks_bytes = chunks_from("bytes")

chunks_str = chunks_from("str")


def readchar():
    """Reads a single char from ``<stdin>`` in raw-mode on Unix-like systems."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@fx
def capture(p, string):
    """Get the text that matches the first Regex pattern ``p``"""
    x = captures(p, string)
    if x:
        return fst(x)


@fx
def captures(p, string):
    """Captures"""
    return re.compile(p).findall(string)


def bytes_to_int(x, byteorder="big"):
    return int.from_bytes(x, byteorder=byteorder)


def int_to_bytes(x, size=None, byteorder="big"):
    if size is None:
        size = (x.bit_length() + 7) // 8
    return x.to_bytes(size, byteorder=byteorder)


def bytes_to_bin(x, sep=""):
    return sep.join(f"{b:08b}" for b in x)


def bin_to_bytes(x):
    return int_to_bytes(int(x, base=2))


def randbytes(n):
    """generate cryptographically secure random bytes"""
    return os.urandom(n)


def rand(x=None, high=None, size=None):
    return (
        [rd.uniform(x, high) for _ in range(size)]
        if size is not None
        else (
            rd.uniform(x, high)
            if high is not None
            else rd.uniform(0, x) if x is not None else rd.random()
        )
    )


def randn(mu=0, sigma=1, size=None):
    return (
        [rd.gauss(mu, sigma) for _ in range(size)]
        if size is not None
        else rd.uniform(mu, sigma)
    )


def randint(x=None, high=None, size=None):
    """generate random integer cryptographically secure and faster than numpy's.
    return random integer(s) in range of [low, high)
    """

    def rint(high=1 << 256, low=0):
        guard(low < high, f"randint: low({low}) must be less than high({high})")
        x = high - low
        return low + (bytes_to_int(randbytes((x.bit_length() + 7) // 8)) % x)

    return (
        [rint(high, x) for _ in range(size)]
        if size is not None
        else (
            rint(high, x)
            if high is not None
            else (rint(x) if x is not None else rint())
        )
    )


@fx
def probify(fn, p=None):
    return fn if rand() < p else id


@fx
def shuffle(x):
    """Fisher-Yates shuffle in a cryptographically secure way"""
    for i in range(len(x) - 1, 0, -1):
        j = randint(0, i + 1)
        x[i], x[j] = x[j], x[i]
    return x


@fx
def choice(x, size=None, *, replace=False, p=None):
    """Generate a sample with/without replacement from a given iterable."""

    def fromp(x, probs):
        guard(
            len(x) == len(probs),
            f"choice, different size: len(x)={len(x)}, len(p)={len(probs)}.",
        )
        guard(
            abs(sum(probs) - 1) < 1e-6,
            f"choice, sum([prob])={sum(probs)}, not equal to 1.",
        )
        r = rand()
        for y, p in zip(x, scanl1(op.add, probs)):
            if r < p:
                return y

    def with_size(size):
        size = int(len(x) * size) if 0 < size < 1 else size
        return [
            x[i]
            for i in (
                randint(0, len(x), size)
                if replace or len(x) < size
                else shuffle(rangel(len(x)))[:size]
            )
        ]

    if not len(x):
        return x
    if p is not None:
        return fromp(x, p)
    if size is None:
        return x[randint(len(x))]
    else:
        return with_size(size)


def singleton(cls):
    """decorate a class and make it a singleton class."""
    o = {}

    def go(*args, **kwargs):
        if cls not in o:
            o[cls] = cls(*args, **kwargs)
        return o[cls]

    return go


def thread(daemon=False):
    """decorator factory that turns functions into threading.Thread.

    >>> mouse = thread()(mouse_listener)()  # doctest: +SKIP
    >>> mouse.start()                       # doctest: +SKIP
    >>> mouse.join()                        # doctest: +SKIP
    """

    def t(f):
        def go(*args, **kwargs):
            return Thread(target=f, args=args, kwargs=kwargs, daemon=daemon)

        return go

    return t


def proc(daemon=False):
    """decorator factory that turns functions into multiprocessing.Process.

    >>> ps = [proc(True)(bruteforce)(x) for x in xs]  # doctest: +SKIP
    >>> for p in ps: p.start()                        # doctest: +SKIP
    >>> for p in ps: p.join()                         # doctest: +SKIP
    """

    def p(f):
        def go(*args, **kwargs):
            return Process(target=f, args=args, kwargs=kwargs, daemon=daemon)

        return go

    return p


def parmap(f, x, *xs, workers=None):
    """easily parallelizes function applications over iterables
    by utilizing multiple cpu cores with ``multiprocessing.Pool``.
    In the same way as ``map``, it very simply maps a function over
    an iterable and executes it in parallel.

    >>> parmap(bruteforce, preImages)  # doctest: +SKIP
    >>> parmap(os.remove, manyFiles)   # doctest: +SKIP
    """
    workers = workers or os.cpu_count()
    with Pool(workers) as pool:
        x = zip(x, *xs) if xs else x
        mapper = pool.starmap if xs else pool.map
        return mapper(f, x)


class polling:
    """repeatedly executes a provided function at fixed time intervals.

    >>> g = f_(cf_(print, force), lazy(randint, 100))  # doctest: +SKIP
    >>> p = polling(1, g)                              # doctest: +SKIP
    >>> p.start()                                      # doctest: +SKIP
    """

    def __init__(self, sec, f, *args, **kwargs):
        self.expr = lazy(f, *args, **kwargs)
        self.timer = f_(Timer, sec, self._g)
        self.on = False
        self.t = None

    def _g(self):
        if self.on:
            self.expr()
            self.t = self.timer()
            self.t.start()

    def start(self):
        if not self.on:
            self.on = True
            self._g()

    def stop(self):
        self.on = False
        if self.t:
            self.t.cancel()


def timer(t, msg="", v=True, callback=None):
    guard(isinstance(t, (int, float)), f"timer: not a number: {t}")
    guard(t > 0, f"timer: must be given a positive number: {t}")
    t = int(t)
    fmt = f"{len(str(t))}d"
    while t >= 0:
        if v:
            print(f"{msg}  {t:{fmt}}", end="\r")
            writer().write("\033[K")
        time.sleep(1)
        t -= 1
    if callback:
        callback()


def timestamp(*, origin=None, w=0, d=0, h=0, m=0, s=0, from_iso=None, to_iso=False):
    if from_iso:
        t = datetime.strptime(from_iso, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
    else:
        dt = timedelta(
            weeks=w,
            days=d,
            hours=h,
            minutes=m,
            seconds=s,
        ).total_seconds()
        if origin is None:
            origin = datetime.utcnow().timestamp()
        t = origin + dt

    return to_iso and f"{datetime.fromtimestamp(t).isoformat()[:26]}Z" or t


def taskbar(x=None, desc="working", *, start=0, total=None, barcolor="white", **kwargs):
    """flexible tqdm-like progress bar relying on ``pip`` package only"""
    import pip._vendor.rich.progress as rp

    class SpeedColumn(rp.ProgressColumn):
        def render(self, task) -> rp.Text:
            if task.speed is None:
                return rp.Text("?", style="progress.data.speed")
            return rp.Text(f"{task.speed:2.2f} it/s", style="progress.data.speed")

    def track(tb, x, start, total):
        with tb:
            if total is None:
                total = len(x) if float(op.length_hint(x)) else None
            if start:
                guard(total is not None, f"taskbar: not subscriptable: {x}")
                start = total + start if start < 0 else start
                x = islice(x, start, None)
                task = tb.add_task(desc, completed=start, total=total)
                yield from tb.track(x, task_id=task, total=total, description=desc)
            if total:
                tb._tasks.get(task).completed = total

    tb = rp.Progress(
        "[progress.description]{task.description}",
        "",
        rp.TaskProgressColumn(),
        "",
        rp.BarColumn(complete_style=barcolor, finished_style=barcolor),
        "",
        rp.MofNCompleteColumn(),
        "",
        rp.TimeElapsedColumn(),
        "<",
        rp.TimeRemainingColumn(),
        "",
        SpeedColumn(),
        **kwargs,
    )
    return tb if x is None else track(tb, x, start, total)
