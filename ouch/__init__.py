import builtins as bi
import multiprocessing
import operator as op
import os
import random as rd
import re
import sys
import termios
import threading
import time
import tty
import zipfile
from ast import literal_eval
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from io import BytesIO, StringIO
from shutil import rmtree
from subprocess import DEVNULL, PIPE, STDOUT, Popen
from textwrap import fill
from unicodedata import east_asian_width

import numpy as np
from foc import *

__version__ = "0.0.24"

__all__ = [
    "HOME",
    "basename",
    "base58d",
    "base58e",
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
    "dataq",
    "dirname",
    "dmap",
    "deepdict",
    "du_hs",
    "exists",
    "flat",
    "flatl",
    "flatten",
    "getext",
    "grep",
    "int_to_bytes",
    "justf",
    "ls",
    "mkdir",
    "neatly",
    "normpath",
    "pp",
    "parmap",
    "pbcopy",
    "pbpaste",
    "polling",
    "probify",
    "proc",
    "prompt",
    "pwd",
    "rand",
    "randbytes",
    "randint",
    "randn",
    "readchar",
    "read_conf",
    "reader",
    "rmdir",
    "shell",
    "shuffle",
    "singleton",
    "stripext",
    "tabulate",
    "thread",
    "timeago",
    "timer",
    "timestamp",
    "tmpfile",
    "tracker",
    "write_conf",
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

    return cf_(*replicate(d, go))(x)


@fx
def flat(*args):
    """Flatten iterables until they can no longer be flattened. (deep flatten)
    Iterables like ``str``, ``bytes`` and ``bytearray`` are not flattened.

    >>> flat([1, [(2,), [[{3}, (x for x in range(3))]]]]) | collect
    [1, 2, 3, 0, 1, 2]
    >>> [1, [(2,), [[{3}, (x for x in range(3))]]]] | flat | collect
    [1, 2, 3, 0, 1, 2]
    """

    def go(args):
        for arg in args:
            if _ns_iterp(arg):
                yield from go(arg)
            else:
                yield arg

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
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray))


def _ns_builtin_iterp(x):
    return isinstance(
        x, (list, tuple, range, deque, set, dict, dict, frozenset, memoryview)
    )


class dmap(dict):
    """Dot-accessible map or ``dict``.

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

    You can use ``pp`` together to print formatted text.
    >>> d | pp
    cliburn |     final | Rachmaninov Piano Concerto No.3, Op.30
              semifinal | concerto | Mozart Piano Concerto No.22, K.482
                           recital | Liszt 12 Transcendental Etudes
       name | yunchan lim
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
            o[path[-1]] = dmap.__val__(val)  # put leaf

        def __bool__(self):
            return False

        def __eq__(self, o):
            return o is None

        def __repr__(self):
            return str()

    def __init__(self, /, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, val in self.items():
            self[key] = self.__val__(val)

    @classmethod
    def __val__(cls, val):
        if isinstance(val, dict):
            return dmap(val)
        elif _ns_builtin_iterp(val):
            return [cls.__val__(x) for x in val]
        return val

    def __key__(self, key):
        if self.__class__.__dwim__:  # dmap using the DWIM key
            for s in self.__class__.__dwim__:
                k = re.sub("_", s, key)
                if k in self:
                    return k
        return key

    def __getattr__(self, key):
        if key not in self and key != "_ipython_canary_method_should_not_exist_":
            k = self.__key__(key)
            return self[k] if k in self else self.node(self, key)
        return self[key]

    def __setattr__(self, key, val):
        if key.startswith("__"):  # disabled for stability
            return
        self[self.__key__(key)] = self.__val__(val)

    def __delattr__(self, key):
        key = key if key in self else self.__key__(key)
        if key in self:
            del self[key]

    def __or__(self, o):
        if type(o) is fx:
            return o(self)
        else:
            return dmap(super().__or__(o))

    def __ror__(self, o):
        return dmap(super().__or__(o))

    def __ior__(self, o):
        self.update(o)
        return self


def deepdict(obj, seen=None):
    """Creates a deep ``dict`` from a given object.
    This recursively converts nested objects into standard ``dict``,
    handling circular references for mutable objects only.

    >>> class Person:
    ...     def __init__(self, name, amount):
    ...         self.name = name
    ...         self.amount = amount
    >>> deepdict(Person("Alice", 42))
    {'name': 'Alice', 'amount': 42}

    >>> a = [1,2,3]
    >>> a.append(a)
    >>> deepdict(a)
    [1, 2, 3, '[...]']
    """
    if seen is None:
        seen = set()
    if obj is None:
        return None

    if isinstance(obj, (dict, list)) or hasattr(obj, "__dict__"):
        obj_id = bi.id(obj)
        if obj_id in seen:
            return "[...]" if isinstance(obj, list) else "{...}"
        seen.add(obj_id)

    if isinstance(obj, dict):
        return {deepdict(k, seen): deepdict(v, seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deepdict(element, seen) for element in obj]
    elif hasattr(obj, "__dict__"):
        return deepdict(obj.__dict__, seen)
    else:
        return obj


@fx
def neatly(d, width=200, sort=True, show=5, gap=1, quote=False, margin=None):
    """Create neatly formatted strings for instances of builtin iterables."""
    __ = " " * gap

    def stringify(x):
        def join(x):
            return ", ".join(map(str, x))

        def contract(x, show):
            return f"[{join(x[:show])}, ..., {join(x[-show:])}]"

        s = str(x) if not show or len(x) < 2 * show else contract(list(x), show)
        return s if len(s) < width else contract(list(x), 1)

    def filln(text, initial_indent, subsequent_indent):
        return fill(
            text,
            width=width,
            initial_indent=initial_indent,
            subsequent_indent=subsequent_indent,
            break_on_hyphens=False,
            drop_whitespace=False,
        )

    def bullet(o, sym):
        return [f" {__}{x}" if i else f"{sym}{__}{x}" for i, x in enumerate(o)]

    if isinstance(d, dict):
        if not d:
            return ""
        margin = margin or max(map(cf_(len, str), d.keys()))
        return unlines(
            filln(
                ln,
                f"{('' if i else k):>{margin}}{__}",
                f"{' ':>{margin+2*gap+1}}",
            )
            for k, v in (sorted if sort else id)(d.items())
            for i, ln in enumerate(
                bullet(
                    lines(
                        neatly(
                            v,
                            width=width,
                            sort=sort,
                            show=show,
                            gap=gap,
                            quote=quote,
                        )
                    ),
                    "|",
                )
            )
        )
    elif _ns_builtin_iterp(d):
        if any(isinstance(x, dict) or _ns_builtin_iterp(x) for x in d):
            return unlines(
                filln(v, "", "")
                for i, o in enumerate(d)
                for v in bullet(
                    lines(
                        neatly(
                            o,
                            width=width,
                            sort=sort,
                            show=show,
                            gap=gap,
                            quote=quote,
                        )
                    ),
                    "," if i else "[",
                )
            )
        else:
            return stringify(d)
    else:
        return (repr if quote else str)(d)


@fx
def pp(d, width=200, sort=True, show=5, gap=1, quote=False, margin=None):
    """Print neatly formatted strings of the builtin iterables by ``neatly``.

    >>> import torch                               # doctest: +SKIP
    >>> torch.nn.Linear(8, 24).state_dict() | pp   # doctest: +SKIP
    >>> map(_ * 7)(seq(5,...)) | takel(200) | pp
    [35, 42, 49, 56, 63, ..., 1400, 1407, 1414, 1421, 1428]
    """
    print(
        neatly(
            d,
            width=width,
            sort=sort,
            show=show,
            gap=gap,
            quote=quote,
            margin=margin,
        )
    )


@fx
def tabulate(rows, nohead=False, style="plain", missing="", fn=id):
    """Create a formatted table from data.

     nohead | first row of ``rows`` is used as head unless ``nohead``
      style | {"plain", "markdown", "org", "grid"}
    missing | placeholder for missing data (``None``)
       fn   | a function that finalizes each row

    >>> data = [['Name', 'Age'], ['Sofia', 9], ['Maria', 7]]
    >>> print(tabulate(data, style='grid'))
    +-------+-----+
    | Name  | Age |
    +=======+=====+
    | Sofia | 9   |
    +-------+-----+
    | Maria | 7   |
    +-------+-----+
    """

    def cell(c, w, missing):
        return f"{str(c) if c is not None else missing:<{w}}"

    def row(r, ws, sep, missing):
        return sep.join(cell(c, w, missing) for c, w in zip(r, ws))

    def separator(l, s, m, r, ws):
        return l + s.join(m * w for w in ws) + r

    guard(
        rows | map(length) | fx(set) | length == 1,
        "error, either empty rows or not all rows have the same length.",
    )
    sty = dmap(
        grid=dict(
            top=f_(separator, "+-", "-+-", "-", "-+"),
            header=f_(separator, "+=", "=+=", "=", "=+"),
            middle=f_(separator, "+-", "-+-", "-", "-+"),
            left="| ",
            right=" |",
            sep=" | ",
        ),
        markdown=dict(
            header=f_(separator, "|-", "-|-", "-", "-|"),
            left="| ",
            right=" |",
            sep=" | ",
        ),
        org=dict(
            header=f_(separator, "|-", "-+-", "-", "-|"),
            left="| ",
            right=" |",
            sep=" | ",
        ),
        plain=dict(
            header=lambda x: "-" * (sum(x) + 4 * (len(x) - 1)),
            left="",
            right="",
            sep="    ",
        ),
    ).get(style) or error(
        f"Error, unsupported border style: '{style}'.\n"
        "Options are 'grid', 'markdown', 'org', and 'plain'."
    )
    ws = [  # [maximum-of-each-coloum-width]
        max(len(str(c) if c is not None else missing) for c in col)
        for col in zip(*rows)
    ]
    fmt_rows = [row(r, ws, sty.sep, missing) for r in rows]
    o = []
    if sty.top:
        o.append(sty.top(ws))
    if not nohead:
        o.append(sty.left + head(fmt_rows) + sty.right)
        if sty.header:
            o.append(sty.header(ws))
    for r in fmt_rows if nohead else tail(fmt_rows):
        o.append(sty.left + r + sty.right)
        if sty.middle:
            o.append(sty.middle(ws))
    return "\n".join(map(fn, o))


def HOME():
    """Get the current user's home directory: the same as ``$HOME``."""
    return os.getenv("HOME")


def cd(path=None):
    """Change directories: similar to the shell command ``cd``."""
    if path:
        os.chdir(normpath(path, abs=True))
    else:
        os.chdir(HOME())
    return pwd()


def pwd():
    """Get the current directory: similar to the shell command ``pwd``."""
    return os.getcwd()


def normpath(path, abs=False):
    """Normalize and expand a givien filepath, ``path``."""
    return cf_(
        os.path.abspath if abs else id,
        os.path.normpath,
        os.path.expanduser,
    )(path)


def exists(path, kind=None):
    """Check if a given filpath ``path`` exists."""
    path = normpath(path)
    if kind == "f":
        return os.path.isfile(path)
    elif kind == "d":
        return os.path.isdir(path)
    else:
        return os.path.exists(path)


def dirname(*args, prefix=False, abs=False):
    """Get the directory name of a filepath.
    If multiple filepaths are provided, returns common directory among them.
    """
    if len(args) > 1:
        args = [normpath(a, abs=True) for a in args]
        return os.path.commonprefix(args) if prefix else os.path.commonpath(args)
    else:
        args = [normpath(a, abs=abs) for a in args]
        d = os.path.dirname(*args)
        return d if d else "."


def basename(path):
    """Get the base name of a filepath."""
    return cf_(os.path.basename, normpath)(path)


def mkdir(path, mode=0o755):
    """Create a directory and any necessary parent directories."""
    path = normpath(path)
    os.makedirs(path, mode=mode, exist_ok=True)
    return path


def rmdir(path, rm_rf=False):
    """Remove a directory.
    If ``rm_rf`` is set, remove directory and all its contents.
    """
    path = normpath(path)
    if rm_rf:
        rmtree(path)
    else:
        os.removedirs(path)


def stripext(path, sep="."):
    """Remove the file extension from a filepath, ``path``."""
    o = path.split(sep)
    return unchars(o[:-1] if len(o) > 1 else o)


def getext(path, sep="."):
    """Get the file extension from a filepath, ``path``."""
    o = path.split(sep)
    return o[-1] if len(o) > 1 else None


def tmpfile(prefix=None, suffix=None, dir="/tmp", size=6, encoder=bytes.hex):
    """Generate a temporary filepath.

    >>> tmpfile(dir=f"{HOME()}/ouch", size=8)              # doctest: +SKIP
    >>> tmpfile(suffix=".key", size=128, encoder=base58e)  # doctest: +SKIP
    """
    mkdir(dir)
    return (
        f"{normpath(dir, abs=True)}/"
        f"{prefix or ''}{encoder(randbytes(size))}{suffix or ''}"
    )


def du_hs(f):
    """Get the total size of a directory in human-readable format.

    >>> du_hs("~/data/corpus")    # doctest: +SKIP
    '232G'
    >>> du_hs("../setup.py")      # doctest: +SKIP
    '4.0K'
    # yields ``None`` for a non-existent path
    >>> du_hs("2252b67d2c5f369f191d51163e5d87a6")
    """
    o = shell(f"du -hs {f} 2>/dev/null")
    if o:
        return o[0].split()[0]


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
    | Option | Description                                  | In shell         |
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

    Getting content of the current directory,
    >>> ls()                  # doctest: +SKIP

    Expands "~" automatically,
    >>> ls("~")               # doctest: +SKIP

    Lists hidden files, (starting with ".", dotfiles)
    >>> ls(a=True)            # doctest: +SKIP

    Available multiple filepaths,
    >>> ls(FILE, DIR, ...)    # doctest: +SKIP

    Supports glob patterns, (``*``, ``?``, ``[``)
    >>> ls("./*/*.py")
    ['ouch/__init__.py']

    Lists ``.git`` directory recursively and pick files ending with digits,
    >>> ls(".git", r=True, grep="\\d$")      # doctest: +SKIP

    Only files in `'.git`' directory,
    >>> ls(".git", r=True, f=True)           # doctest: +SKIP

    Only directories in '`.git`' directory,
    >>> ls(".git", r=True, d=True)           # doctest: +SKIP

    Search recursivley, then match patterns with `grep`.
    `'i=True'` for case-insensitive grep pattern.
    >>> ls(".", r=True, i=True, grep=".PY")  # doctest: +SKIP

    In more convenient way,
    >>> ls(".", r=True, grep=".py$")         # doctest: +SKIP

    Found the location of the current file.
    >>> ls(".", r=True, grep="^(ouch).*py$")
    ['ouch/__init__.py']

    Same as above,
    >>> ls("ouch/*.py")
    ['ouch/__init__.py']
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

    >>> shell("ls -1 ~")                     # doctest: +SKIP
    >>> shell("find . | sort" o=-1)          # doctest: +SKIP
    >>> shell("cat *.md", o=writer(FILE))    # doctest: +SKIP
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
    """Copy text to the clipboard. (``macOS`` only)

    >>> "Long Long string ..." | pbcopy
    >>> dict(sofia="piano", maria="violin") | neatly | pbcopy
    """
    Popen("pbcopy", stdin=PIPE).communicate(x.encode())


def pbpaste():
    """Paste text from the clipboard. (``macOS`` only)"""
    return Popen("pbpaste", stdout=PIPE).stdout.read().decode()


def reader(f=None, mode="r", zipf=False):
    """Get ready to read stream from a file or stdin, then returns the handle."""
    if f is not None:
        guard(exists(f, "f"), f"reader, not found such a file: {f}")
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


def prompt(prompt, ok=void, fail=void):
    """Display a yes-or-no prompt and execute corresponding function
    based on user input.
    Execute lazy function `ok` if user inputs ``y`` or ``Y``.
    Execute lazy function `fail` if user inputs anything else.

    >>> prompt(
    ...     f"{model}" exist. Are you sure to proceed?",
    ...     fail=lazy(error, f"Will not overwrite {model}."),
    ... )    # doctest: +SKIP
    """
    print(f"{prompt} [y/N] ", end="", flush=True)
    c = readchar()
    print(c)
    if capture("[yY]", c):
        return ok()
    else:
        return fail()


@fx
def read_conf(f, o=True):
    """Read a conf file and return a dmap of key-value pairs.
    If ``o`` is set, evaluate values as Python literals;
    otherwise, keep as strings.

    >>> read_conf("~/.gnupg/gpg-agent.conf")  # doctest: +SKIP
    """
    read = literal_eval if o else id

    def k_v(s):
        s = s.split("#")[0].strip()
        k, *v = s.split(None, 1)
        return k, read(v[0]) if len(v) else None

    return dmap(
        k_v(s)
        for s in filter(
            cf_(lambda x: x and not x.startswith("#"), str.strip),
            reader(f).read().splitlines(),
        )
    )


@fx
def write_conf(f, conf, o=True):
    """Write a dictionary to a conf file.
    If ``o`` is set, write values as Python repr; otherwise, write as is.

    >>> write_conf("io.conf", dict(mode="loopback"), o=False)  # doctest: +SKIP
    """
    dump = repr if o else id
    f = writer(f)
    for k, v in conf.items():
        f.write(f"{k} {dump(v) if v is not None else ''}\n")


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
    """Generate cryptographically secure random bytes."""
    return os.urandom(n)


def rand(a=0, b=None, size=None):
    a, b = (0, 1) if not a and b is None else (0, a) if b is None else (a, b)
    rng = lazy(rd.uniform, a, b)
    return [rng() for _ in range(size)] if size else rng()


def randn(mu=0, sigma=1, size=None):
    return (
        [rd.gauss(mu, sigma) for _ in range(size)]
        if size is not None
        else rd.gauss(mu, sigma)
    )


def randint(a=None, b=None, size=None):
    """generate cryptographically secure random integer.
    returns random integer(s) in range of [a, b).
    """

    def r(a, b):
        guard(a < b, f"randint, low({a}) >= high({b})")
        x = b - a
        return a + (bytes_to_int(randbytes((x.bit_length() + 7) // 8)) % x)

    a, b = (0, 1 << 256) if not a and b is None else (0, a) if b is None else (a, b)
    rng = lazy(r, a, b)
    return [rng() for _ in range(size)] if size else rng()


@fx
def probify(fn, p=0.5):
    """Conditionally applies a function based on a probability, ``p``.

    Coin flipping:
    >>> probify(p=0.5)(const("H"))("T")     # doctest: +SKIP

    Russian Roulette:
    >>> probify(p=1/6)(fire_gun)("bullet")  # doctest: +SKIP
    """

    def go(*args, **kwargs):
        return (
            fn(*args, **kwargs)
            if rand() < p
            else None if not args else args[0] if len(args) == 1 else tuple(args)
        )

    return go


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
        indices = (
            randint(0, len(x), size)
            if replace or len(x) < size
            else shuffle(rangel(len(x)))[:size]
        )
        return [x[i] for i in indices]

    if not len(x):
        return x
    if p is not None:
        return fromp(x, p)
    if size is None:
        return x[randint(len(x))]
    else:
        return with_size(size)


def singleton(cls):
    """Decorate a class and make it a singleton class."""
    o = {}

    def go(*args, **kwargs):
        if cls not in o:
            o[cls] = cls(*args, **kwargs)
        return o[cls]

    return go


def thread(daemon=False):
    """Decorator factory that turns functions into threading.Thread.

    >>> mouse = thread()(mouse_listener)()  # doctest: +SKIP
    >>> mouse.start()                       # doctest: +SKIP
    >>> mouse.join()                        # doctest: +SKIP
    """

    def t(f):
        def go(*args, **kwargs):
            return threading.Thread(
                target=f,
                args=args,
                kwargs=kwargs,
                daemon=daemon,
            )

        return go

    return t


def proc(daemon=False):
    """Decorator factory that turns functions into multiprocessing.Process.

    >>> ps = [proc(True)(bruteforce)(x) for x in xs]  # doctest: +SKIP
    >>> for p in ps: p.start()                        # doctest: +SKIP
    >>> for p in ps: p.join()                         # doctest: +SKIP
    """

    def p(f):
        def go(*args, **kwargs):
            return multiprocessing.Process(
                target=f,
                args=args,
                kwargs=kwargs,
                daemon=daemon,
            )

        return go

    return p


def parmap(f, x, *xs, workers=None):
    """Parallelizes function applications over iterables
    by utilizing multiple cpu cores with ``multiprocessing.Pool``.
    In the same way as ``map``, it very simply maps a function over
    an iterable and executes it in parallel.

    >>> parmap(bruteforce, preImages)  # doctest: +SKIP
    >>> parmap(os.remove, manyFiles)   # doctest: +SKIP
    """
    workers = workers or os.cpu_count()
    with multiprocessing.Pool(workers) as pool:
        x = zip(x, *xs) if xs else x
        mapper = pool.starmap if xs else pool.map
        return mapper(f, x)


class polling:
    """Repeatedly executes a provided function at fixed time intervals.

    >>> g = f_(cf_(print, force), lazy(randint, 100))  # doctest: +SKIP
    >>> p = polling(1, g)                              # doctest: +SKIP
    >>> p.start()                                      # doctest: +SKIP
    """

    def __init__(self, sec, f, *args, **kwargs):
        self.expr = lazy(f, *args, **kwargs)
        self.timer = f_(threading.Timer, sec, self._g)
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
    """Create a countdown timer with optional message and callback.

    >>> timer(60)                                   # doctest: +SKIP
    >>> timer(60 * 25, msg="Pomodoro")              # doctest: +SKIP
    >>> timer(5, callback=lazy(randint, 1, 46, 6))  # doctest: +SKIP
    """
    guard(isinstance(t, (int, float)), f"timer, not a number: {t}")
    guard(t > 0, f"timer, must be given a positive number: {t}")
    t = int(t)
    fmt = f"{len(str(t))}d"
    while t >= 0:
        if v:
            print(f"{msg}  {t:{fmt}}", end="\r")
            writer().write("\033[K")
        time.sleep(1)
        t -= 1
    if callback:
        return callback()


def timestamp(t=None, to_utc=True, to_iso=False, decimal=0):
    """Convert time inputs to ``UTC``/local timestamp, or ``ISO`` format.

    # curruent UTC timestamp
    >>> timestamp()  # doctest: +SKIP

    # parse string to UTC
    >>> timestamp("2025-01-01 09:00:00Z")
    1735689600.0

    # ISO format with milliseconds
    >>> timestamp(1735657200.123456, to_iso=True, decimal=3)
    '2024-12-31T15:00:00.123Z'

    # using ``datetime`` object
    >>> timestamp(datetime.now(), to_utc=False)  # doctest: +SKIP
    """

    def _from(t):
        if t is None:
            return datetime.now().timestamp()
        elif isinstance(t, (int, float)):
            return float(t)
        elif isinstance(t, str):
            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        elif isinstance(t, datetime):
            return t.timestamp()
        else:
            error(f"The format is not supported: {t}")

    def convert(t):
        return datetime.utcfromtimestamp(t).timestamp() if to_utc else t

    def _to(t):
        formatted = datetime.fromtimestamp(t).isoformat()
        cut = 19 + (decimal + 1 if 0 < decimal <= 6 else 0)
        return f"{formatted[:cut]}Z" if to_iso else t

    return cf_(_to, convert, _from)(t)


def timeago(dt):
    """Convert a time difference in seconds to a human-readable string.

    >>> timeago(3600)
    '1 hour ago'
    >>> timeago(86400 * 2)
    '2 days ago'
    >>> timeago(0)
    'just now'
    """
    sec = int(dt)
    units = [
        (60 * 60 * 24 * 365, "year"),
        (60 * 60 * 24 * 30, "month"),
        (60 * 60 * 24 * 7, "week"),
        (60 * 60 * 24, "day"),
        (60 * 60, "hour"),
        (60, "minute"),
        (1, "second"),
    ]
    for unit_sec, unit in units:
        if sec >= unit_sec:
            count = sec // unit_sec
            return f"{count} {unit}{'s' if count > 1 else ''} ago"
    return "just now"


def justf(x, width, align="<", pad=" "):
    """Adjust the alignment of a given string considering wide characters.

    >>> justf("do not have love, 아무것도", 40, "^", "-")
    '-------do not have love, 아무것도-------'
    >>> justf("나에게 사랑이 없으면 I gain nothing", 40, "^", "-")
    '--나에게 사랑이 없으면 I gain nothing---'
    """

    def uni(s):
        return sum(2 if east_asian_width(c) in {"W", "F"} else 1 for c in s)

    d = max(0, width - uni(x))
    if align == "<":
        return x + pad * d
    elif align == ">":
        return pad * d + x
    elif align == "^":
        return pad * (d // 2) + x + pad * (d - d // 2)
    else:
        error(f"Error, no such text-justification: {align}")


_lock = threading.Lock()


def tracker(it, description="", total=None, start=0, barcolor="white", **kwargs):
    """Create a thread-safe progress bar with support for nested loops.

    ``tracker`` wraps ``rich.progress`` from pip's bundle to provide
    a simpler interface for creating progress bars, with column formats
    very similar to ``tqdm``.

    - Supports nested progress bars without 'additional boilerplate code'
    - Allowing use in multi-thread environments

    Refer to ``rich.progress`` as all keywords follow its conventions.

    # single progress bar
    >>> for batch in tracker(dataloader, "training"):  # doctest: +SKIP
    ...     model(batch)

    # nested progress bars
    >>> for i in tracker(range(10), "outer"):          # doctest: +SKIP
    ...     time.sleep(0.02)
    ...     for j in tracker(range(20), "inner"):
    ...         time.sleep(0.02)

    # generator with known length
    >>> g = (x for x in range(100))
    >>> for item in tracker(g, "task", total=100):     # doctest: +SKIP
    ...     process(item)
    """
    import pip._vendor.rich.progress as rp
    from pip._vendor.rich.console import Console

    class JobSpeedColumn(rp.ProgressColumn):
        def render(self, task):
            if task.speed is None:
                return rp.Text("?", style="progress.data.speed")
            return rp.Text(f"{task.speed:2.2f} it/s", style="progress.data.speed")

    @contextmanager
    def create(barcolor=barcolor, **kwargs):
        console = (
            Console(force_jupyter=True) if "get_ipython" in globals() else Console()
        )
        prog = rp.Progress(
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
            JobSpeedColumn(),
            console=console,
            **kwargs,
        )
        local.stack.append(prog)
        local.head = prog
        try:
            with prog:
                yield prog
        finally:
            local.stack.pop()
            local.head = local.stack[-1] if local.stack else None

    if not hasattr(tracker, "local"):
        tracker.local = threading.local()
    local = tracker.local

    if not hasattr(local, "stack"):
        local.stack = []
        local.head = None

    if total is None:
        length = op.length_hint(it)
        total = length if length > 0 else None

    if start:
        it = islice(it, start, None)

    if local.head is None:
        with _lock:
            with create(barcolor=barcolor, **kwargs) as tb:
                task = tb.add_task(description, total=total, completed=start)
                for item in it:
                    yield item
                    tb.advance(task, 1)
                if total is not None:
                    tb._tasks[task].completed = total
                    tb.refresh()
    else:
        tb = local.head
        task = tb.add_task(description, total=total, completed=start)
        for item in it:
            yield item
            tb.advance(task, 1)
        if total is not None:
            tb._tasks[task].completed = total
            tb.refresh()
        tb.remove_task(task)


_BASE58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58e(x):
    """Encode bytes to Base58.

    >>> base58e(b"sofia-maria-golden-girls")
    'BXNAGjq4ty8AeedspDYRnHZwFTXtyQWNe'
    """
    num = bytes_to_int(x)
    result = ""
    while num > 0:
        num, rem = divmod(num, 58)
        result = _BASE58_CHARS[rem] + result
    return result


def base58d(x):
    """Decode the Base58-encoded back to bytes.

    >>> base58d('BXNAGjq4ty8AeedspDYRnHZwFTXtyQWNe')
    b'sofia-maria-golden-girls'
    """
    num = 0
    for c in x:
        num = num * 58 + _BASE58_CHARS.index(c)
    return int_to_bytes(num)


class dataq:
    def __init__(self, x=100):
        data = _ns_builtin_iterp(x)
        self.n = op.length_hint(x) if data else x
        self.data = deque(maxlen=self.n)
        self._cache = None
        if data:
            self.update(x)

    def update(self, *args):
        self.data.extend(flat(args))
        self._cache = None
        return self

    def nan(f):
        def wrapper(self, *args, **kwargs):
            if not self.data:
                return float("nan")
            return f(self, *args, **kwargs)

        return wrapper

    def __bool__(self):
        return bool(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        if isinstance(x, slice):
            return self.cache[x]
        return self.data.__getitem__(x)

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.array(self.data)
        return self._cache

    @property
    def size(self):
        return self.data.maxlen

    @nan
    def percentile(self, q):
        return np.percentile(self.cache, q)

    @nan
    def quantile(self, q):
        return np.quantile(self.cache, q)

    @property
    def q1(self):
        return np.percentile(self.cache, 25)

    @property
    @nan
    def median(self):
        return np.median(self.cache)

    @property
    def q3(self):
        return np.percentile(self.cache, 75)

    @property
    def iqr(self):
        return self.q3 - self.q1

    @property
    @nan
    def quartile(self):
        return np.percentile(self.cache, [25, 50, 75])

    @property
    @nan
    def mad(self):
        return np.median(np.abs(self.cache - self.median))

    @property
    @nan
    def mean(self):
        return np.mean(self.cache)

    @property
    @nan
    def var(self):
        return np.var(self.cache)

    @property
    @nan
    def std(self):
        return np.std(self.cache)

    @property
    @nan
    def muad(self):
        return np.mean(np.abs(self.cache - self.mean))

    @property
    @nan
    def cv(self):
        return self.std / self.mean

    @property
    @nan
    def hmean(self):
        if np.any(self.cache < 0):
            return float("nan")
        return len(self) / np.sum(1 / self.cache)

    @property
    @nan
    def gmean(self):
        if np.any(self.cache < 0):
            return float("nan")
        return np.exp(np.mean(np.log(self.cache)))

    @property
    @nan
    def skew(self):
        return np.mean((self.cache - self.mean) ** 3) / (self.std**3)

    @property
    @nan
    def kurtosis(self):
        return np.mean((self.cache - self.mean) ** 4) / (self.std**4) - 3

    @property
    @nan
    def min(self):
        return np.min(self.cache)

    @property
    @nan
    def max(self):
        return np.max(self.cache)

    @property
    @nan
    def minmax(self):
        return self.min, self.max

    @property
    def sort(self):
        return np.sort(self.cache)

    @property
    def sum(self):
        return np.sum(self.cache)

    @property
    def cumsum(self):
        return np.cumsum(self.sort)

    @property
    def prod(self):
        return np.prod(self.cache)

    @property
    def cumprod(self):
        return np.cumprod(self.sort)

    def describe(self, as_dict=False):
        return dmap(
            count=len(self),
            min=self.min,
            max=self.max,
            mean=self.mean,
            var=self.var,
            std=self.std,
            median=self.median,
            q1=self.q1,
            q3=self.q3,
            iqr=self.iqr,
            mad=self.mad,
            skewness=self.skew,
            kurtosis=self.kurtosis,
            gmean=self.gmean,
            hmean=self.hmean,
            CV=self.cv,
        ) | (id if as_dict else pp(sort=False))
