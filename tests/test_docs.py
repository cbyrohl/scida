import pathlib

import pytest

# unfortunately cannot use existing solution for pycon code blocks, so have to write our own
# (https://github.com/nschloe/pytest-codeblocks/issues/77)

ignore_files = ["largedatasets.md"]  # , "visualization.md"]


class DocFile:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.lines = []
        self.codeblocks = []
        self.extract_codeblocks()

    def extract_codeblocks(self):
        with open(self.path, "r") as f:
            lines = f.readlines()
        # check line for codeblock start/end
        lidx = [i for i, k in enumerate(lines) if k.startswith("```")]
        print(lidx)
        # assert that there are an even number of codeblock start/end
        assert len(lidx) % 2 == 0
        cblines = []
        for i in range(0, len(lidx), 2):
            start = lidx[i]  # includes codeblock start line
            end = lidx[i + 1]
            # we check for the type. if its 'pycon', we need to remove the '>>>' prompt
            blocktype = lines[start].strip().replace("```", "").strip()
            blocktype = blocktype.split()[0].strip()
            if blocktype == "py" or blocktype == "python":
                cblines.append(lines[start + 1 : end])
            elif blocktype == "pycon":
                cblines.append(
                    [k[4:] for k in lines[start + 1 : end] if k.startswith(">>>")]
                )
            elif blocktype in ["bash", "yaml", "json", "console", "text", "html"]:
                # not python; ignore
                pass
            else:
                raise ValueError("Unknown codeblock type: %s" % blocktype)
        self.codeblocks = ["".join(cbl) for cbl in cblines]

    def replace_line(self, str, newstr, searchmode="match", replacemode="line"):
        codeblocks_new = []
        for codeblock in self.codeblocks:
            lines = codeblock.split("\n")
            if searchmode == "match":
                idx = [i for i, k in enumerate(lines) if k == str]
            elif searchmode == "startswith":
                idx = [i for i, k in enumerate(lines) if k.startswith(str)]
            elif searchmode == "contains":
                idx = [i for i, k in enumerate(lines) if str in k]
            else:
                raise ValueError("Unknown searchmode: %s" % searchmode)
            # now replace
            for i in idx:
                if replacemode == "line":
                    lines[i] = newstr
                elif replacemode == "onlystring":
                    lines[i] = lines[i].replace(str, newstr)
                else:
                    raise ValueError("Unknown replacemode: %s" % replacemode)
            codeblock = "\n".join(lines)
            codeblocks_new.append(codeblock)
        self.codeblocks = codeblocks_new

    def evaluate(self):
        # evaluate all at once (for now)
        code = "\n".join(self.codeblocks)
        print("Evaluating code:")
        print('"""' + code + '"""')
        exec(code, globals())  # need to use global() to allow using imports


def get_docfiles():
    fixturename = "mddocfile"
    docfiles = []
    paths = []
    ids = []

    docpath = pathlib.Path(__file__).parent.parent / "docs"
    for p in docpath.glob("*.md"):
        name = p.name
        if name in ignore_files:
            continue
        print("Evaluating %s" % p)
        # read lines
        docfile = DocFile(p)
        docfiles.append(docfile)
        paths.append(p)
        ids.append(name)

    params = docfiles
    return pytest.mark.parametrize(fixturename, params, ids=ids)


@pytest.mark.xfail
@get_docfiles()
def test_docs(mddocfile):
    mddocfile.replace_line(
        'load("./snapdir_030")',
        'load("TNG50-4_snapshot")',
        searchmode="contains",
        replacemode="onlystring",
    )  # workaround
    mddocfile.evaluate()
