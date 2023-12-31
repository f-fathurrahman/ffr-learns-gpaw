"""Generate new release of ASE.

This script does not attempt to import ASE - then it would depend on
which ASE is installed and how - but assumes that it is run from the
ASE root directory."""

import subprocess
import re
import argparse
from time import strftime


def runcmd(cmd, output=False, error_ok=False):
    print('Executing:', cmd)
    try:
        if output:
            txt = subprocess.check_output(cmd, shell=True)
            return txt.decode('utf8')
        else:
            return subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as err:
        if error_ok:
            print('Failed: {}'.format(err))
            print('Continuing...')
        else:
            raise


bash = runcmd


def py(cmd, output=False):
    return runcmd('python3 {}'.format(cmd))


def py2(cmd, output=False):
    return runcmd('python2 {}'.format(cmd))


def git(cmd, error_ok=False):
    cmd = 'git {}'.format(cmd)
    return runcmd(cmd, output=True, error_ok=error_ok)


versionfile = 'gpaw/__init__.py'


def get_version():
    with open(versionfile) as fd:
        return re.search(r"__version__ = '(\S+)'", fd.read()).group(1)


def main():
    p = argparse.ArgumentParser(description='Generate new release of GPAW.',
                                epilog='Run from the root directory of GPAW.')
    p.add_argument('version', nargs='?',
                   help='new version number')
    p.add_argument('--clean', action='store_true',
                   help='delete release branch and tag')
    p.add_argument('--sign', action='store_true')
    p.add_argument('--wheel', action='store_true')
    args = p.parse_args()

    try:
        current_version = get_version()
    except Exception as err:
        p.error('Cannot get version: {}.  Are you in the root directory?'
                .format(err))

    print('Current version: {}'.format(current_version))

    if not args.version:
        p.print_help()
        raise SystemExit

    version = args.version

    branchname = 'gpaw-{}'.format(version)
    current_version = get_version()

    if args.clean:
        print('Cleaning {}'.format(version))
        git('checkout master')
        git('tag -d {}'.format(version), error_ok=True)
        git('branch -D {}'.format(branchname), error_ok=True)
        git('branch -D {}'.format('web-page'), error_ok=True)
        return

    print('New release: {}'.format(version))

    txt = git('status')
    branch = re.match(r'On branch (\S+)', txt).group(1)
    print('Currently on branch {}'.format(repr(branch)))
    if branch != 'master':
        git('checkout master')

    git('checkout -b {}'.format(branchname))

    majormiddle, minor = version.rsplit('.', 1)
    minor = int(minor)
    nextminor = minor + 1
    next_devel_version = '{}.{}b1'.format(majormiddle, nextminor)

    def update_version(version):
        print('Editing {}: version {}'.format(versionfile, version))
        new_versionline = "__version__ = '{}'\n".format(version)
        lines = []
        ok = False
        with open(versionfile) as fd:
            for line in fd:
                if line.startswith('__version__'):
                    ok = True
                    line = new_versionline
                lines.append(line)
        assert ok
        with open(versionfile, 'w') as fd:
            for line in lines:
                fd.write(line)

    update_version(version)

    releasenotes = 'doc/releasenotes.rst'
    lines = []

    searchtxt = re.escape("""\
Git master branch
=================

:git:`master <>`.
""")

    replacetxt = """\
Git master branch
=================

:git:`master <>`.

* No changes yet


{header}
{underline}

{date}: :git:`{version} <../{version}>`
"""

    date = strftime('%d %B %Y').lstrip('0')
    header = 'Version {}'.format(version)
    underline = '=' * len(header)
    replacetxt = replacetxt.format(header=header, version=version,
                                   underline=underline, date=date)

    print('Editing {}'.format(releasenotes))
    with open(releasenotes) as fd:
        txt = fd.read()
    txt, n = re.subn(searchtxt, replacetxt, txt, re.MULTILINE)
    assert n == 1

    with open(releasenotes, 'w') as fd:
        fd.write(txt)

    searchtxt = """\
News
====
"""

    replacetxt = """\
News
====

* :ref:`GPAW version {version} <releasenotes>` released ({date}).
"""

    replacetxt = replacetxt.format(version=version, date=date)

    frontpage = 'doc/index.rst'
    lines = []
    print('Editing {}'.format(frontpage))
    with open(frontpage) as fd:
        txt = fd.read()
    txt, n = re.subn(searchtxt, replacetxt, txt)
    assert n == 1
    with open(frontpage, 'w') as fd:
        fd.write(txt)

    installdoc = 'doc/install.rst'
    print('Editing {}'.format(installdoc))

    with open(installdoc) as fd:
        txt = fd.read()

    txt, nsub = re.subn(r'gpaw-\d+\.\d+.\d+',
                        'gpaw-{}'.format(version), txt)
    assert nsub > 0
    txt, nsub = re.subn(r'git clone -b \d+\.\d+.\d+',
                        'git clone -b {}'.format(version), txt)
    assert nsub == 1

    with open(installdoc, 'w') as fd:
        fd.write(txt)

    sphinxconf = 'doc/conf.py'
    print('Editing {}'.format(sphinxconf))
    comment = '# This line auto-edited by newrelease script'
    line1 = "dev_version = '{}'  {}\n".format(next_devel_version, comment)
    line2 = "stable_version = '{}'  {}\n".format(version, comment)
    lines = []
    with open(sphinxconf) as fd:
        for line in fd:
            if re.match('dev_version = ', line):
                line = line1
            if re.match('stable_version = ', line):
                line = line2
            lines.append(line)
    with open(sphinxconf, 'w') as fd:
        fd.write(''.join(lines))

    git('add {}'.format(' '.join([versionfile, sphinxconf, installdoc,
                                  frontpage, releasenotes])))
    git('commit -m "GPAW version {}"'.format(version))
    git('tag {0} {1} -m "gpaw-{1}"'
        .format('-s' if args.sign else '',
                version))

    py('setup.py sdist > setup_sdist.log')
    if args.wheel:
        py2('setup.py bdist_wheel > setup_bdist_wheel2.log')
        py('setup.py bdist_wheel > setup_bdist_wheel3.log')
    if args.sign:
        bash('gpg --armor --yes --detach-sign dist/gpaw-{}.tar.gz'
             .format(version))
    git('checkout -b web-page')
    git('branch --set-upstream-to=origin/web-page')
    git('checkout {}'.format(branchname))
    update_version(next_devel_version)
    git('add {}'.format(versionfile))
    git('branch --set-upstream-to=master')
    git('commit -m "bump version number to {}"'.format(next_devel_version))

    print()
    print('Automatic steps done.')
    print()
    print('Now is a good time to:')
    print(' * check the diff')
    print(' * run the tests')
    print(' * verify the web-page build')
    print()
    print('Remaining steps')
    print('===============')
    print('git show {}  # Inspect!'.format(version))
    print('git checkout master')
    print('git merge {}'.format(branchname))
    print('twine upload '
          'dist/gpaw-{v}.tar.gz '
          'dist/gpaw-{v}-py2-none-any.whl '
          'dist/gpaw-{v}-py3-none-any.whl '
          'dist/gpaw-{v}.tar.gz.asc'.format(v=version))
    print('git push --tags origin master  # Assuming your remote is "origin"')
    print('git checkout web-page')
    print('git push --force origin web-page')


if __name__ == '__main__':
    main()
