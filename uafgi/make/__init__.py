import os
import itertools
from uafgi import ioutil,gicollections

# ---------------------------------------------------

def mod_date(path):
    if os.path.exists(path):
        return os.path.getmtime(path)
    else:
        return None

class Rule(gicollections.MutableNamedTuple):
    __slots__ = ('action', 'inputs', 'outputs', 'precious')

    # action() can be called like a function

    def __call__(self):
        """Useful for one-off calls of a rule, outside of a Makefile"""
        with ioutil.TmpDir() as tdir:
            self.action(tdir)

#Rule = collections.namedtuple('Rule', ('inputs', 'outputs', 'action', 'precious'))

class Makefile(object):

    def __init__(self):
        self.rules = dict()    # {target : rule}

#    def add(self, action, inputs, outputs):
#        rule = Rule(inputs, outputs, action, False)
#        if action is not None:
#            # Dummy rules aren't added to the dependency DAG
#            for output in outputs:
#                self.rules[output] = rule
#        return rule

    def add(self, rule):
        for output in rule.outputs:
            self.rules[output] = rule
        return rule.outputs

    def format(self):
        """Converts to a (very long) string"""

        # uniq-ify the rules, keep the same order
        uniq_rules = dict()
        for rule in self.rules.values():
            if id(rule) not in uniq_rules:
                uniq_rules[id(rule)] = rule
        # Scan...
        out = list()
        for rule in uniq_rules.values():
            out.append('========================== {}'.format(rule.action))
            for file in rule.inputs:
                out.append('  I {}'.format(file))
            for file in rule.outputs:
                out.append('  O {}'.format(file))

        return '\n'.join(out)

class build(object):
    def __init__(self, makefile, targets, tdir_fn=ioutil.TmpDir):
        """Call this to create an ioutil.TmpDir"""
        self.makefile = makefile
        self.dates = dict()
        self.made = set()
        self._get_dates(targets)
        self.tdir_fn = tdir_fn
        self._build(targets)

    def _get_dates(self, targets):
        """
        dates: {target : (date, max_sub)}
        """
        target_dates = list()
        for target in targets:
            # Cut off repeated searching of a DAG
            if target in self.dates:
                target_dates.append(self.dates[target])
                continue

            # Slightly different logic for leaves (which must be there)
            if target in self.makefile.rules:
                rule = self.makefile.rules[target]
                input_dates = self._get_dates(rule.inputs)  # [(date,max_sub), ...]
                max_input_date = max(x for x in itertools.chain.from_iterable(input_dates) if x is not None)

            else:
                max_input_date = None

            dt = (mod_date(target), max_input_date)
            self.dates[target] = dt
            target_dates.append(dt)

        return target_dates

    def _build(self, targets):
        """tdir_kwargs:
            kwargs needed to create a new temporary directory (one per rule)
        """
        for target in targets:
            # Maybe we made it on a previous iteration around this loop
            if target in self.made:
                continue

            # Decide whether this needs to be made
            date,max_sub = self.dates[target]

            # Leaf in DAG
            if max_sub is None:
                if os.path.exists(target):
                    self.made.add(target)
                else:
                    raise ValueError('No rule to make {}, and it does not exist'.format(target))
            else:
                if (date is None) or (date < max_sub):
                    # This needs to be remade
                    rule = self.makefile.rules[target]
                    self._build(rule.inputs)
                    print('========================== Building {} {}'.format(rule.outputs[0], rule.action))
                    with self.tdir_fn() as tdir:    # See ioutil.TmpDir
                        rule.action(tdir)

                    # Add to the set of things we've made
                    self.made.update(rule.outputs)


def yes_or_no(question):
    while True:
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False

class cleanup(object):
    def __init__(self, makefile, targets):
        self.makefile = makefile
#        self.all_files = set()
        self.all_files = dict()    # Really need an OrderedSet
        self._get_all_files(targets)

        # Remove our targets from set of files to remove
        for target in targets:
            del self.all_files[target]

        # Cull to only files that still exist
        made_files = [x for x in self.all_files if os.path.exists(x)]

        # Nothing to do!
        if len(made_files) == 0:
            print('No temporary files to remove')
            return

        # Files to delete
        print('============= Permanent files to keep:')
        print('\n'.join(targets))
        print('============= Temporary files to remove:')
        print('\n'.join(made_files))
        if yes_or_no('Remove these files?'):
            for path in made_files:
                os.remove(path)
        

    def _get_all_files(self, targets):
        """Lists all files involved in a make"""
        for target in targets:

            # Never list primary files as able to be cleaned up
            if target not in self.makefile.rules:
                continue

            # Maybe we made it on a previous iteration around this loop
            if target in self.all_files:
                continue

            # Add this to our list of files
#            self.all_files.add(target)
            self.all_files[target] = None

            # Recurse
            rule = self.makefile.rules[target]
            self._get_all_files(rule.inputs)

def opath(ipath, odir, suffix, replace=None):
    """Converts from [idir]/[ipath][ext] to [odir]/[opath][suffix][ext]
    ipath:
        Full pathname of input file
    odir:
        Directory where to place output file
    """
    idir,ileaf = os.path.split(ipath)
    iroot,iext = os.path.splitext(ileaf)
    if replace is None:
        leaf = '{}{}{}'.format(iroot,suffix,iext)
    else:
        leaf = '{}{}'.format(iroot.replace(replace, suffix), iext)
    return os.path.join(odir, leaf)
