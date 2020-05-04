import os
import collections.abc
import itertools

# https://codereview.stackexchange.com/questions/173045/mutable-named-tuple-or-slotted-data-structure
class MutableNamedTuple(collections.abc.Sequence): 
    """Abstract Base Class for objects as efficient as mutable
    namedtuples. 
    Subclass and define your named fields with __slots__.
    """
    __slots__ = ()
    def __init__(self, *args):
        for slot, arg in zip(self.__slots__, args):
            setattr(self, slot, arg)
    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))
    # more direct __iter__ than Sequence's
    def __iter__(self): 
        for name in self.__slots__:
            yield getattr(self, name)
    # Sequence requires __getitem__ & __len__:
    def __getitem__(self, index):
        return getattr(self, self.__slots__[index])
    def __len__(self):
        return len(self.__slots__)
# ---------------------------------------------------

def mod_date(path):
    if os.path.exists(path):
        return os.path.getmtime(path)
    else:
        return None

class Rule(MutableNamedTuple):
    __slots__ = ('inputs', 'outputs', 'action', 'precious')

#Rule = collections.namedtuple('Rule', ('inputs', 'outputs', 'action', 'precious'))

class Makefile(object):

    def __init__(self):
        self.rules = dict()    # {target : rule}

    def add(self, action, inputs, outputs):
        rule = Rule(inputs, outputs, action, False)
        if action is not None:
            # Dummy rules aren't added to the dependency DAG
            for output in outputs:
                self.rules[output] = rule
        return rule

class build(object):
    def __init__(self, makefile, targets):
        self.makefile = makefile
        self.dates = dict()
        self.made = set()
        self._get_dates(targets)
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
                    rule.action()

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

        # Remove our targets
        for target in targets:
            del self.all_files[target]

        # Cull to only files that stil exist
        made_files = [x for x in self.all_files if os.path.exists(x)]

        # Nothing to do!
        if len(made_files) == 0:
            print('No temporary files to remove')
            return

        # Files to delete
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
