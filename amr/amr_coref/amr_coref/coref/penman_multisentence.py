#!/usr/bin/python3
import re
import logging
from   copy import deepcopy
import penman

logger = logging.getLogger(__name__)


# Check to see if a graph is a multi-sentence graph
def is_multi_sentence(pgraph):
    if pgraph.top == 'm':
        instances = [t for t in pgraph.instances() if t.source=='m']
        assert len(instances) == 1
        if instances[0].target == 'multi-sentence':
            return True
    return False


# Split multi-sentence graphs into a separate graph for each :sntx
# This code does uses string manipulation to do this. It could also be done by traversing the graph.
# However, there are some graphs with where nodes fron one :snt reference a node in a previous
# :snt.  These are usually "i" or "you" nodes. See DF-200-192400-625_7046.11 in dfa.txt
re_attrib = re.compile(r'^[a-z]\d*$')
def split_multi_sentence(pgraph):
    # Get the graph string and variable to concept dictionary
    pgraph = deepcopy(pgraph)
    gid = pgraph.metadata.get('id', 'none') # for logging
    pgraph.metadata = {}
    var2concept = {t.source:t.target for t in pgraph.instances()}
    gstring = penman.encode(pgraph, indent=0)
    # delete the multi-sentence line and any modifiers like (:li, :mode)
    glines  = gstring.split('\n')
    assert glines[0].startswith('(m / multi-sentence')
    glines  = glines[1:]
    while glines:
        if glines[0].startswith(':') and not glines[0].startswith(':snt'):
            glines = glines[1:]
        else:
            break
    # rejoin the lines remove extra spaces and remove ending paren
    gstring = ' '.join(glines)
    gstring = re.sub(r' +', ' ', gstring).strip()
    assert gstring.endswith(')')
    gstring = gstring[:-1]
    # Split on the :snt lines and separate each sentence to its own graph
    gs_list = [gs.strip() for gs in re.split(':snt\d+', gstring)]
    gs_list = [gs for gs in gs_list if gs]
    # Convert the separated graphs to penman objects
    pgraphs = []
    for gidx, gstring in enumerate(gs_list):
        try:
            pgraph = penman.decode(gstring)
        except penman.DecodeError:
            logger.error('Error decoding %s %d\n%s' % (gid, gidx, gstring))
            continue
        # If a variable is not in this portion of the graph then penman will treat it like an
        # attribute.  In this case we need to add an instance for it.  The way penman 1.1.0
        # works, this will fix the graph.
        missing_set = set(t.target for t in pgraph.attributes() if re_attrib.match(t.target))
        if missing_set:
            logger.info('%s %d missing variables: %s' % (gid, gidx, str(missing_set)))
        # Add the variables and re-decode the graph
        for var in missing_set:
            concept = var2concept.get(var, None)
            if concept is not None:
                pgraph.triples.append((var, ':instance', concept))
        pgraphs.append(pgraph)
    return pgraphs
